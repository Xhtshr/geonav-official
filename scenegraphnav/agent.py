import re
import os
import cv2
import json
import numpy as np
from skimage import color
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.space import Pose4D, Point3D, Point2D
from gsamllavanav.dataset.episode import Episode
from scenegraphnav.llm_controller import LLMController # only used for our scenegraphnav methods
from scenegraphnav.evaluate import move


from openai import OpenAI
from ggb.QwenAPI import encode_image_from_pil
from PIL import Image

from scenegraphnav.prompt.navgpt import *
from scenegraphnav.prompt.geonav_cot import *
from gsamllavanav.actions import DiscreteAction
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from datetime import datetime
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap

def rgb_entropy(rgb_img):
    hsv_img = color.rgb2hsv(rgb_img)
    entropy = 0
    for i in range(3):
        channer = (hsv_img[..., i]*255).astype(np.uint8)
        hist = cv2.calcHist([channer], [0], None, [256], [0, 256])
        hist = hist / hist.sum() + 1e-10
        entropy += -np.sum(hist * np.log2(hist))
        return entropy

def structural_similarity(current, previous):
    """计算结构相似性指标"""
    from skimage.metrics import structural_similarity as ssim
    return ssim(current, previous, multichannel=True, win_size=3)

class ExplorationAnalyzer:
    def __init__(self, threshold=0.08):
        self.buffer = []
        self.threshold = threshold
        self.history_entropy = []
        self.grid_size = 20  # 将地图划分为20x20网格
        
    def _grid_analysis(self, rgb_map):
        """网格级信息变化检测"""
        height, width = rgb_map.shape[:2]
        cell_h = height // self.grid_size
        cell_w = width // self.grid_size
        
        grid_entropy = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = rgb_map[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                grid_entropy[i,j] = rgb_entropy(cell)
        
        return grid_entropy.std()  # 返回网格熵值的标准差

    def should_continue(self, current_map, previous_map):
        # 信息熵差异
        current_e = rgb_entropy(current_map)
        previous_e = rgb_entropy(previous_map)
        delta_e = abs(current_e - previous_e)
        print(f"Delta Entropy: {delta_e}")
        
        # # 结构相似性
        # ssim_score = structural_similarity(current_map, previous_map)
        
        # 空间分布变化
        spatial_variation = self._grid_analysis(current_map - previous_map)
        
        # 综合决策（可调节权重）
        decision_score = delta_e #+ 0.3*(1-ssim_score) + 0.2*spatial_variation
        
        self.history_entropy.append(decision_score)
        return decision_score > self.threshold

def extract_json_from_msg(msg):
    """
    从包含JSON代码块的文本中提取并解析JSON数据
    
    参数：
    msg (str): 包含JSON代码块的原始文本
    
    返回：
    dict: 解析后的JSON字典，未找到返回None，解析失败返回None
    """
    # 匹配 ```json 包裹的JSON内容（支持多行匹配）
    pattern = r'```json\s*(.*?)\s*```'
    match = re.search(pattern, msg, re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None
def String2DisActionList(action_str):
    action_map = {
        "STOP": DiscreteAction.STOP,
        "MOVE_FORWARD": DiscreteAction.MOVE_FORWARD,
        "TURN_RIGHT": DiscreteAction.TURN_RIGHT,
        "TURN_LEFT": DiscreteAction.TURN_LEFT,
        "GO_UP": DiscreteAction.GO_UP,
        "GO_DOWN": DiscreteAction.GO_DOWN,
    }

    # 提取字符串中所有匹配的动作名称（忽略大小写）
    matches = re.findall(r'\b(STOP|MOVE_FORWARD|TURN_RIGHT|TURN_LEFT|GO_UP|GO_DOWN)\b', action_str, re.IGNORECASE)

    # 将匹配的动作名称映射为对应的离散动作，并返回列表
    return [action_map[match.upper()] for match in matches]

class Agent(object):
    ''' Base class for an agent to generate and save trajectories. '''
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode):
        self.name = None
        self.memory = []
        self.args = args
        self.pose = initial_pose
        self.episode = episode  # 存储episode信息
    
    def parse_instruction(self, instruction: str):
        # 解析任务描述，提取关键信息
        from prompt.instruction import create_prompt, gpt_api_call, parse_response
        prompt = create_prompt(instruction)
        response = gpt_api_call(prompt)
        self.parsed_instruction = parse_response(response)

    def set_target(self, target: Point2D):
        # 设置Agent的目标位置（可以是landmark坐标，也可以是CV模型提取出的waypoint位置）
        self.target = target

    def run(self):
        raise NotImplementedError
    
    def get_results(self):
        # save trajectories and images
        raise NotImplementedError

class ChatAgent(Agent):
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode, vlmodel, set_height=None):
        super().__init__(args, initial_pose, episode)
        self.episode = episode
        self.target = self.episode.target_position
        self.set_height = set_height  # 保存set_height参数
        if set_height is not None:
            initial_pose.with_z(set_height)
        self.controller = LLMController(args, initial_pose)
        self.model = vlmodel
        #  local model
        if isinstance(self.model, Qwen2VLForConditionalGeneration):
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                "/data1/FoundationModels/Qwen", min_pixels=min_pixels, max_pixels=max_pixels
            )
        
        self.prompts = self.init_prompts()
        self.history = ''
        self.previous_action = ''
        self.results = {
            "target": (self.target.x, self.target.y),
            "steps": []
        }
        self.landmark_nav_map = LandmarkNavMap(
            episode.map_name, args.map_shape, args.map_pixels_per_meter, 
            episode.description_landmarks, episode.description_target, episode.description_surroundings, args.gsam_params, id=episode.id
        )

    def get_next_position(self, target_json: dict):
        if target_json['decision'] == 'Explore':
            direction = target_json['navigation_recommendation'].lower()
            
            if 'northwest' in direction:
                next_pos = (self.controller.pose.x - self.view_width/5, self.controller.pose.y + self.view_width/5)
            elif 'northeast' in direction:
                next_pos = (self.controller.pose.x + self.view_width/5, self.controller.pose.y + self.view_width/5)
            elif 'southwest' in direction:
                next_pos = (self.controller.pose.x - self.view_width/5, self.controller.pose.y - self.view_width/5)
            elif 'southeast' in direction:
                next_pos = (self.controller.pose.x + self.view_width/5, self.controller.pose.y - self.view_width/5)
            elif 'south' in direction:
                next_pos = (self.controller.pose.x, self.controller.pose.y - self.view_width/2)
            elif 'north' in direction:
                next_pos = (self.controller.pose.x, self.controller.pose.y + self.view_width/2)
            elif 'west' in direction:
                next_pos = (self.controller.pose.x - self.view_width/2, self.controller.pose.y)
            elif 'east' in direction:
                next_pos = (self.controller.pose.x + self.view_width/2, self.controller.pose.y)
            else:
                next_pos = (self.controller.pose.x, self.controller.pose.y)
        elif target_json['decision'] == 'Locate':
            next_pos = target_json['selected_pos']
        else:
            raise ValueError('Invalid decision')
        return next_pos


    def init_prompts(self):
        """
        初始化所有任务相关的 Prompt 模板。
        """
        return {
            "task_description": TASK_DESCRIPTION_PROMPT,
            "action_prompt": ACTION_PROMPT,
            "planner_prompt": PLANNER_PROMPT,
            "history_prompt": HISTORY_PROMPT,
            "observation_prompt": OBSERVATION_SUMMARY,
            "object_caption": OBSERVATION_CAPTION,
            "object_grounding":OBJECT_GROUNDING,
        }
    
    def call_response(self, sysprompt, userprompt, image_64):
        if image_64 and sysprompt:
            rep = self.model.chat.completions.create(
                    model="qwen-vl-max-latest",
                    messages=[
                        {"role": "system", "content": sysprompt},
                        {"role": "user", "content": [
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url":f"data:image/png;base64,{image_64}"
                                }
                            },
                            {
                                "type": "text", 
                                "text": userprompt
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
        elif userprompt:
            rep = self.model.chat.completions.create(
                    model="qwen-vl-max-latest",
                    messages=[
                        {"role": "user", "content": userprompt}
                    ],
                    max_tokens=300
                )
        return rep.choices[0].message.content.strip()

    def run(self, naive=False):
        t = 0
        Success = False
        pos_log = []
        while t < self.args.eval_max_timestep:
            t += 1
            print(f"Step {t}")
            rgb, _ = self.controller.perceive(self.controller.pose, self.episode.map_name)
            rgb_img = Image.fromarray(rgb)

            # dep_img = Image.fromarray(depth.squeeze(), mode='L')  # 'L'表示灰度模式
            # measure the z distance between the camera and the target on the ground
            image_64 = encode_image_from_pil(rgb_img)
            # detect and memory the scene objects
            detect_mode = None #'GSAM'
            if detect_mode == 'VLM':
                detections = self.controller.understand(image_64, None, self.episode)
                # Graph memory
                self.controller.build_scene_graph(detections, self.landmark_nav_map.landmark_map.landmarks)
            elif detect_mode == 'GSAM':
                # 从地图中获取检测结果
                detections = self.landmark_nav_map.target_map.obj_list + self.landmark_nav_map.surroundings_map.obj_list
                # memory
                self.controller.build_scene_nodes(detections, self.landmark_nav_map.landmark_map.landmarks)
            else:
                _, landmark = self.controller.build_scene_nodes([], self.landmark_nav_map.landmark_map.landmarks, show=False)
            
            if isinstance(self.model, Qwen2VLForConditionalGeneration):
                raise NotImplementedError
            elif isinstance(self.model, OpenAI):
                # Step 1: system Prompt
                if naive:
                    landmark = ''
                task_prompt = self.prompts["task_description"].format(instruction=self.episode.target_description, surroundings = landmark)
                self.view_width = 2 * (self.controller.pose.z-self.landmark_nav_map.ground_level)

                # Step 2：# 产生关于环境的描述

                xyxy = [(self.controller.pose.x -self.view_width/2, self.controller.pose.y -self.view_width/2),(self.controller.pose.x +self.view_width/2, self.controller.pose.y +self.view_width/2)]
                observation_prompt = self.prompts["observation_prompt"].format(pos=self.controller.pose.xy, shape=rgb.shape[:2], area=xyxy)
                
                self.observation = self.call_response(task_prompt, observation_prompt, image_64)
                print(f"observation: {self.observation}")
                # step 3: Plan to explore or locate
                target_json = extract_json_from_msg(self.observation)
                if target_json is None:
                    continue
                next_pos = self.get_next_position(target_json)
                # Step 4: 生成动作
                self.controller.pose = Pose4D(next_pos[0],next_pos[1],self.controller.pose.z, self.controller.pose.yaw)#move(self.controller.pose, Point2D(next_pos[0], next_pos[1]), 10, _)


            # 记录当前步骤的信息
            self.results["steps"].append({
                "time_step": t,
                "pose": (self.controller.pose.x, self.controller.pose.y, self.controller.pose.z, self.controller.pose.yaw),
                "distance_to_target": self.controller.pose.xy.dist_to(self.target.xy),
                "observation_suggestion":self.observation
            })

            # 记录当前位置
            pos_log.append(self.controller.pose)
            if self.controller.reached_target(self.controller.pose, self.target):
                print("Target reached.")
                Success = True

        self.save_results(Success, naive)
        return Success, pos_log


    def save_results(self, success, naive=False):
        # 生成唯一的文件名，包含set_height信息
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        height_info = f"height_{self.set_height}" if self.set_height is not None else "default_height"
        filename = f"ChatAgent_{self.episode.id}_{height_info}_{timestamp}.json"
        if naive:
            filepath = os.path.join("results/qwen/naive", filename)
        else:
            filepath = os.path.join("results/qwen", filename)

        # 确保结果目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 添加成功状态到结果
        self.results["success"] = success

        # 保存结果到文件
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)

        print(f"Results saved to {filepath}")

        """
        使用 VLM 模型推理下一步 UAV 动作。
        """
        # 生成 VLM 输出
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        # 解码生成的动作
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(f"Generated Suggestion: {output_text}")
        return output_text[0]  # 假设返回单步动作




class SceneAgent(Agent):
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode, vlmodel, set_height=None):
        super().__init__(args, initial_pose, episode)
        self.episode = episode
        self.target = self.episode.target_position
        self.set_height = set_height  # 保存set_height参数
        if set_height is not None:
            initial_pose.with_z(set_height)
        self.controller = LLMController(args, initial_pose)
        self.model = vlmodel
        #  local model
        if isinstance(self.model, Qwen2VLForConditionalGeneration):
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                "/data1/FoundationModels/Qwen", min_pixels=min_pixels, max_pixels=max_pixels
            )
        
        self.prompts = self.init_prompts()
        self.history = ''
        self.previous_action = ''
        self.results = {
            "target": (self.target.x, self.target.y),
            "steps": []
        }
        # Parse the instruction
        self.task_prior_knowledge = self.controller.parse_instruction(self.episode.target_description)
        self.landmark_nav_map = LandmarkNavMap(
            episode.map_name, args.map_shape, args.map_pixels_per_meter, 
            episode.description_landmarks, self.task_prior_knowledge["Target"]["class"], self.task_prior_knowledge["Surrounding"], args.gsam_params, id=episode.id
        )

    def init_prompts(self):
        """
        初始化所有任务相关的 Prompt 模板。
        """
        return {
            "task_description": TASK_DESCRIPTION_PROMPT,
            "action_prompt": ACTION_PROMPT,
            "planner_prompt": PLANNER_PROMPT,
            "history_prompt": HISTORY_PROMPT,
            "observation_prompt": OBSERVATION_SUMMARY,
            "object_caption": OBSERVATION_CAPTION,
            "object_grounding":OBJECT_GROUNDING,
            "map_prompt": MAP_SUMMARY,
        }
    def call_response(self, sysprompt, userprompt, image_64):
        if image_64 and sysprompt:
            rep = self.model.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": sysprompt},
                        {"role": "user", "content": [
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url":f"data:image/png;base64,{image_64}"
                                }
                            },
                            {
                                "type": "text", 
                                "text": userprompt
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
        elif userprompt:
            rep = self.model.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": userprompt}
                    ],
                    max_tokens=300
                )
        return rep.choices[0].message.content.strip()
    
    def get_next_position(self, target_json: dict, semap_img, task_prompt):
        if target_json['decision'] == 'Explore':
            self.action = self.call_response(sysprompt=task_prompt, userprompt=self.prompts["map_prompt"], image_64=semap_img)
            print(self.action)

            target_json = extract_json_from_msg(self.action)
            if target_json is None:
                return (self.controller.pose.x, self.controller.pose.y)
            elif target_json['decision'] == 'Exploit':
                print('The target is near your current viewpoint according to the map')
                return (self.controller.pose.x, self.controller.pose.y)
            direction = target_json['navigation_recommendation'].lower()
            if 'northwest' in direction:
                next_pos = (self.controller.pose.x - self.view_width/5, self.controller.pose.y + self.view_width/5)
            elif 'northeast' in direction:
                next_pos = (self.controller.pose.x + self.view_width/5, self.controller.pose.y + self.view_width/5)
            elif 'southwest' in direction:
                next_pos = (self.controller.pose.x - self.view_width/5, self.controller.pose.y - self.view_width/5)
            elif 'southeast' in direction:
                next_pos = (self.controller.pose.x + self.view_width/5, self.controller.pose.y - self.view_width/5)
            elif 'south' in direction:
                next_pos = (self.controller.pose.x, self.controller.pose.y - self.view_width/2)
            elif 'north' in direction:
                next_pos = (self.controller.pose.x, self.controller.pose.y + self.view_width/2)
            elif 'west' in direction:
                next_pos = (self.controller.pose.x - self.view_width/2, self.controller.pose.y)
            elif 'east' in direction:
                next_pos = (self.controller.pose.x + self.view_width/2, self.controller.pose.y)
            else:
                next_pos = (self.controller.pose.x, self.controller.pose.y)
        elif target_json['decision'] == 'Locate':
            next_pos = target_json['selected_pos']
        else:
            raise ValueError('Invalid decision')
        return next_pos

    def run(self):
        Success = False
        pos_log = []
        Prompt_type = 'text with map' # 还有 text_only
        map_type = 'topdown_map' # 还有 map with grid
        
        while self.controller.timestep < self.args.eval_max_timestep:
            rgb, _ = self.controller.perceive(self.controller.pose, self.episode.map_name)
            rgb_img = Image.fromarray(rgb)

            # dep_img = Image.fromarray(depth.squeeze(), mode='L')  # 'L'表示灰度模式
            # measure the z distance between the camera and the target on the ground
            image_64 = encode_image_from_pil(rgb_img)
            # update semantic map
            self.landmark_nav_map.update_observations(
                self.controller.pose, rgb, None, use_gsam_map_cache=False
            )
            # detect and memory the scene objects
            detect_mode = 'GSAM' #'GSAM'
            if detect_mode == 'VLM':
                detections = self.controller.understand(image_64, self.task_prior_knowledge, self.episode)
                # Graph memory
                self.controller.build_scene_nodes(detections, self.landmark_nav_map.landmark_map.landmarks)
            elif detect_mode == 'GSAM':
                # 从地图中获取检测结果
                detections = self.landmark_nav_map.target_map.obj_list + self.landmark_nav_map.surroundings_map.obj_list
                # Graph memory
                _, landmark = self.controller.build_scene_nodes(detections, self.landmark_nav_map.landmark_map.landmarks, show=False)

            if isinstance(self.model, Qwen2VLForConditionalGeneration):
                raise NotImplementedError
            elif isinstance(self.model, OpenAI):
                # Step 1: system Prompt
                task_prompt = self.prompts["task_description"].format(instruction=self.episode.target_description, surroundings = landmark)
                self.view_width = 2 * (self.controller.pose.z-self.landmark_nav_map.ground_level)

                # Step 2：# 产生关于环境的描述

                xyxy = [(self.controller.pose.x -self.view_width/2, self.controller.pose.y -self.view_width/2),(self.controller.pose.x +self.view_width/2, self.controller.pose.y +self.view_width/2)]
                observation_prompt = self.prompts["observation_prompt"].format(pos=self.controller.pose.xy, shape=rgb.shape[:2], area=xyxy)
                
                self.observation = self.call_response(task_prompt, observation_prompt, image_64)
                print(f"observation: {self.observation}")
                # step 3: Plan to explore or locate
                target_json = extract_json_from_msg(self.observation)
                if target_json is None:
                    continue
                # TODO: Add an AOI generation module
                if Prompt_type == 'text only':
                    semap_img = None
                elif Prompt_type == 'text with map':
                    if map_type == 'vanilla_map':
                        semap_img = self.landmark_nav_map.plot(
                            start_point=self.episode.start_pose.xy,
                            current_pose=self.controller.pose
                        )
                    elif map_type == 'topdown_map':
                        semap_img = self.landmark_nav_map.plot(
                            start_point=self.episode.start_pose.xy,
                            current_pose=self.controller.pose
                        )
                    elif map_type == 'map with grid':
                        self.landmark_nav_map.eva_plot(
                            goal_description=self.episode.description_target,
                            start_point=self.episode.start_pose.xy,
                            true_goal=self.episode.target_position.xy,
                            show=False
                        )
                next_pos = self.get_next_position(target_json, semap_img, task_prompt)
                # Step 4: 生成动作
                self.controller.pose = Pose4D(next_pos[0],next_pos[1],self.controller.pose.z, self.controller.pose.yaw)#move(self.controller.pose, Point2D(next_pos[0], next_pos[1]), 10, _)
            
            # 记录当前步骤的信息
            self.results["steps"].append({
                "time_step": t,
                "pose": (self.controller.pose.x, self.controller.pose.y, self.controller.pose.z, self.controller.pose.yaw),
                "distance_to_target": self.controller.pose.xy.dist_to(self.target.xy),
                "observation_suggestion":self.observation,
                "action_suggestion": self.action
            })
            # 记录当前位置
            pos_log.append(self.controller.pose)

            # self.controller.pose = self.controller.act(self.controller.pose, action_suggestion)

            # # Step 4: 记录历史
            # self.history = self.prompts["history_prompt"].format(history=self.history, observation=self.observation, previous_action=self.previous_action)
            # print(f"History: {self.history}")
            # # Step 5: 更新场景图
            # scene_graph = self.controller.build_scene_nodes(self.controller.args, self.controller.pose)
            # self.process_scene_graph(scene_graph)

            if self.controller.reached_target(self.controller.pose, self.target):
                print("Target reached.")
                Success = True

        self.save_results(Success)
        return Success, pos_log


    def save_results(self, success):
        # 生成唯一的文件名，包含set_height信息
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        height_info = f"height_{self.set_height}" if self.set_height is not None else "default_height"
        filename = f"SceneAgent_{self.episode.id}_{height_info}_{timestamp}.json"
        filepath = os.path.join("results/gpt-4o_map", filename)

        # 确保结果目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 添加成功状态到结果
        self.results["success"] = success

        # 保存结果到文件
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)

        print(f"Results saved to {filepath}")

    # 以下是生成Prompt的函数，用于分析观测、生成动作提示和记录历史。
    def generate_prompt(self, image, map = None, mode="observation"):
        """
        基于 RGB 图像和任务描述生成 LLM 所需的 Prompt 输入。
        mode 参数决定生成哪种类型的 Prompt 对话。
        """
        task_prompt = self.prompts["task_description"].format(instruction=self.episode.target_description)
        if mode == "observation":
            # 如果是感知环境，则利用self.prompts["observation_prompt"]生成观测描述
            observation_prompt = self.prompts["observation_prompt"]
            rationale_prompt = self.prompts["rationale"]
            conversation = [
                {
                    "role": "system", 
                    "content": [
                        {"type": "text", "text": task_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": observation_prompt},
                    ],
                    },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": rationale_prompt}
                    ],
                    },
            ]
            # 转换为处理器格式
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], images=[image, map], padding=True, return_tensors="pt")
            return inputs.to("cuda")
        elif mode == "object_caption_only":
            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": OBSERVATION_CAPTION},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        elif mode == "object_grounding_only":
            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": OBJECT_GROUNDING},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")

        elif mode == "history":
            # 如果是多轮对话，需要记录历史对话
            # change into ChatGPT
            #导入openai key配置
            os.environ["OPENAI_API_KEY"] = "sk-8xBWP046CnOzBAEaC262872c0f4d40EeAc366eB651B7C020" # 3.5--1美元
            # 设置 OPENAI_BASE_URL 环境变量
            os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
            from ggb.entity_extra import gpt_api_call
            history_prompt = self.prompts["history_prompt"].format(history=self.history, observation=self.observation, previous_action = self.previous_action)
            self.history = gpt_api_call(prompt=history_prompt)
            
            # conversation = [
            #     {
            #         "role": "user", 
            #         "content": [
            #             {
            #                 "type": "image",
            #             },
            #             {"type": "text", "text": history_prompt},
            #         ],
            #     }
            # ]
            # text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            # inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        elif mode == "action":
            # 将任务描述和动作提示合并
            prompt_template = self.prompts["action_prompt"].format(history=self.history, observation=self.observation, key_feature='')
            conversation = [
                {
                    "role": "system", 
                    "content": [
                        {"type": "text", "text": task_prompt},
                    ],
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": prompt_template},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        else:
            raise ValueError("Invalid mode. Choose from 'observation', 'history', or 'action'.")

        
        return inputs.to("cuda")

    def get_response(self, inputs):
        """
        使用 VLM 模型推理下一步 UAV 动作。
        """
        # 生成 VLM 输出
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        # 解码生成的动作
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(f"Generated Suggestion: {output_text}")
        return output_text[0]  # 假设返回单步动作


class GeonavAgent(Agent):
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode, vlmodel, set_height=None):
        super().__init__(args, initial_pose, episode)
        self.episode = episode
        self.target = self.episode.target_position
        self.set_height = set_height  # 保存set_height参数
        if set_height is not None:
            initial_pose.with_z(set_height)
        
        self.controller = LLMController(args, initial_pose)
        self.model = vlmodel
        #  local model
        if isinstance(self.model, Qwen2VLForConditionalGeneration):
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                "/data1/FoundationModels/Qwen", min_pixels=min_pixels, max_pixels=max_pixels
            )
        
        self.prompts = self.init_prompts()
        self.history = {'decision':[], 'surrounding': [], 'landmark': []}
        self.action = ''
        self.observation = ''
        self.results = {
            "target": (self.target.x, self.target.y),
            "steps": []
        }
        # Parse the instruction
        self.plan = None
        self.task_prior_knowledge = None
        self.current_task_index = 0  # 当前执行的任务索引
        self.subtask_status = "pending"  # 子任务状态（pending/running/completed）
        self.search_threshold = 0.05  # 信息增益阈值，可配置参数

        self.save_path = args.output_dir
        
        # Whether parse the instruction or not, Retry parse instruction until process
        while self.task_prior_knowledge is None:
            self.task_prior_knowledge = self.controller.parse_instruction(self.episode.target_description, self.episode.description_landmarks)
        self.landmark_nav_map = LandmarkNavMap(
                episode.map_name, args.map_shape, args.map_pixels_per_meter, 
                episode.description_landmarks, episode.description_target, episode.description_surroundings, args.gsam_params, id=episode.id, save_path=self.save_path)
        # if self.task_prior_knowledge["Target"]["class"] and self.task_prior_knowledge["Surrounding"]:
        #     self.landmark_nav_map = LandmarkNavMap(
        #         episode.map_name, args.map_shape, args.map_pixels_per_meter, 
        #         episode.description_landmarks, self.task_prior_knowledge["Target"]["class"], self.task_prior_knowledge["Surrounding"], args.gsam_params, id=episode.id,
        #     save_path=self.save_path)
        # else:
        #     self.landmark_nav_map = LandmarkNavMap(
        #         episode.map_name, args.map_shape, args.map_pixels_per_meter, 
        #         episode.description_landmarks, episode.description_target, episode.description_surroundings, args.gsam_params, id=episode.id, save_path=self.save_path)
        self.previous_sem_map = np.zeros((*self.landmark_nav_map.shape, 3), dtype=np.float32) # RGB matrix
        self.strategy_distances = {}
    def init_prompts(self):
        """
        初始化所有任务相关的 Prompt 模板。
        """
        return {
            "task_description": TASK_DESCRIPTION_PROMPT,
            "short_description": TASK_DESCRIPTION_SHORT,
            "action_prompt": ACTION_PROMPT,
            "planner_prompt": PLANNER_PROMPTV2,
            "history_prompt": HISTORY_PROMPT,
            "observation_prompt": OBSERVATION_SUMMARY,
            "multi_observation": MULTI_OBSERVATION_SUMMARY,
            "object_caption": OBSERVATION_CAPTION,
            "object_grounding":OBJECT_GROUNDING,
            "map_prompt": MAP_SUMMARY,
            "task_description_map":TASK_DESCRIPTION_FORMAP,
            "goal_description_nav": GOAL_DESCRIPTION_NAV,# used
            "landmark_navigate":LANDMARK_NAVIGATION_PROMPT,# used
            "goal_description_sea": GOAL_DESCRIPTION_SEA,# used
            "object_search":OBJECT_SEARCH_PROMPT,# used
            "goal_description_loc": GOAL_DESCRIPTION_LOC,# used
            "object_locate":TARGET_LOCATE_PROMPT,# used
        }
    def call_response(self, sysprompt, userprompt, image_list):
        if len(image_list)<=3 and sysprompt:
            rep = self.model.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": sysprompt},
                        {"role": "user", "content": [
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url":f"data:image/png;base64,{image_list[-1]}"
                                }
                            },
                            {
                                "type": "text", 
                                "text": userprompt
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
        elif 3<len(image_list)<=5:
            rep = self.model.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": [
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url":f"data:image/png;base64,{image_list[0]}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url":f"data:image/png;base64,{image_list[1]}"
                                }
                            },
                            {
                                "type": "text", 
                                "text": userprompt
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
        elif len(image_list)>=5:
            rep = self.model.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": [
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url":f"data:image/png;base64,{image_list[-5]}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url":f"data:image/png;base64,{image_list[-3]}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url":f"data:image/png;base64,{image_list[-1]}"
                                }
                            },
                            {
                                "type": "text", 
                                "text": userprompt
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
        elif userprompt:
            rep = self.model.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": userprompt}
                    ],
                    max_tokens=500
                )
        return rep.choices[0].message.content.strip()
    def get_next_position(self, mode: str, target_json: dict):
        if mode=='Navigate' and target_json['movement'] is not None:
            self.history['decision'].append('Navigate')
            direction = target_json['movement'].lower()
            if 'northwest' in direction:
                next_pos = (self.controller.pose.x - self.view_width/10, self.controller.pose.y + self.view_width/10)
            elif 'northeast' in direction:
                next_pos = (self.controller.pose.x + self.view_width/10, self.controller.pose.y + self.view_width/10)
            elif 'southwest' in direction:
                next_pos = (self.controller.pose.x - self.view_width/10, self.controller.pose.y - self.view_width/10)
            elif 'southeast' in direction:
                next_pos = (self.controller.pose.x + self.view_width/10, self.controller.pose.y - self.view_width/10)
            elif 'south' in direction:
                next_pos = (self.controller.pose.x, self.controller.pose.y - self.view_width/4)
            elif 'north' in direction:
                next_pos = (self.controller.pose.x, self.controller.pose.y + self.view_width/4)
            elif 'west' in direction:
                next_pos = (self.controller.pose.x - self.view_width/4, self.controller.pose.y)
            elif 'east' in direction:
                next_pos = (self.controller.pose.x + self.view_width/4, self.controller.pose.y)
            else:
                next_pos = (self.controller.pose.x, self.controller.pose.y)
        elif mode=='Search':
            self.history['decision'].append('Search')
            try:
                direction = target_json['movement'].lower()
            except KeyError:
                direction = target_json.get('answer', {}).get('movement', '').lower()
            if 'northwest' in direction:
                next_pos = (self.controller.pose.x - self.view_width/20, self.controller.pose.y + self.view_width/20)
            elif 'northeast' in direction:
                next_pos = (self.controller.pose.x + self.view_width/20, self.controller.pose.y + self.view_width/20)
            elif 'southwest' in direction:
                next_pos = (self.controller.pose.x - self.view_width/20, self.controller.pose.y - self.view_width/20)
            elif 'southeast' in direction:
                next_pos = (self.controller.pose.x + self.view_width/20, self.controller.pose.y - self.view_width/20)
            elif 'south' in direction:
                next_pos = (self.controller.pose.x, self.controller.pose.y - self.view_width/8)
            elif 'north' in direction:
                next_pos = (self.controller.pose.x, self.controller.pose.y + self.view_width/8)
            elif 'west' in direction:
                next_pos = (self.controller.pose.x - self.view_width/8, self.controller.pose.y)
            elif 'east' in direction:
                next_pos = (self.controller.pose.x + self.view_width/8, self.controller.pose.y)
            else:
                next_pos = (self.controller.pose.x, self.controller.pose.y)
        elif mode=='Locate':
            self.history['decision'].append('Locate')
            next_pos = target_json['selected_pos']
        else:
            next_pos = (self.controller.pose.x, self.controller.pose.y)
        return next_pos
    
    def gen_map(self, query_type):
        if query_type == 'landmark':# channel 1: lanmark map
                semap_img = self.landmark_nav_map.plot(
                    map_type='landmark'
                )
        elif query_type == 'semantic': # channel 2: semantic map
            semap_img = self.landmark_nav_map.plot(
                map_type='semantic',
                query_engine=self.controller.query_engine,
                current_pos=self.controller.pose.xy
            )
        elif query_type == 'w/o_annotation': # channel 3: no annotation
            semap_img = self.landmark_nav_map.plot(
                map_type='w/o annotation'
            )
        elif query_type == 'topdown': # channel 4:full map
            semap_img = self.landmark_nav_map.plot(map_type = 'topdown_map')
        else:
            raise ValueError('Invalid query type')
        return semap_img
    
    def initialize_subtask(self, current_task):
        if current_task['strategy'] == 'Navigate':
            self.sys_prompt = self.prompts["goal_description_nav"]
        elif current_task['strategy'] == 'Search':
            self.sys_prompt = self.prompts["goal_description_sea"]
        else: # 'Locate'
            self.sys_prompt = self.prompts["goal_description_loc"]
    
    def execute_subtask(self, task, geoinstruct, img_64):
        if task['strategy'] == 'Navigate':
            # 输入地标地图、任务描述和地理指示，输出推理和行动，考虑采取一次性生成多个动作
            img = self.gen_map(query_type='landmark')
            text_prompt = self.prompts['landmark_navigate'].format(geoinstruct=geoinstruct, goal=task['goal'], state=task['desired_state'])
        elif task['strategy'] == 'Search':
            img = self.gen_map(query_type='semantic')
            text_prompt = self.prompts['object_search'].format(geoinstruct=geoinstruct, goal=task['goal'], state=task['desired_state'])
        elif task['strategy'] == 'Locate':
            img = img_64
            text_prompt = self.prompts['object_locate'].format(pos=self.controller.pose.xy, area=self.xyxy, goal=self.episode.target_description)
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            enhanced_prompt = f"{text_prompt}\n\nPlease strictly follow the requirements below when responding:\n1. Use standard JSON format\n2. Include all required fields\n3. Wrap the response with ```json"
            try:
                self.action = self.call_response(sysprompt=self.sys_prompt, userprompt=enhanced_prompt, image_list=[img])
                target_json = extract_json_from_msg(self.action)
                if target_json is not None:
                    break
            except Exception as e:
                print(f"Error occurred during API call: {e}")
                retry_count += 1
        if retry_count == max_retries:
            return self.controller.pose.xy
        next_pos = self.get_next_position(task['strategy'], target_json)
        self.landmark_nav_map.step += 1
        self.observation = text_prompt
        return next_pos
    
    def check_subgoals(self, task):
        if task['strategy'] == 'Navigate':
            if self.controller.timestep > 6: #防止陷入NotFound
                self.strategy_distances['Navigate'] = self.controller.pose.xy.dist_to(self.episode.target_position.xy)
                return True
            # check if any landmarks are reached
            return all(lm.position.xy.dist_to(self.controller.pose.xy) < 50.0 for lm in self.landmark_nav_map.landmark_map.landmarks)
        elif task['strategy'] == 'Search':
            if self.controller.timestep > 15: #防止陷入search around
                self.strategy_distances['Search'] = self.controller.pose.xy.dist_to(self.episode.target_position.xy)
                return True
            # evaluate whether the information has changed
            sem_map = self.landmark_nav_map.get_semantic_map()
            
            if not hasattr(self, 'exploration_analyzer'):
                self.exploration_analyzer = ExplorationAnalyzer(threshold=1e-6)
            # 与上一帧比较
            if self.exploration_analyzer.should_continue(sem_map, self.previous_sem_map):
                self.previous_sem_map = sem_map.copy()
                return False
        elif task['strategy'] == 'Locate':
            # 利用知识推断目标名称，飞到对应位置
            # 假如没找到目标，还要继续
            return True
        return True

    def run(self):
        Success = False
        strategy = ''
        pos_log = []
        image_list = []
        self.strategy_distances['Start'] = self.controller.pose.xy.dist_to(self.episode.target_position.xy)
        # 添加终止条件变量
        while self.controller.timestep < self.args.eval_max_timestep:
            rgb, _ = self.controller.perceive(self.controller.pose, self.episode.map_name)
            image_64 = encode_image_from_pil(Image.fromarray(rgb))
            image_list.append(image_64)
            # TODO: measure the z distance between the camera and the target on the ground
            # dep_img = Image.fromarray(depth.squeeze(), mode='L')  # 'L'表示灰度模式
            
            # update map with observations
            self.landmark_nav_map.update_observations(
                self.controller.pose, rgb, None, use_gsam_map_cache=False, strategy=strategy
            )
            # detect and memory the scene objects
            detect_mode = 'VLM' #'GSAM'
            landmark = self.controller.build_geo_nodes(self.landmark_nav_map.landmark_map.landmarks)
            if detect_mode == 'VLM' and strategy == 'Search':
                # TODO test the effect of VLM
                subgraph = self.controller.understand(image_64, self.episode)
                # Translate into world_xy
                self.controller.build_scene_graph(subgraph, self.landmark_nav_map.target_map)
            elif detect_mode == 'GSAM':
                # Graph memory
                surrounding, recent_objects = self.controller.build_scene_nodes(self.landmark_nav_map.target_map.obj_list, self.landmark_nav_map.surroundings_map.obj_list, show=False)

            # Step 1: system Prompt
            geoinstruct = self.prompts["history_prompt"].format(landmark=landmark, recent_objects='', surroundings='')
            self.view_width = 2 * (self.controller.pose.z-self.landmark_nav_map.ground_level)

            # Step 2: static planning
            self.xyxy = [(self.controller.pose.x -self.view_width/2, self.controller.pose.y -self.view_width/2),(self.controller.pose.x +self.view_width/2, self.controller.pose.y +self.view_width/2)]

            while self.plan is None:
                planner_prompt = self.prompts["planner_prompt"].format(instruction=self.episode.target_description, geoinstruct=geoinstruct)
                print('Retry generating plan')
                response = self.call_response(sysprompt=None, userprompt=planner_prompt, image_list=[])
                self.plan = extract_json_from_msg(response)
            
            # Step 3：# execute the subtask
            current_task = self.plan['sub_goals'][self.current_task_index]
            print(f"Current Task [{self.current_task_index+1}/{len(self.plan['sub_goals'])}]: {current_task['goal']}")
            if self.subtask_status == "pending":
                # initialize the subtask prompt
                self.initialize_subtask(current_task)
                self.subtask_status = "running"
            # Step 4: Choose strategy to exectue the goal
            # function: using different map type as different sacle observation
            next_pos = self.execute_subtask(current_task, geoinstruct, image_64)
            self.controller.pose = Pose4D(next_pos[0],next_pos[1],self.controller.pose.z, self.controller.pose.yaw) # move(self.controller.pose, Point2D(next_pos[0], next_pos[1]), 10, _)
            strategy = current_task['strategy'] # temp
            if self.controller.reached_target(self.controller.pose, self.target):
                print("Target reached.")
                Success = True
            # Step 5: check the subgoals
            if self.check_subgoals(current_task):
                self.subtask_status = "completed"
                self.current_task_index += 1
                # 检查是否所有任务完成
                if self.current_task_index >= len(self.plan['sub_goals']):
                    print("All sub-tasks completed!")
                    self.strategy_distances['Locate'] = self.controller.pose.xy.dist_to(self.episode.target_position.xy)
                    break  # 提前退出循环
            self.results["steps"].append({
                "time_step": self.controller.timestep,
                "pose": (self.controller.pose.x, self.controller.pose.y, self.controller.pose.z, self.controller.pose.yaw),
                "distance_to_target": self.controller.pose.xy.dist_to(self.target.xy),
                "plan": current_task,
                "observation_suggestion":self.observation,
                "action_suggestion": self.action
            })
            # 记录当前位置
            pos_log.append(self.controller.pose)
            
        self.save_results(Success)
        return Success, pos_log


    def save_results(self, success):
        # 生成唯一的文件名，包含set_height信息
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GeonavAgent_{self.episode.id}_{timestamp}.json"
        filepath = os.path.join(self.save_path, filename)# self.save_path

        # 确保结果目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 添加成功状态到结果
        self.results["success"] = success

        # 保存结果到文件
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)

        print(f"Results saved to {filepath}")