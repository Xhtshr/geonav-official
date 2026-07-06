import re
import os, ast
import cv2
import json
import numpy as np
from skimage import color

from gsamllavanav.space import Pose4D, Point2D
from gsamllavanav.dataset.episode import Episode
from scenegraphnav.parser import parse_args
from scenegraphnav.parser import ExperimentArgs
from scenegraphnav.llm_controller import LLMController # only used for our scenegraphnav methods
from scenegraphnav.evaluate import move

from PIL import Image
from openai import OpenAI
from utils.QwenAPI import encode_image_from_pil
from scenegraphnav.prompt.config import VLM_NAME, LLM_NAME

from scenegraphnav.prompt.navgpt import *
from gsamllavanav.actions import DiscreteAction
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from gsamllavanav.observation import cropclient

from datetime import datetime
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
args = parse_args()

if args.ablation == 'wo_cot':
    from scenegraphnav.prompt.geonav import *
else:
    from scenegraphnav.prompt.geonav_cot import *

#计算图像信息熵，用于判断环境是否发生变化
def rgb_entropy(rgb_img):
    hsv_img = color.rgb2hsv(rgb_img)
    entropy = 0
    for i in range(3):
        channer = (hsv_img[..., i]*255).astype(np.uint8)
        hist = cv2.calcHist([channer], [0], None, [256], [0, 256])
        hist = hist / hist.sum() + 1e-10
        entropy += -np.sum(hist * np.log2(hist))
    return entropy

#计算结构相似性，用于判断环境是否发生变化
def structural_similarity(current, previous):
    """计算结构相似性指标"""
    from skimage.metrics import structural_similarity as ssim
    return ssim(current, previous, multichannel=True, win_size=3)

#主动探索控制器：通过网格化分析地图变化，判断当前“搜索”策略是否还有价值，如果信息增益低于阈值（threshold=0.08），则认为已充分探索，切换到下一子任务
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

import re
import json

def fix_json_error(json_str):
    """
    修复常见的JSON格式错误
    
    常见错误类型：
    1. 多余的 } 或 ] (如 {"key": "value"}})
    2. 缺少逗号
    3. 多余的逗号
    4. 括号不匹配
    
    参数：
    json_str (str): 可能有错误的JSON字符串
    
    返回：
    str: 修复后的JSON字符串，如果无法修复则返回原字符串
    """
    original = json_str
    
    # 首先尝试直接解析
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    
def fix_json_error(json_str):
    """
    修复常见的JSON格式错误
    
    常见错误类型：
    1. 多余的 } 或 ] (如 {"key": "value"}})
    2. 缺少逗号
    3. 多余的逗号
    
    参数：
    json_str (str): 可能有错误的JSON字符串
    
    返回：
    str: 修复后的JSON字符串，如果无法修复则返回原字符串
    """
    original = json_str
    
    # 首先尝试直接解析
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    
    # 1. 移除行尾多余的逗号
    json_str = re.sub(r',(\s*[\]\}])', r'\1', json_str)
    
    # 2. 逐个尝试移除多余的闭合括号，找到可以解析的那个
    test_json = json_str
    
    # 最多尝试移除10个字符
    for _ in range(20):
        try:
            json.loads(test_json)
            return test_json
        except json.JSONDecodeError as e:
            # 找到错误位置，尝试移除附近的 } 或 ]
            pos = e.pos
            if pos is None or pos >= len(test_json):
                break
            
            # 尝试移除错误位置的字符
            test_json = test_json[:pos] + test_json[pos+1:]
    
    # 3. 如果上述方法失败，尝试括号平衡法
    json_str = re.sub(r',(\s*[\]\}])', r'\1', original)
    
    # 从后向前扫描，移除多余的闭合括号
    result = []
    i = len(json_str) - 1
    in_string = False
    escaped = False
    extra_closing = 0
    
    while i >= 0:
        char = json_str[i]
        
        if escaped:
            result.append(char)
            escaped = False
            i -= 1
            continue
        
        if char == '\\':
            result.append(char)
            escaped = True
            i -= 1
            continue
        
        if char == '"':
            in_string = not in_string
            result.append(char)
            i -= 1
            continue
        
        if not in_string:
            if char in '}]':
                extra_closing += 1
            elif char in '{[':
                if extra_closing > 0:
                    extra_closing -= 1
                else:
                    result.append(char)
            else:
                result.append(char)
        else:
            result.append(char)
        
        i -= 1
    
    json_str = ''.join(reversed(result))
    
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    
    # 4. 最终策略：使用括号匹配提取JSON
    json_str = json_str.strip()
    
    if json_str.startswith('['):
        first_bracket = '['
        closing_bracket = ']'
    elif json_str.startswith('{'):
        first_bracket = '{'
        closing_bracket = '}'
    else:
        return original
    
    depth = 0
    in_string = False
    escaped = False
    end_idx = -1
    
    for idx, char in enumerate(json_str):
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if char == '"' and not escaped:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == first_bracket:
                depth += 1
            elif char == closing_bracket:
                depth -= 1
                if depth == 0:
                    end_idx = idx + 1
                    break
    
    if end_idx > 0:
        extracted = json_str[:end_idx]
        try:
            json.loads(extracted)
            return extracted
        except:
            pass
    
    return original  # 无法修复，返回原始字符串


def extract_json_from_msg(msg):
    """
    从包含JSON代码块的文本中提取并解析JSON数据

    参数：
    msg (str): 包含JSON代码块的原始文本

    返回：
    dict: 解析后的JSON字典，未找到返回None，解析失败返回None
    """
    # 1. 优先尝试匹配 ```json 包裹的JSON内容
    pattern = r'```json\s*(.*?)\s*```'
    match = re.search(pattern, msg, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed_json_str = fix_json_error(json_str)
            try:
                return json.loads(fixed_json_str)
            except json.JSONDecodeError:
                return None

    # 2. 如果没有代码块，尝试直接解析整条消息（支持裸 JSON 数组或对象）
    stripped = msg.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. 如果仍失败，尝试修复后解析
    fixed_json_str = fix_json_error(stripped)
    try:
        return json.loads(fixed_json_str)
    except json.JSONDecodeError:
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
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode, vlmodel, llmodel, set_height=None):
        super().__init__(args, initial_pose, episode)
        self.episode = episode
        self.target = self.episode.target_position
        self.set_height = set_height  # 保存set_height参数
        if set_height is not None:
            initial_pose.with_z(set_height)
        self.controller = LLMController(args, initial_pose, vlmodel, llmodel)
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
            "observation_prompt": OBSERVATION_SUMMARY,
            "object_caption": OBSERVATION_CAPTION,
            "object_grounding":OBJECT_GROUNDING,
        }
    
    def call_response(self, sysprompt, userprompt, image_64):
        if image_64 and sysprompt:
            rep = self.model.chat.completions.create(
                    model=VLM_NAME,
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
                    max_tokens=500
                )
        elif userprompt:
            rep = self.model.chat.completions.create(
                    model=VLM_NAME,
                    messages=[
                        {"role": "user", "content": userprompt}
                    ],
                    max_tokens=500
                )
        return rep.choices[0].message.content.strip()

    def run(self, naive=True):
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
                landmark = ''
                # _, landmark = self.controller.build_scene_nodes([], self.landmark_nav_map.landmark_map.landmarks, show=False)
            
            geo_info = ' at ' + str(self.controller.pose.xy)
            for lm in self.landmark_nav_map.landmark_map.landmarks:
                geo_info += '. '+ str(lm.name) + ' is located at '+str(lm.position) + '. Its contour is' + str(lm.contour)
            
            if isinstance(self.model, Qwen2VLForConditionalGeneration):
                raise NotImplementedError
            elif isinstance(self.model, OpenAI):
                # Step 1: system Prompt
                if naive:
                    landmark = geo_info
                task_prompt = self.prompts["task_description"].format(instruction=self.episode.target_description, geoinstruct = landmark)
                self.view_width = 2 * (self.controller.pose.z-self.landmark_nav_map.ground_level)
                #print(f"Task Prompt: {task_prompt}")
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
                self.controller.pose = Pose4D(next_pos[0],next_pos[1],self.controller.pose.z, self.controller.pose.yaw) #move(self.controller.pose, Point2D(next_pos[0], next_pos[1]), 10, _)

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
            filepath = os.path.join("results/gpt-4o/naive", filename)
        else:
            filepath = os.path.join("results/gpt-4o", filename)

        # 确保结果目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 添加成功状态到结果
        self.results["success"] = success

        # 保存结果到文件
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)

        print(f"Results saved to {filepath}")
        
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
            "observation_prompt": OBSERVATION_SUMMARY,
            "object_caption": OBSERVATION_CAPTION,
            "object_grounding":OBJECT_GROUNDING,
            "map_prompt": MAP_SUMMARY,
        }
    def call_response(self, sysprompt, userprompt, image_64):
        if image_64 and sysprompt:
            rep = self.model.chat.completions.create(
                    model=VLM_NAME, #"gpt-4o",
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
                    max_tokens=500
                )
        elif userprompt:
            rep = self.model.chat.completions.create(
                    model=VLM_NAME, #"gpt-4o",
                    messages=[
                        {"role": "user", "content": userprompt}
                    ],
                    max_tokens=500
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
                # load detection results from map
                detections = self.landmark_nav_map.target_map.obj_list + self.landmark_nav_map.surroundings_map.obj_list
                # Graph memory
                _, landmark = self.controller.build_scene_nodes(detections, self.landmark_nav_map.landmark_map.landmarks, show=False)
                if args.ablation == 'wo_landmark':
                    landmark = ''

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
                "time_step": self.controller.timestep,
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

        # elif mode == "history":
        #     # 如果是多轮对话，需要记录历史对话
        #     # change into ChatGPT
        #     #导入openai key配置
        #     os.environ["OPENAI_API_KEY"] = "sk-8xBWP046CnOzBAEaC262872c0f4d40EeAc366eB651B7C020" # 3.5--1美元
        #     # 设置 OPENAI_BASE_URL 环境变量
        #     os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
        #     from utils.entity_extra import gpt_api_call
        #     history_prompt = self.prompts["history_prompt"].format(history=self.history, observation=self.observation, previous_action = self.previous_action)
        #     self.history = gpt_api_call(prompt=history_prompt)
            
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
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode, vlmodel, llmodel, set_height=None):
        super().__init__(args, initial_pose, episode)
        self.episode = episode
        self.target = self.episode.target_position
        if set_height is not None:
            initial_pose.with_z(set_height)
        
        self.controller = LLMController(args, initial_pose, vlmodel, llmodel)
        self.model = vlmodel
        
        self.prompts = self.init_prompts()
        self.history = {'decision':[], 'action': [], 'observation': []}
        self.action = ''
        self.last_movement = ''
        self.observation = ''
        self.results = {
            "target": (self.target.x, self.target.y),
            "steps": [],
            "scene_graph_summary": "",
            "query_logs": []
        }
        # Parse the instruction
        self.plan = None
        self.task_prior_knowledge = None
        self.current_task_index = 0  # 当前执行的任务索引
        self.subtask_status = "pending"  # 子任务状态（pending/running/completed）
        self.search_threshold = 0.05  # 信息增益阈值，可配置参数 (可以测试的参数)
        
        self.save_path = os.path.join(args.output_dir, args.ablation)
        if not os.path.exists(self.save_path):
            try:
                os.makedirs(self.save_path)
            except OSError as e:
                # Handle the error appropriately
                raise Exception(f"Failed to create directory {self.save_path}: {str(e)}")
        
        # Whether parse the instruction or not, Retry parse instruction until process
        # while self.task_prior_knowledge is None:
        #     self.task_prior_knowledge = self.controller.parse_instruction(self.episode.target_description, self.episode.description_landmarks)
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
        #self.previous_sem_map = np.zeros((*self.landmark_nav_map.shape, 3), dtype=np.float32) # RGB matrix
        self.previous_sem_map = np.ones((*self.landmark_nav_map.shape, 3), dtype=np.float32) #全1初始化，避免首步空地图直接退出
        self.strategy_distances = {}

        # 添加用于控制 VLM 识别频率的变量
        self.last_vlm_detection_time = 0  # 上次 VLM 检测的时间步
        self.vlm_detection_interval = 20   # VLM 检测的间隔步数（可配置）
        self.vlm_detection_distance = 20  # 移动距离阈值（米），超过此距离触发新的检测
        self.last_vlm_detection_position = initial_pose.xy  # 上次 VLM 检测的位置
        
        # 添加策略步数追踪
        self.strategy_timesteps = {
            'Navigate': 0,
            'Search': 0,
            'Locate': 0
        }
    def init_prompts(self):
        """
        初始化所有任务相关的 Prompt 模板。
        """
        return {# used only
            "planner_prompt": PLANNER_PROMPTV2,
            "landmark_prompt": LANDMARK_DESCRIPTION,
            "goal_description_nav": GOAL_DESCRIPTION_NAV,
            "landmark_navigate":LANDMARK_NAVIGATION_PROMPT,
            "goal_description_sea": GOAL_DESCRIPTION_SEA,
            "object_search":OBJECT_SEARCH_PROMPT,
            "goal_description_loc": GOAL_DESCRIPTION_LOC,
            "object_locate":TARGET_LOCATE_PROMPT,
        }
    def call_response(self, sysprompt, userprompt, image_list):
        if 0<len(image_list) and sysprompt:
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
                    ]
        elif sysprompt:
            messages=[
                        {"role": "system", "content": sysprompt},
                        {"role": "user", "content": userprompt}
                    ]
        elif userprompt:
            messages=[
                        {"role": "user", "content": userprompt}
                    ]
        rep = self.model.chat.completions.create(
                model=VLM_NAME, #"gpt-4o",
                messages=messages,
                max_tokens=800,
                extra_body={
                    "thinking":{"type":"disabled",}
                }
            )
        return rep.choices[0].message.content.strip()
    
    def get_next_position(self, mode: str, target_json: dict):
        try:
            next_pos = (self.controller.pose.x, self.controller.pose.y)
            base_distance = self.view_width/6
            # easy上面的设置为1.0，medium 设置为1.5,hard设置为3.5
            multiplier = 1.5 if mode == 'Navigate' else 1  # Navigate模式时距离增大3.5倍，Search模式时不变
            OPPOSITE_DIRS = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east',
            'northeast': 'southwest',
            'southwest': 'northeast',
            'northwest': 'southeast',
            'southeast': 'northwest',}
            direction_map = {
                'northwest': (-base_distance * multiplier, base_distance * multiplier),
                'northeast': (base_distance * multiplier, base_distance * multiplier),
                'southwest': (-base_distance * multiplier, -base_distance * multiplier),
                'southeast': (base_distance * multiplier, -base_distance * multiplier),
                'north': (0, base_distance * multiplier),
                'south': (0, -base_distance * multiplier),
                'west': (-base_distance * multiplier, 0),
                'east': (base_distance * multiplier, 0),
                'stop': (0, 0),
            }
            action_scale = {
                'Navigate': 1,
                'Search': 3,
            }
            
            # 验证target_json类型
            if not isinstance(target_json, dict):
                logger.warning(f"Invalid target_json type: {type(target_json)}, using stop action")
                self.history['decision'].append('Invalid_Stop')
                self.history['action'].append('stop')
                return next_pos
                
            if mode in ['Navigate', 'Search']:
                self.history['decision'].append(mode)
                # 安全获取movement
                direction = str(target_json.get('movement', 'stop')).lower()
                if OPPOSITE_DIRS.get(direction) == self.last_movement:
                    self.history['action'].append('stop')
                    return next_pos
                
                # 默认使用stop
                dx, dy = direction_map.get('stop', (0, 0))
                
                # 尝试匹配方向
                for key in direction_map:
                    if key in direction:
                        dx, dy = direction_map[key]
                        self.history['action'].append(key)
                        break
                else:
                    self.history['action'].append('stop')
                    
                next_pos = (
                    self.controller.pose.x + dx/action_scale.get(mode, 1),
                    self.controller.pose.y + dy/action_scale.get(mode, 1)
                )
                
            elif mode == 'Locate':
                self.history['decision'].append('Locate')
                # 安全获取selected_pos
                if 'selected_pos' in target_json:
                    next_pos = target_json['selected_pos']
                    self.history['action'].append(next_pos)
                else:
                    self.history['action'].append('stop')
                    
            return next_pos
                
        except Exception as e:
            self.history['decision'].append('Error_Stop')
            self.history['action'].append('stop')
            return (self.controller.pose.x, self.controller.pose.y)
    
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
        try:
            # Strategy 1: Navigate
            if task['strategy'] == 'Navigate':
                img = self.gen_map(query_type='landmark')
                text_prompt = self.prompts['landmark_navigate'].format(geoinstruct=geoinstruct, goal=task['goal'], state=task['desired_state'])
                #text_prompt = self.prompts['landmark_navigate'].format(describe=landmark_descriptions, position=position, goal=task['goal'], state=task['desired_state'])

            # Strategy 2: Search
            elif task['strategy'] == 'Search':
                if args.ablation == 'wo_scm':
                    img = img_64
                else:
                    img = self.gen_map(query_type='semantic')
                text_prompt = self.prompts['object_search'].format(geoinstruct=geoinstruct, goal=task['goal'], state=task['desired_state'])

            # Strategy 3: Locate, using Scene Graph
            # Strategy 3: Locate
            # 改进版链式查询
            # elif task['strategy'] == 'Locate'and args.ablation != 'wo_sg':
                
            #     print(f"Executing locate query: {self.episode.target_description}")
            #     try:
            #         # Step 1: Generate operation chain using V2 prompt
            #         #operation_chain = self.generate_operation_chain_v2(self.episode.target_description)
            #         operation_chain = self.generate_operation_chain_v2(self.episode.target_description)
            #         if not operation_chain:
            #             print("Failed to generate operation chain")
            #             return self.controller.pose.xy
            #         # Step 2: Execute operation chain to get candidates and extra log info
            #         candidates, extra_info = self.execute_operation_chain_v2(operation_chain, debug=True)
            #         # Step 3: Always save query log (even if candidates is empty)
            #         self.save_operation_chain_log(
            #             goal=task['goal'],
            #             strategy="locate_v2",
            #             operation_chain=operation_chain,
            #             result_nodes=candidates,
            #             step_logs=extra_info.get("step_logs", []),
            #             verify_log=extra_info.get("verify_log", {})
            #         )
            #         # Step 4: Return first candidate or fallback to geo node
            #         if not candidates:
            #             print("No candidates found via operation chain")
            #             return self.controller.pose.xy
            #         selected_node = candidates[0]
            #         next_pos = (selected_node.position.x, selected_node.position.y)
            #         self.history['action'].append(next_pos)
            #         return next_pos
            #     except Exception as e:
            #         print(f"Error in Locate strategy: {e}")
            #         import traceback
            #         traceback.print_exc()
            #         return self.controller.pose.xy
            # 原版查询    
            elif task['strategy'] == 'Locate' and args.ablation != 'wo_sg':
                # The 'Locate' strategy uses a direct scene graph query. If successful, 
                # it returns immediately, skipping the subsequent LLM call flow.
                is_complex = self.is_complex_query(task['goal'])
                
                if is_complex:
                    # First, attempt a direct scene graph query for complex queries.
                    print(f"Executing complex query: {task['goal']}")
                    try:
                        target_nodes, query_info = self.controller.complex_query_scene_graph(task['goal'], debug=True)
                    except Exception as e:
                        print(f"Error in complex_query_scene_graph: {e}")
                        target_nodes = None
                        query_info = None
                    # 保存查询日志
                    if query_info:
                        self.results["query_logs"].append({
                            "goal": task['goal'],
                            "strategy": "complex query",
                            "result_count": len(target_nodes) if target_nodes else 0,
                            "result_ids": [n.id for n in target_nodes] if target_nodes else [],
                            "query_details": query_info
                        })
                    # If the direct query fails, fall back to a multi-step operation chain.
                    if not target_nodes:
                        print("Complex query returned no results, attempting multi-step operation chain.")
                        #operation_chains = self.generate_operation_chain(self.episode.target_description)
                        operation_chains = self.generate_operation_chain(task['goal'])
                        target_nodes = self.recursive_query(operation_chains, max_depth=5, current_depth=0)
                        # 保存操作链信息到日志
                        if operation_chains:
                            self.results["query_logs"].append({
                                "goal": task['goal'],
                                "strategy": "complex fallback",
                                "operation_chain": operation_chains
                            })
                else:
                    # For simple queries, directly use the enhanced subgraph query.
                    print(f"Executing simple query: {task['goal']}")
                    #operation_chain = self.generate_operation_chain(self.episode.target_description)
                    operation_chain = self.generate_operation_chain(task['goal'])
                    target_nodes = self.recursive_query(operation_chain, max_depth=5, current_depth=0)
                    
                    # 保存操作链信息到日志
                    if operation_chain:
                        self.results["query_logs"].append({
                            "goal": task['goal'],
                             "strategy": "simple query",
                            "operation_chain": operation_chain
                        })
                
                if target_nodes:
                    # If target nodes are found, select the one with the highest confidence and return its position directly.
                    selected_node = max(target_nodes, key=lambda n: getattr(n, 'confidence', 0) if hasattr(n, 'confidence') else 0)
                    next_pos = (selected_node.position.x, selected_node.position.y)
                    # Note: This is an early return on a successful path.
                    return next_pos
                else:
                    # If the scene graph query fails, prepare info and proceed to the LLM call flow below as a fallback.
                    img = img_64
                    text_prompt = self.prompts['object_locate'].format(pos=self.controller.pose.xy, area=self.xyxy, goal=self.episode.target_description)
            
            # Strategy 4: Locate without Scene Graph, or the fallback from a failed Scene Graph query
            # Covers 'Locate' with args.ablation == 'wo_sg' or the fallback case from above
            else:
                img = img_64
                text_prompt = self.prompts['object_locate'].format(pos=self.controller.pose.xy, area=self.xyxy, goal=self.episode.target_description)
            
            # --- LLM Call Flow (for Navigate, Search, and Locate fallback) ---
            max_retries = 3
            retry_count = 0
            target_json = None

            while retry_count < max_retries:
                enhanced_prompt = f"{text_prompt}\n\nPlease strictly follow the requirements below when responding:\n1. Use standard JSON format\n2. Include all required fields\n3. Wrap the response with ```json"
                try:
                    self.action = self.call_response(sysprompt=self.sys_prompt, userprompt=enhanced_prompt, image_list=[img])
                    target_json = extract_json_from_msg(self.action)
                    if target_json is not None:
                        # Successfully retrieved and parsed JSON, breaking the retry loop.
                        break
                except Exception as e:
                    # Catch specific errors during the API call or JSON parsing.
                    print(f"An error occurred on attempt {retry_count + 1}: {e}")
                    retry_count += 1
            
            # If a valid JSON is not obtained after all retries, the task fails.
            if target_json is None:
                print("API call failed after all retries. Staying at the current position.")
                return self.controller.pose.xy

            # --- Processing after a successful LLM response ---
            next_pos = self.get_next_position(task['strategy'], target_json)
            self.landmark_nav_map.step += 1
            self.observation = text_prompt
            
            return next_pos

        except Exception as e:
            # Catch any unexpected exceptions during the function's execution.
            #print(f"Full error type: {type(e).__name__}")
            #print(f"Full error message: {e}")
            print(f"An unexpected global error occurred in execute_subtask: {e}")
            # As required, return the current position on any error.
            return self.controller.pose.xy

    def generate_operation_chain(self, goal_description):
        """生成操作链"""
        # 使用常规方式生成操作链
        prompt = QUERY_OPERATION_CHAIN_PROMPT.format(instruction=goal_description)
        response = self.call_response(sysprompt=None, userprompt=prompt, image_list=[])
        """ response = self.controller.llm_client.chat.completions.create(
                    model= LLM_NAME, #"gpt-4-turbo"
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a professional query planner."},
                        {"role": "user", "content": prompt}
                    ]
                ) """
        print(f"Generated operation chain has responsed: {response}")
        operation_chain = extract_json_from_msg(response)
        print(f"Parsed operation_chain: {operation_chain}")
        print(f"Type of operation_chain: {type(operation_chain)}")
        return operation_chain

    def generate_operation_chain_v2(self, goal_description):
        """使用 V2 prompt 生成操作链，包含 verify_spatial_relation 支持"""
        prompt = QUERY_OPERATION_CHAIN_PROMPT_V2.format(instruction=goal_description,landmark=self.landmark_nav_map.landmark_map.landmarks, target=self.episode.target_description)
        response = self.call_response(sysprompt=None, userprompt=prompt, image_list=[])
        print(f"Generated operation chain (V2): {response}")
        operation_chain = extract_json_from_msg(response)
        print(f"Parsed operation_chain: {operation_chain}")
        return operation_chain

    def execute_operation_chain_v2(self, operation_chain, debug=False):
        """执行 V2 操作链，返回候选节点列表

        如果操作链中包含 verify_spatial_relation，会先获取初始候选，
        然后对每个候选执行空间关系验证。
        如果候选为空，返回 get_geonode_by_name 的第一个结果。
        """
        return self.controller.query_engine.query_operation(
            operation_chain=operation_chain,
            target_description=self.episode.target_description,
            debug=debug
        )

    def save_operation_chain_log(self, goal, strategy, operation_chain, result_nodes, step_logs=None, verify_log=None):
        """保存操作链查询日志"""
        log_entry = {
            "goal": goal,
            "strategy": strategy,
            "operation_chain": operation_chain,
            "result_count": len(result_nodes) if result_nodes else 0,
            "result_ids": [n.id for n in result_nodes] if result_nodes else []
        }
        if step_logs:
            log_entry["step_logs"] = step_logs
        if verify_log:
            log_entry["verify_log"] = verify_log
        self.results["query_logs"].append(log_entry)

    def is_complex_query(self, query):
        """判断查询是否为复杂查询"""
        # 包含多个句子
        if '.' in query and query.count('.') > 1:
            return True
        
        # 包含多个关系词
        relation_keywords = ['in front of', 'behind', 'next to', 'beside', 'between', 
                            'across from', 'opposite', 'facing', 'same direction',
                            'opposite direction', 'parked', 'in', 'on']
        relation_count = sum(keyword in query.lower() for keyword in relation_keywords)
        if relation_count >= 2:
            return True
        
        # 包含多个连接词
        if query.lower().count(' and ') > 1 or query.lower().count(',') > 1:
            return True
        
        return False
    
    # def generate_complex_operation_chain(self, goal_description):
        """为复杂查询生成更适合的操作链"""
        # 创建提示词，要求LLM将复杂查询分解为多个步骤
        prompt = f"""
        将以下复杂指令分解为多个步骤，每个步骤使用一个单独的操作链：
        
        指令: "{goal_description}"
        
        各步骤应该是有顺序、循序渐进的。例如，对于指令"在Davey Road上停着一辆白色汽车，它前面有一辆灰色汽车面向相反方向"，可分解为：
        
        步骤1: 找到Davey Road上的白色汽车
        操作链: [
            {{"method": "get_geonode_by_name", "args": ["Davey Road"]}},
            {{"method": "get_child_nodes", "kwargs": {{"relation_type": "contains"}}}},
            {{"method": "filter_by_class", "args": ["vehicle"]}},
            {{"method": "filter_by_attribute", "kwargs": {{"color": "white"}}}}
        ]
        
        步骤2: 在步骤1结果的基础上，找出前面的灰色汽车
        操作链: [
            {{"method": "get_child_nodes", "kwargs": {{"relation_type": "north_of"}}}},
            {{"method": "filter_by_class", "args": ["vehicle"]}},
            {{"method": "filter_by_attribute", "kwargs": {{"color": "gray"}}}}
        ]
        
        请按照上面的格式，为给定指令生成多步骤的操作链。
        输出格式必须为有效的JSON格式，请确保每个操作链都被正确包裹在方括号内，且使用正确的JSON语法。
        """
        
        response = self.call_response(sysprompt=None, userprompt=prompt, image_list=[])
        
        # 改进的JSON提取方法
        try:
            # 首先尝试从响应中提取所有JSON代码块
            import re
            # 更精确的JSON匹配模式，确保捕获完整的JSON数组
            json_pattern = r'\[\s*{\s*"method"\s*:.*?}\s*\]'
            json_blocks = re.findall(json_pattern, response, re.DOTALL)
            
            if json_blocks:
                # 尝试解析每个代码块为JSON
                operation_chains = []
                for block in json_blocks:
                    try:
                        import json
                        chain = json.loads(block)
                        if isinstance(chain, list) and len(chain) > 0:
                            # 验证每个操作是否有有效的method
                            valid_chain = True
                            for op in chain:
                                if not isinstance(op, dict) or 'method' not in op:
                                    valid_chain = False
                                    break
                            if valid_chain:
                                operation_chains.append(chain)
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e} in block: {block[:50]}...")
                        continue
                
                if operation_chains:
                    print(f"成功分解为{len(operation_chains)}个操作链")
                    return operation_chains
                
            # 如果上面的方法失败，尝试使用LLM直接输出JSON格式
            print("尝试使用LLM直接输出JSON格式")
            enhanced_prompt = f"""
            请将以下复杂指令分解为多个步骤的操作链，并以有效的JSON数组格式输出：
            
            指令: "{goal_description}"
            
            请确保输出格式为有效的JSON数组，包含多个操作链，例如:
            [
                [
                    {{"method": "get_geonode_by_name", "args": ["Davey Road"]}},
                    {{"method": "get_child_nodes", "kwargs": {{"relation_type": "contains"}}}},
                    {{"method": "filter_by_class", "args": ["vehicle"]}}
                ],
                [
                    {{"method": "get_child_nodes", "kwargs": {{"relation_type": "north_of"}}}},
                    {{"method": "filter_by_class", "args": ["vehicle"]}}
                ]
            ]
            
            请直接返回上述格式的JSON数组，不要包含任何其他文本。
            """
            
            response = self.call_response(sysprompt=None, userprompt=enhanced_prompt, image_list=[])
            try:
                # 清理响应，只保留可能的JSON部分
                cleaned_response = re.search(r'\[\s*\[.*\]\s*\]', response, re.DOTALL)
                if cleaned_response:
                    operation_chains = json.loads(cleaned_response.group(0))
                    if isinstance(operation_chains, list) and len(operation_chains) > 0:
                        print(f"第二次尝试成功分解为{len(operation_chains)}个操作链")
                        return operation_chains
            except:
                pass
                
        except Exception as e:
            print(f"分解复杂查询时出错: {str(e)}")
        
        # 如果无法分解，回退到常规方式
        print("无法分解复杂查询，使用常规方式")
        prompt = QUERY_OPERATION_CHAIN_PROMPT.format(instruction=goal_description)
        response = self.call_response(sysprompt=None, userprompt=prompt, image_list=[])
        operation_chain = extract_json_from_msg(response)
        return [operation_chain] if operation_chain else []

    def check_subgoals(self, task):
        if task['strategy'] == 'Navigate':
            self.switch_time = self.controller.timestep
            self.last_movement = self.history['action'][-1] if self.history['action'] else 'stop'
            print(f"Checking Navigate subgoal at timestep {self.controller.timestep}, last movement: {self.last_movement}")
            if self.controller.timestep > 12: #防止陷入NotFound
                self.strategy_distances['Navigate'] = self.controller.pose.xy.dist_to(self.episode.target_position.xy)
                self.strategy_timesteps['Navigate'] = self.controller.timestep
                return True
            # check if any landmarks are reached
            if all(lm.position.xy.dist_to(self.controller.pose.xy) < 40.0 for lm in self.landmark_nav_map.landmark_map.landmarks):
                self.strategy_timesteps['Navigate'] = self.controller.timestep
                return True
            else:
                return False
            #return all(lm.position.xy.dist_to(self.controller.pose.xy) < 40.0 for lm in self.landmark_nav_map.landmark_map.landmarks)
        elif task['strategy'] == 'Search':
            if self.controller.timestep > (self.switch_time + 5): #防止陷入search around
                self.strategy_distances['Search'] = self.controller.pose.xy.dist_to(self.episode.target_position.xy)
                self.strategy_timesteps['Search'] = self.controller.timestep-self.strategy_timesteps['Navigate']
                return True
            # evaluate whether the information has changed
            sem_map = self.landmark_nav_map.get_semantic_map()
            #初始化探索分析器
            if not hasattr(self, 'exploration_analyzer'):
                self.exploration_analyzer = ExplorationAnalyzer(threshold=1e-8)
            # 与上一帧比较（已初始化一张全零的RGB图像张量）
            if self.exploration_analyzer.should_continue(sem_map, self.previous_sem_map):
                self.previous_sem_map = sem_map.copy()
                return False
            else: 
                self.strategy_timesteps['Search'] = self.controller.timestep-self.strategy_timesteps['Navigate']
                return True
        elif task['strategy'] == 'Locate':
            # 利用知识推断目标名称，飞到对应位置
            # 假如没找到目标，还要继续
            return True
        return False
    
    def mark_target_on_global_map(self, map_name: str):
        global_rgb = cropclient.get_global_rgb_map(map_name)
        # 获取目标的世界坐标
        target_x = self.episode.target_object.position.x
        target_y = self.episode.target_object.position.y
    
        # 转换为像素坐标 (col, row)
        col, row = cropclient.world_to_pixel(map_name, target_x, target_y)
        # 边界检查
        H, W = global_rgb.shape[:2]
        if 0 <= col < W and 0 <= row < H:
            cv2.circle(global_rgb, (col, row), 10, (255, 0, 0), -1)  # 半径10更明显
        else:
            print(f"Target ({target_x}, {target_y}) is outside map {map_name}!")
        # 保存带标记的目标位置的全局卫星图
        save_dir = os.path.join("results", "geonav", "hard", "final")
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f'target_{self.episode.id}.png'), cv2.cvtColor(global_rgb, cv2.COLOR_RGB2BGR))

    def mark_final_position_on_global_map(self, map_name: str):
        """
        在全局卫星图上标记:
        - 目标位置: 红点
        - 最终定位点: 蓝点
        """
        global_rgb = cropclient.get_global_rgb_map(map_name)
        
        # ===== 1. 标记目标位置 (红点) =====
        target_x = self.episode.target_object.position.x
        target_y = self.episode.target_object.position.y
        col_target, row_target = cropclient.world_to_pixel(map_name, target_x, target_y)
        
        H, W = global_rgb.shape[:2]
        if 0 <= col_target < W and 0 <= row_target < H:
            cv2.circle(global_rgb, (col_target, row_target), 10, (255, 0, 0), -1)  # 红色
        else:
            print(f"Target ({target_x}, {target_y}) is outside map {map_name}!")
        # ===== 2. 标记最终定位点 (蓝点) =====
        final_x = self.controller.pose.x
        final_y = self.controller.pose.y
        col_final, row_final = cropclient.world_to_pixel(map_name, final_x, final_y)
        
        if 0 <= col_final < W and 0 <= row_final < H:
            cv2.circle(global_rgb, (col_final, row_final), 10, (0, 0, 255), -1)  # 蓝色 (BGR)
        else:
            print(f"Final position ({final_x}, {final_y}) is outside map {map_name}!")
        # 保存结果
        save_dir = os.path.join("results", "geonav", "hard", "final")
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f'final_{self.episode.id}.png'), cv2.cvtColor(global_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved final position map: {self.episode.id}")

    def run(self):
        Success = False
        strategy = ''
        pos_log = []
        image_list = []
        self.strategy_distances['Start'] = self.controller.pose.xy.dist_to(self.episode.target_position.xy)
        #在卫星图上标出目标位置（红点）
        #self.mark_target_on_global_map(self.episode.map_name)
        # 添加终止条件变量
        while self.controller.timestep < self.args.eval_max_timestep:
            #获取当前位姿下的卫星图像，并将其转换为Base64编码，用于后续VLM调用
            rgb, _ = self.controller.perceive(self.controller.pose, self.episode.map_name)
            # print(rgb.shape)
            image_64 = encode_image_from_pil(Image.fromarray(rgb))
            Image.fromarray(rgb).save(args.output_dir+str(self.episode.id) + f'rgb_{self.controller.timestep}.png')
            image_list.append(image_64)
            # TODO: measure the z distance between the camera and the target on the ground
            # dep_img = Image.fromarray(depth.squeeze(), mode='L')  # 'L'表示灰度模式
            # update map with observations
            self.landmark_nav_map.update_observations(
                self.controller.pose, rgb, None, use_gsam_map_cache=False, strategy=strategy
            )
            # detect and memory the scene objects
            detect_mode = 'VLM'  # 'GSAM'
            if args.ablation != 'wo_landmark':
                landmark = self.controller.build_geo_nodes(self.landmark_nav_map.landmark_map.landmarks)
            else:
                landmark = self.controller.build_geo_nodes([])
            #这里返回的landmark是当前无人机姿态和地标之间的位置关系描述（新增）
            #landmark_descriptions = self.controller.build_geo_nodes_describe(self.landmark_nav_map.landmark_map.landmarks)
            # 判断是否需要执行 VLM 检测
            current_position = self.controller.pose.xy
            time_condition = self.controller.timestep - self.last_vlm_detection_time >= self.vlm_detection_interval
            distance_condition = current_position.dist_to(self.last_vlm_detection_position) >= self.vlm_detection_distance
            
            if detect_mode == 'VLM' and strategy == 'Search' and (time_condition or distance_condition):
                # 执行 VLM 检测
                print(f"Executing VLM detection at timestep {self.controller.timestep}")
                subgraph = self.controller.understand(image_64, self.episode)
                # 转换为世界坐标，将局部检测到的子图融进全局场景图
                self.controller.build_scene_graph(subgraph, self.landmark_nav_map.target_map)
                # 保存场景图摘要（使用字典格式，便于JSON保存）
                if not self.results.get("scene_graph_summary"):
                    self.results["scene_graph_summary"] = self.controller.scene_graph.format_summary(as_dict=True)
                # 更新上次检测的时间和位置
                self.last_vlm_detection_time = self.controller.timestep
                self.last_vlm_detection_position = current_position
            elif detect_mode == 'GSAM':
                # Graph memory
                surrounding, recent_objects = self.controller.build_scene_nodes(
                    self.landmark_nav_map.target_map.obj_list, 
                    self.landmark_nav_map.surroundings_map.obj_list, 
                    show=False
                )
            
            # Step 1: system Prompt
            geoinstruct = self.prompts["landmark_prompt"].format(landmark=landmark, recent_objects='', surroundings='')
            self.view_width = 2 * (self.controller.pose.z-self.landmark_nav_map.ground_level)
            #position=(self.controller.pose.x, self.controller.pose.y)
            # Step 2: static planning
            self.xyxy = [(self.controller.pose.x -self.view_width/2, self.controller.pose.y -self.view_width/2),(self.controller.pose.x +self.view_width/2, self.controller.pose.y +self.view_width/2)]

            #img_landmark = self.gen_map(query_type='landmark')
            while self.plan is None:
                planner_prompt = self.prompts["planner_prompt"].format(instruction=self.episode.target_description, geoinstruct=geoinstruct)
                #planner_prompt = self.prompts["planner_prompt"].format(instruction=self.episode.target_description, geoinstruct=geoinstruct, img_landmark=img_landmark)
                print('Retry generating plan')
                response = self.call_response(sysprompt=None, userprompt=planner_prompt, image_list=[])
                #response = self.call_response(sysprompt=None, userprompt=planner_prompt, image_list=[img_landmark])
                #print(f"Planner response(用于检查输出格式): {response}")
                self.plan = extract_json_from_msg(response)
            
            # Step 3：# execute the subtask
            current_task = self.plan['sub_goals'][self.current_task_index]
            print(f"Current Task [{self.current_task_index+1}/{len(self.plan['sub_goals'])}]: {current_task['goal']}")
            print(f" → Strategy: {current_task['strategy']}")
            if self.subtask_status == "pending":
                # initialize the subtask prompt
                self.initialize_subtask(current_task)
                self.subtask_status = "running"
            # Step 4: Choose strategy to exectue the goal
            # function: using different map type as different sacle observation
            #next_pos = self.execute_subtask(current_task, geoinstruct, landmark_descriptions, position, image_64)
            next_pos = self.execute_subtask(current_task, geoinstruct, image_64)
            self.controller.pose = Pose4D(next_pos[0],next_pos[1],self.controller.pose.z, self.controller.pose.yaw) # move(self.controller.pose, Point2D(next_pos[0], next_pos[1]), 10, _)
            strategy = current_task['strategy'] # temp
            if self.controller.reached_target(self.controller.pose, self.target):
                print("Target reached.")
            # Step 5: check the subgoals
            if self.check_subgoals(current_task):
                self.subtask_status = "completed"
                self.current_task_index += 1
                print(f"Subtask {current_task['goal']} is completed.")
                # Check that all tasks are complete
                if self.current_task_index >= len(self.plan['sub_goals']):
                    print("All sub-tasks completed!")
                    self.strategy_distances['Locate'] = self.controller.pose.xy.dist_to(self.episode.target_position.xy)
                    pos_log.append(self.controller.pose)
                    self.results["steps"].append({
                    "time_step": self.controller.timestep,
                    "pose": (self.controller.pose.x, self.controller.pose.y, self.controller.pose.z, self.controller.pose.yaw),
                    "distance_to_target": self.controller.pose.xy.dist_to(self.target.xy),
                    "plan": current_task,
                    "observation_suggestion":self.observation,
                    "action_suggestion": self.action,
                })
                    break  # 提前退出循环
            self.results["steps"].append({
                "time_step": self.controller.timestep,
                "pose": (self.controller.pose.x, self.controller.pose.y, self.controller.pose.z, self.controller.pose.yaw),
                "distance_to_target": self.controller.pose.xy.dist_to(self.target.xy),
                "plan": current_task,
                "observation_suggestion":self.observation,
                "action_suggestion": self.action,
            })
            # 记录当前位置
            pos_log.append(self.controller.pose)
        if self.controller.reached_target(self.controller.pose, self.target):
            print("Target reached.")
            Success = True
        
        # 将策略步数信息添加到 results 中
        self.results["strategy_timesteps"] = self.strategy_timesteps
        
        # 在全局卫星图上标记最终位置和目标位置
        self.mark_final_position_on_global_map(self.episode.map_name)
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

    def recursive_query(self, operation_chain, max_depth=5, current_depth=0):
        """递归查询以获取与任务相关的子图"""
        # 检查递归深度
        if current_depth >= max_depth:
            print(f"Reached the maximum recursion depth {max_depth}, stopped recursion")
            return []
        
        # 检查operation_chain是否为空
        if not operation_chain:
            print("The operation chain is empty and the query cannot be executed.")
            return []
        
        # 检查是否为多个操作链的情况
        if isinstance(operation_chain, list) and len(operation_chain) > 0 and isinstance(operation_chain[0], list):
            print(f"A multi-step operation chain was detected ({len(operation_chain)} steps)")
            return self.recursive_multi_step_query(operation_chain, max_depth, current_depth)
        
        try:
            # 验证操作链中的关系类型
            for op in operation_chain:
                if not isinstance(op, dict):
                    print(f"Warning: Non-dictionary object in operation chain: {op}")
                    continue
                    
                if op['method'] == 'get_child_nodes' and 'kwargs' in op and 'relation_type' in op['kwargs']:
                    relation_type = op['kwargs']['relation_type']
                    valid_relations = [
                        "contains", "adjacent_to", "near_corner", 
                        "north_of", "south_of", "east_of", "west_of",
                        "northeast_of", "northwest_of", "southeast_of", "southwest_of"
                    ]
                    if relation_type not in valid_relations:
                        print(f"Warning: Invalid relation type '{relation_type}'. Using 'contains' instead.")
                        op['kwargs']['relation_type'] = "contains"
            
            # 使用增强版的robust_subgraph_query替代原始的subgraph_query
            current_nodes = self.controller.query_engine.robust_subgraph_query(
                operation_chain, 
                fallback=True,  # 启用回退机制
                min_results=1,  # 最小结果数量
                debug=True      # 打印调试信息
            )
            
            if not current_nodes:
                print("No nodes found with the current operation chain.")
                return None

            # 打印找到的节点信息，帮助调试
            print(f"Found {len(current_nodes)} nodes:")
            for node in current_nodes:
                print(f"  - {node.id} ({node.type})")
            
            # if len(current_nodes) > 1:
            #     print(f"找到多个节点({len(current_nodes)})，需要进一步筛选")
            #     # 使用LLM询问答案
            #     continue_query = self.ask_llm_to_continue(current_nodes)
            #     return continue_query
            # else:
            return current_nodes
        except Exception as e:
            print(f"Error in recursive_query: {e}")
            # 返回一个默认节点或者None
            return None

    # def recursive_query(self, operation_chain, max_depth=5, current_depth=0):
    #     """递归查询以获取与任务相关的子图"""
    #     # 检查递归深度
    #     if current_depth >= max_depth:
    #         print(f"Reached the maximum recursion depth {max_depth}, stopped recursion")
    #         return []
        
    #     # 检查operation_chain是否为空
    #     if not operation_chain:
    #         print("The operation chain is empty and the query cannot be executed.")
    #         return []
        
    #     # 检查是否为多个操作链的情况
    #     if isinstance(operation_chain, list) and len(operation_chain) > 0 and isinstance(operation_chain[0], list):
    #         print(f"A multi-step operation chain was detected ({len(operation_chain)} steps)")
    #         return self.recursive_multi_step_query(operation_chain, max_depth, current_depth)
        
    #     try:
    #         # 验证操作链中的关系类型
    #         for op in operation_chain:
    #             if not isinstance(op, dict):
    #                 print(f"Warning: Non-dictionary object in operation chain: {op}")
    #                 continue
                    
    #             if op['method'] == 'get_child_nodes' and 'kwargs' in op and 'relation_type' in op['kwargs']:
    #                 relation_type = op['kwargs']['relation_type']
    #                 valid_relations = [
    #                     "contains", "overlaps", "separates", 
    #                     "north_of", "south_of", "east_of", "west_of",
    #                     "northeast_of", "northwest_of", "southeast_of", "southwest_of"
    #                 ]
    #                 if relation_type not in valid_relations:
    #                     print(f"Warning: Invalid relation type '{relation_type}'. Using 'contains' instead.")
    #                     op['kwargs']['relation_type'] = "contains"
            
    #         # 使用增强版的robust_subgraph_query替代原始的subgraph_query
    #         current_nodes, query_logs = self.controller.query_engine.robust_subgraph_query(
    #             operation_chain, 
    #             fallback=True,  # 启用回退机制
    #             min_results=1,  # 最小结果数量
    #             debug=True      # 打印调试信息
    #         )
            
    #         # 保存查询日志
    #         if query_logs:
    #             self.results["query_logs"].extend(query_logs)
            
    #         if not current_nodes:
    #             print("No nodes found with the current operation chain.")
    #             return None

    #         # 打印找到的节点信息，帮助调试
    #         print(f"Found {len(current_nodes)} nodes:")
    #         for node in current_nodes:
    #             print(f"  - {node.id} ({node.type})")
            
    #         # if len(current_nodes) > 1:
    #         #     print(f"找到多个节点({len(current_nodes)})，需要进一步筛选")
    #         #     # 使用LLM询问答案
    #         #     continue_query = self.ask_llm_to_continue(current_nodes)
    #         #     return continue_query
    #         # else:
    #         return current_nodes
    #     except Exception as e:
    #         print(f"Error in recursive_query: {e}")
    #         # 返回一个默认节点或者None
    #         return None
            
    def recursive_multi_step_query(self, operation_chains, max_depth=5, current_depth=0):
        """处理多步骤操作链的递归查询"""
        # 检查递归深度
        if current_depth >= max_depth:
            print(f"The multi-step query reaches the maximum recursion depth {max_depth}, and stops recursion")
            return []
            
        current_nodes = None
        all_nodes = []
        
        for i, chain in enumerate(operation_chains):
            try:
                print(f"Execute step {i+1}/{len(operation_chains)}")
                
                # 如果是后续步骤，且需要使用前一步骤的结果
                if i > 0 and current_nodes and len(current_nodes) > 0:
                    # 检查是否第一个操作是get_child_nodes，如果是，则使用当前节点集合
                    if len(chain) > 0 and chain[0]['method'] == 'get_child_nodes':
                        # 此处使用前一步骤的结果作为父节点
                        print(f"Use the result of the previous step ({len(current_nodes)} nodes) as the parent node")
                        
                        # 创建临时查询引擎
                        temp_current_nodes, temp_logs = self.controller.query_engine.robust_subgraph_query(
                            chain,
                            fallback=True,
                            min_results=1,
                            debug=True
                        )
                        if temp_logs:
                            self.results["query_logs"].extend(temp_logs)
                        
                        if temp_current_nodes:
                            current_nodes = temp_current_nodes
                            all_nodes.extend(current_nodes)
                            print(f"Step {i+1} finds {len(current_nodes)} nodes")
                        else:
                            print(f"Step {i+1} did not find the node")
                    else:
                        # 如果第一个操作不是get_child_nodes，则单独执行此链
                        temp_current_nodes, temp_logs = self.controller.query_engine.robust_subgraph_query(
                            chain,
                            fallback=True,
                            min_results=1,
                            debug=True
                        )
                        if temp_logs:
                            self.results["query_logs"].extend(temp_logs)
                        
                        if temp_current_nodes:
                            current_nodes = temp_current_nodes
                            all_nodes.extend(current_nodes)
                            print(f"Step {i+1} finds {len(current_nodes)} nodes")
                        else:
                            print(f"Step {i+1} did not find the node")
                else:
                    # 对于第一个步骤，或者前面步骤没有结果的情况
                    temp_current_nodes, temp_logs = self.controller.query_engine.robust_subgraph_query(
                        chain,
                        fallback=True,
                        min_results=1,
                        debug=True
                    )
                    if temp_logs:
                        self.results["query_logs"].extend(temp_logs)
                    
                    if temp_current_nodes:
                        current_nodes = temp_current_nodes
                        all_nodes.extend(current_nodes)
                        print(f"Step {i+1} finds {len(current_nodes)} nodes")
                    else:
                        print(f"Step {i+1} did not find the node")
                
            except Exception as e:
                print(f"Error in step {i+1}: {str(e)}")
                continue
        
        # 如果有多个步骤的结果，使用LLM筛选最符合原始查询的结果
        if len(all_nodes) > 0:
            if len(all_nodes) > len(current_nodes or []):
                print(f"Multi-step query finds {len(all_nodes)} total nodes, the last step has {len(current_nodes or [])} nodes")
                # 可以选择返回all_nodes或current_nodes
                # 这里选择返回最后一步的结果
                return current_nodes
            else:
                return current_nodes
        else:
            print("No nodes found for all steps")
            return None
    
    def ask_llm_to_continue(self, nodes):
        node_descriptions = []
        for node in nodes:
            desc = {
                "id": node.id,
                "type": node.type.upper(),
                "class": getattr(node, 'obj_class', 'N/A'),
                "position": (round(node.position.x,1), round(node.position.y,1)),
                "confidence": getattr(node, 'confidence', 1.0),
                "is_target": getattr(node, 'target', False),
                "attributes": {k:v for k,v in vars(node).items() 
                            if k not in ['id','type','position','confidence','target']}
            }
            node_descriptions.append(json.dumps(desc, ensure_ascii=False))
        # 构建边关系描述
        edge_descriptions = []
        for node in nodes:
            for edge in self.controller.query_engine.graph.digraph_nx.out_edges(node.id, data=True):
                edge_info = {
                    "source": edge[0],
                    "target": edge[1],
                    "relation": edge[2].get('relation_type', 'unknown'),
                    "attributes": {k:v for k,v in edge[2].items() if k != 'relation_type'}
                }
                edge_descriptions.append(json.dumps(edge_info, ensure_ascii=False))
        # 构建提示
        newline = '\n'
        prompt = f"""
        ## task
        We're looking for a target: {self.episode.target_description}

        ## Current Scene graph
        ### Current nodes:
        {newline.join(node_descriptions[:5])}{'...' if len(nodes) > 5 else ''}

        ### Current edges:
        {newline.join(edge_descriptions[:5])}{'...' if len(edge_descriptions) > 5 else ''}
        Which node is the closest to the target? 
        Please output the node id in a formatted JSON object with key "node_id". 
        For example:
        {{
            "node_id": "vehicle_1"
        }}
        Make sure the node_id format is correct.
        """
        response = self.call_response(sysprompt=None, userprompt=prompt, image_list=[]).strip()
        try:
            output_json = json.loads(response)
            node_id = output_json.get("node_id")
            for node in nodes:
                if node.id == node_id:
                    # 如果找到匹配的节点，返回该节点
                    return node
        except json.JSONDecodeError:
            # 如果解析失败，返回nodes[0]作为默认值
            return None


    def generate_operation_chain_from_nodes(self, nodes):
        """根据当前节点生成新的操作链"""
        # 生成描述当前节点的文本
        node_descriptions = [f"Node {n.id} of type {n.type}" for n in nodes]
        node_text = ", ".join(node_descriptions)
        
        # 生成提示
        prompt = f"""
        Based on the current nodes: {node_text}

        Generate a new operation chain to further refine the search for the target object.
        Available operations:
        - get_child_nodes(parent, relation_type): Gets the child node with the specified relationship to the parent node.
        The available relationship types are: "contains", "adjacent_to", "near_corner", "north_of", "south_of", "east_of", "west_of", "northeast_of", "northwest_of", "southeast_of", "southwest_of"
        - filter_by_class(obj_class): Filter object nodes by class.
        - filter_by_attribute(key, value): Filter object nodes by attribute.

        Return the operation chain in JSON format.
        """
        
        # 调用LLM生成新的操作链
        response = self.call_response(sysprompt=None, userprompt=prompt, image_list=[])
        operation_chain = extract_json_from_msg(response)
        
        if not operation_chain:
            # 如果LLM未能生成有效的操作链，返回一个默认的操作链
            return []
        
        return operation_chain

