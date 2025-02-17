from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.space import Pose4D, Point2D
from gsamllavanav.dataset.episode import Episode
from scenegraphnav.llm_controller import LLMController # only used for our scenegraphnav methods

import re
from openai import OpenAI
from ggb.QwenAPI import encode_image_from_pil
from PIL import Image
from scenegraphnav.prompt.navgpt import *
from scenegraphnav.prompt.rationales import *
from gsamllavanav.actions import DiscreteAction
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os
import json
from datetime import datetime
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap


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

class MapAgent(Agent):
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode):
        super().__init__(args, initial_pose, episode)


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
        # API model
        elif isinstance(vlmodel, OpenAI):
            self.model = vlmodel
        else:
            raise ValueError("model must be realized")
        
        self.prompts = self.init_prompts()
        self.history = ''
        self.previous_action = ''
        self.results = {
            "target": (self.target.x, self.target.y),
            "steps": []
        }
        self.landmark_nav_map = LandmarkNavMap(
            episode.map_name, args.map_shape, args.map_pixels_per_meter, 
            episode.description_landmarks, episode.description_target, episode.description_surroundings, args.gsam_params
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
            "rationale": RATIONALE,
            "object_caption": OBSERVATION_CAPTION,
            "object_grounding":OBJECT_GROUNDING,
        }

    def run(self):
        t = 0
        Success = False
        # Parse the instruction
        task_prior_knowledge = self.controller.parse_instruction(self.episode.target_description)
        while not self.controller.reached_target(self.controller.pose, self.target) and t < self.args.eval_max_timestep:
            t += 1
            print(f"Step {t}")
            rgb, depth = self.controller.perceive(self.controller.pose, self.episode.map_name)
            rgb_img = Image.fromarray(rgb)
            # dep_img = Image.fromarray(depth.squeeze(), mode='L')  # 'L'表示灰度模式
            # measure the z distance between the camera and the target on the ground
            
            # update semantic map
            self.landmark_nav_map.update_observations(
                self.controller.pose, rgb, None, use_gsam_map_cache=False
            )
            # TODO: Add an AOI generation module
            semap_img = self.landmark_nav_map.plot(
                goal_description=self.episode.description_target,
                start_point=self.episode.start_pose.xy,
                current_pose=self.controller.pose,
                show=False
            )

            # create a module for inferencing the area of interest from the semantic map
            # Assuming you have structured prior knowledge of the task in a dictionary format
            # task_prior_knowledge = {
            #     "landmarks": self.episode.description_landmarks,
            #     "target": self.episode.description_target,
            #     "surroundings": self.episode.description_surroundings
            # }


            # Integrate the prior knowledge with the semantic map
            AOI_TEXT = self.landmark_nav_map.integrate_prior_knowledge(task_prior_knowledge)

            # Generate the area of interest (AOI) based on the integrated map and prior knowledge
            aoi = self.landmark_nav_map.generate_aoi(AOI_TEXT, semap_img)

            # Visualize or log the AOI for debugging purposes
            print(f"Generated AOI: {aoi}")


            if isinstance(self.model, Qwen2VLForConditionalGeneration):
                # Step 1: 生成 Prompt 并推理 Observation
                prompt_inputs = self.generate_prompt(rgb_img, semap_img, mode="observation")
                self.observation = self.get_response(prompt_inputs)
                
                prompt_inputs = self.generate_prompt(rgb_img, semap_img, mode="object_caption_only")
                self.object_list = self.get_response(prompt_inputs)

                prompt_inputs = self.generate_prompt(rgb_img, semap_img, mode="object_grounding_only")
                self.object_bbox = self.get_response(prompt_inputs)

                # Step 2: 生成 Prompt 并推理动作
                prompt_inputs = self.generate_prompt(rgb_img, semap_img, mode="action")
                self.previous_action = self.get_response(prompt_inputs)

                action_suggestion = String2DisActionList(self.previous_action)
                print(f"Action Suggestion: {action_suggestion}")
            elif isinstance(self.model, OpenAI):
                # Step 1: 生成 Prompt 并推理 Observation
                task_prompt = self.prompts["task_description"].format(instruction=self.episode.target_description)
                
                observation_prompt = self.prompts["observation_prompt"].format(instruction=self.episode.target_description)
                observation_response = self.model.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": observation_prompt}
                    ],
                    max_tokens=150
                )
                self.observation = observation_response.choices[0].message['content'].strip()

                # Step 2: 生成 Prompt 并推理动作
                action_prompt = self.prompts["action_prompt"].format(history=self.history, observation=self.observation)
                action_response = self.model.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": action_prompt}
                    ],
                    max_tokens=150
                )
                self.previous_action = action_response.choices[0].message['content'].strip()

                action_suggestion = String2DisActionList(self.previous_action)
                print(f"Action Suggestion: {action_suggestion}")
                # Step 3: Process the action suggestion
                if action_suggestion:
                    action = action_suggestion[0]
                else:
                    action = DiscreteAction.STOP

                # Perform the action
                self.controller.pose = self.controller.act(self.controller.pose, action)

            # 记录当前步骤的信息
            self.results["steps"].append({
                "time_step": t,
                "pose": (self.controller.pose.x, self.controller.pose.y, self.controller.pose.z, self.controller.pose.yaw),
                "distance_to_target": self.controller.pose.xy.dist_to(self.target.xy),
                "action_suggestion": self.previous_action
            })

            self.controller.pose = self.controller.act(self.controller.pose, action_suggestion)

            # Step 4: 记录历史
            self.history = self.prompts["history_prompt"].format(history=self.history, observation=self.observation, previous_action=self.previous_action)
            print(f"History: {self.history}")
            # Step 5: 更新场景图
            scene_graph = self.controller.build_scene_graph(self.controller.args, self.controller.pose)
            self.process_scene_graph(scene_graph)

        if t != self.args.eval_max_timestep:
            print("Target reached.")
            Success = True

        self.save_results(Success)
        return Success

    def save_results(self, success):
        # 生成唯一的文件名，包含set_height信息
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        height_info = f"height_{self.set_height}" if self.set_height is not None else "default_height"
        filename = f"results_{self.episode.id}_{height_info}_{timestamp}.json"
        filepath = os.path.join("results", filename)

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

    def process_scene_graph(self, scene_graph):
        """
        更新场景图逻辑（可扩展为记录日志、可视化或存储）。
        """
        # 示例逻辑（根据具体需求完善）
        print(f"Processed Scene Graph: {scene_graph}")
