from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.space import Pose4D, Point2D
from gsamllavanav.dataset.episode import Episode
from scenegraphnav.llm_controller import LLMController # only used for our scenegraphnav methods
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from scenegraphnav.prompt.navgpt import *
from gsamllavanav.actions import DiscreteAction

class Agent(object):
    ''' Base class for an agent to generate and save trajectories. '''
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode):
        self.name = None
        self.memory = []
        self.args = args
        self.pose = initial_pose
        self.episode = episode  # 存储episode信息

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
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode):
        super().__init__(args, initial_pose, episode)
        self.episode = episode
        self.controller = LLMController(args, initial_pose)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "/data1/FoundationModels/Qwen",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            "/data1/FoundationModels/Qwen", min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.prompts = self.init_prompts()

    def init_prompts(self):
        """
        初始化所有任务相关的 Prompt 模板。
        """
        return {
            "task_description": TASK_DESCRIPTION_PROMPT,
            "action_prompt": ACTION_PROMPT,
            "planner_prompt": PLANNER_PROMPT,
            "history_prompt": HISTORY_PROMPT,
        }

    def run(self):
        """
        主循环控制逻辑：感知、生成 Prompt、推理动作、执行动作、更新场景图。
        """
        while not self.controller.reached_target(self.controller.pose, self.target):
            # Step 1: 感知环境
            rgb, depth = self.controller.perceive(self.controller.pose, self.episode.map_name)
            rgb_img = Image.fromarray(rgb)
            dep_img = Image.fromarray(depth)  # depth 数据暂时未使用

            # Step 2: 生成 Prompt 并推理动作
            prompt_inputs = self.generate_prompt(rgb_img)
            action_suggestion = self.get_action_suggestion(prompt_inputs)

            # Step 3: 执行动作并更新位置
            self.controller.pose = self.controller.act(self.controller.pose, action_suggestion)

            # Step 4: 更新场景图
            scene_graph = self.controller.build_scene_graph(self.controller.args, self.controller.pose)
            self.process_scene_graph(scene_graph)

    def generate_prompt(self, image):
        """
        基于 RGB 图像和任务描述生成 LLM 所需的 Prompt 输入。
        """
        # 将任务描述和动作提示合并
        prompt_template = self.prompts["task_description"] + "\n" + self.prompts["action_prompt"]
        task_prompt = self.prompts["task_description"].format(self.episode.instruction)

        # 构建 LLM 输入对话格式
        conversation = [
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": prompt_template},
        ]

        # 转换为处理器格式
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        return inputs.to("cuda")

    def get_action_suggestion(self, inputs):
        """
        使用 VLM 模型推理下一步 UAV 动作。
        """
        # 生成 LLM 输出
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        # 解码生成的动作
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(f"Generated Action Suggestion: {output_text}")
        return output_text[0]  # 假设返回单步动作

    def process_scene_graph(self, scene_graph):
        """
        更新场景图逻辑（可扩展为记录日志、可视化或存储）。
        """
        # 示例逻辑（根据具体需求完善）
        print(f"Processed Scene Graph: {scene_graph}")
