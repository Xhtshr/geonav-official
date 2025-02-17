import torch
from scenegraphnav.parser import parse_args
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.models.goal_predictor import GoalPredictor
from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose
from gsamllavanav.observation import cropclient
# from scenegraphnav.city_scene_graph import build_scene_graph
from scenegraphnav.landmark_search import landmark_loc
from gsamllavanav.actions import DiscreteAction
import numpy as np

class LLMController:
    def __init__(self, args: ExperimentArgs, pose: Pose4D):
        self.args = args
        self.pose = pose

    def perceive(self, pose: Pose4D, map_name: str):
        # 使用感知模块处理RGB和深度图像
        rgb = cropclient.crop_image(map_name, pose, (1024, 1024), 'rgb')
        depth = cropclient.crop_image(map_name, pose, (1024, 1024), 'depth') / self.args.max_depth
        return rgb, depth

    def understand(self, instruction: str):
        # 从指令中提取地标和目标信息，使用自然语言处理模型来解析指令
        landmarks, target = self.extract_landmarks_and_target(instruction)
        return landmarks, target

    def act(self, pose: Pose4D, actions: list):

        for action in actions:
            # 执行动作并更新位置
            new_pose = _moved_pose(pose, *action.value)
        
        return new_pose

    def reached_target(self, pose: Pose4D, target: Point2D):
        # 判断是否到达目标
        print('dist:', pose.xy.dist_to(target.xy))
        return pose.xy.dist_to(target.xy) < self.args.success_dist -10.0

    def build_scene_graph(self, args: ExperimentArgs, pose: Pose4D):
        pass
        # # 构建场景图
        # scene_graph = build_scene_graph(args, pose)
        # return scene_graph

    def parse_instruction(self, instruction: str):
        # 解析指令，提取地标和目标
        from scenegraphnav.prompt.instruction import create_prompt, gpt_api_call, parse_response
        prompt = create_prompt(instruction)
        response = gpt_api_call(prompt)
        self.intr_knowledge = parse_response(response)
        return self.intr_knowledge
