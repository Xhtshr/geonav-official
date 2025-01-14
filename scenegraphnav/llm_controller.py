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

class LLMController:
    def __init__(self, args: ExperimentArgs, predictor: GoalPredictor, device: str):
        self.args = args
        self.predictor = predictor
        self.device = device

    def perceive(self, pose: Pose4D, map_name: str):
        # 使用感知模块处理RGB和深度图像
        rgb = cropclient.crop_image(map_name, pose, (224, 224), 'rgb')
        depth = cropclient.crop_image(map_name, pose, (256, 256), 'depth') / self.args.max_depth
        return rgb, depth

    def understand(self, instruction: str):
        # 从指令中提取地标和目标信息，使用自然语言处理模型来解析指令
        landmarks, target = self.extract_landmarks_and_target(instruction)
        return landmarks, target

    def act(self, pose: Pose4D, target: Point2D):
        # 使用路径规划和场景图进行移动
        action = lookahead_discrete_action(pose, [target])
        new_pose = _moved_pose(pose, *action.value)
        return new_pose

    def run(self, episodes: list[Episode]):
        for episode in episodes:
            pose = episode.start_pose
            landmarks, target = self.understand(episode.instruction)

            while not self.reached_target(pose, target):
                rgb, depth = self.perceive(pose, episode.map_name)
                # 这里可以调用理解模块的其他功能
                pose = self.act(pose, target)

    def reached_target(self, pose: Pose4D, target: Point2D):
        # 判断是否到达目标
        return pose.xy.dist_to(target) < self.args.success_dist

    def extract_landmarks_and_target(self, instruction: str):
        # 解析指令，提取地标和目标
        return [], Point2D(0, 0)

