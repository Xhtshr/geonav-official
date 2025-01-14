from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.space import Pose4D, Point2D
from gsamllavanav.dataset.episode import Episode
from scenegraphnav.llm_controller import LLMController

class Agent:
    def __init__(self, args: ExperimentArgs, initial_pose: Pose4D, episode: Episode):
        # 初始化Agent，创建一个LLMController实例
        self.controller = LLMController(args, initial_pose)
        self.episode = episode  # 存储episode信息
        self.target = None  # 初始化目标为None
        self.scene_graph = None  # 初始化场景图为None

    def set_target(self, target: Point2D):
        # 设置Agent的目标位置（可以是landmark坐标，也可以是CV模型提取出的waypoint位置）
        self.target = target

    def run(self):
        # 运行Agent的感知-思维-动作循环，直到到达目标
        while not self.controller.reached_target(self.controller.pose, self.target):
            # 感知环境，获取RGB和深度图像
            rgb, depth = self.controller.perceive(self.controller.pose, self.episode.map_name)
            # 
            # 执行动作，更新位置
            self.controller.pose = self.controller.act(self.controller.pose, self.target)
            # 构建场景图
            self.scene_graph = self.controller.build_scene_graph(self.controller.args, self.controller.pose)
            # 处理场景图（例如，更新目标位置等）
            self.process_scene_graph(self.scene_graph)

    def process_scene_graph(self):
        # 处理场景图的逻辑
        # 这里可以实现对场景图的分析和决策
        pass 