from typing import Optional
import re
import cv2
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


from utils.QwenAPI import encode_image_from_pil
from gsamllavanav.observation import cropclient
from gsamllavanav.defaultpaths import GSAM_MAPS_DIR
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.dataset.episode import Episode
from scenegraphnav.city_scene_graph import QueryEngine

from .map import Map
from .tracking_map import TrackingMap
from .landmark_map import LandmarkMap
from .gsam_map import GSamMap, GSamParams


def _fig_to_pil(fig):
    """Convert a Matplotlib figure to a PIL.Image in RGB format with multiple fallbacks.

    This handles backends where `fig.canvas.tostring_rgb()` may not exist (raises AttributeError).
    """
    # Ensure the canvas is rendered
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    # Try the common API first
    try:
        buf = fig.canvas.tostring_rgb()
        return Image.frombytes('RGB', (width, height), buf)
    except Exception:
        # Fallback: try buffer_rgba -> convert to RGB
        try:
            buf = fig.canvas.buffer_rgba()
            # buffer_rgba returns RGBA bytes
            im = Image.frombuffer('RGBA', (width, height), buf, 'raw', 'RGBA', 0, 1)
            return im.convert('RGB')
        except Exception:
            # Last resort: try renderer's buffer_rgba
            try:
                renderer = fig.canvas.get_renderer()
                buf = renderer.buffer_rgba()
                im = Image.frombuffer('RGBA', (width, height), buf, 'raw', 'RGBA', 0, 1)
                return im.convert('RGB')
            except Exception:
                raise


class LandmarkNavMap(Map):
    """
    Map_type: 
        -plot.py: w/o annotation (remove lanmark anno), landmark, semantic, TopV, TopV+annotation
    """
    def __init__(
        self,
        map_name: str,
        map_shape: tuple[int, int],#地图尺寸
        map_pixels_per_meter: float,#比例尺
        landmark_names: list[str],
        target_name: str, surroundings_names: list[str],
        gsam_params: GSamParams,
        id: tuple = {},#存在风险，tuple是元组，{}代表字典
        grid_size_meters: float = 20.0,
        save_path: str = 'results/geonav',
    ):
        super().__init__(map_name, map_shape, map_pixels_per_meter)#继承父类
        self.step = 0
        self.id = id
        self.save_path = save_path
        self.grid_size_meters = grid_size_meters
        self.grid_size_pixels = int(grid_size_meters * map_pixels_per_meter)
        # 预计算网格边界
        self.grid_x_min = - map_shape[1] // 2 / map_pixels_per_meter
        self.grid_y_min = -map_shape[0] // 2 / map_pixels_per_meter

        # todelete: 地图历史
        self.history_actions = ['None']
        self.history_AOI = ['None']
        self.trajectory = [] # 存储历史Pose4D

        self.tracking_map = TrackingMap(map_name, map_shape, map_pixels_per_meter)
        self.landmark_map = LandmarkMap(map_name, map_shape, map_pixels_per_meter, landmark_names)
        self.target_map = GSamMap(map_name, map_shape, map_pixels_per_meter, [target_name], gsam_params, layer='target')
        self.surroundings_map = GSamMap(map_name, map_shape, map_pixels_per_meter, surroundings_names, gsam_params, layer='surroundings')
    
    def update_observations(
        self,
        camera_pose: Pose4D,
        rgb: np.ndarray,
        depth_perspective: Optional[np.ndarray] = None,
        use_gsam_map_cache=True,
        strategy='',
    ):
        self.trajectory.append(camera_pose)
        self.tracking_map.mark_current_view_area(camera_pose)
        if use_gsam_map_cache:
            self.target_map.update_from_map_cache(camera_pose)
            self.surroundings_map.update_from_map_cache(camera_pose)
        else:
            self.target_map.update_observation(camera_pose, rgb[..., ::-1], depth_perspective, strategy=strategy)
            self.surroundings_map.update_observation(camera_pose, rgb[..., ::-1], depth_perspective, strategy=strategy)

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.concatenate([
            self.tracking_map.to_array(dtype),
            self.landmark_map.to_array(dtype),
            self.target_map.to_array(dtype),
            self.surroundings_map.to_array(dtype),
        ])

    @classmethod
    def generate_maps_for_an_episode(
        cls,
        episode: Episode,
        map_shape: tuple[int, int],
        pixels_per_meter: float,
        update_interval: int,
        image_shape: tuple[int, int],
        gsam_params: GSamParams,
        use_gsam_map_cache=True,
    ):
        trajectory = episode.sample_trajectory(update_interval)

        # tracking map
        tracking_map = TrackingMap(episode.map_name, map_shape, pixels_per_meter)
        tracking_maps = np.stack([tracking_map.mark_current_view_area(pose).to_array() for pose in trajectory])
        assert tracking_maps.shape == (len(trajectory), 2, *map_shape)

        # landmark maps
        landmark_map = LandmarkMap(episode.map_name, map_shape, pixels_per_meter, episode.target_processed_description.landmarks)
        landmark_maps = np.tile(landmark_map.to_array(), (len(trajectory), 1, 1, 1))
        assert landmark_maps.shape == (len(trajectory), 1, *map_shape)

        # target & object maps
        target_map = GSamMap(episode.map_name, map_shape, pixels_per_meter, [episode.target_processed_description.target], gsam_params)
        surrounding_map = GSamMap(episode.map_name, map_shape, pixels_per_meter, episode.target_processed_description.surroundings, gsam_params)
        
        if use_gsam_map_cache:
            target_maps = np.stack([target_map.update_from_map_cache(pose).to_array() for pose in trajectory])
            surrounding_maps = np.stack([surrounding_map.update_from_map_cache(pose).to_array() for pose in trajectory])
        else:
            cropclient.load_image_cache()
            bgrs = [cropclient.crop_image(episode.map_name, pose, image_shape, 'rgb')[..., ::-1] for pose in trajectory]
            target_maps = np.stack([target_map.update_observation(pose, bgr).to_array() for pose, bgr in zip(trajectory, bgrs)])
            surrounding_maps = np.stack([surrounding_map.update_observation(pose, bgr).to_array() for pose, bgr in zip(trajectory, bgrs)])
        
        gsam_maps = np.concatenate((target_maps, surrounding_maps), axis=1)
        assert gsam_maps.shape == (len(trajectory), 2, *map_shape)

        episode_maps = np.concatenate((tracking_maps, landmark_maps, gsam_maps), axis=1)
        assert episode_maps.shape == (len(trajectory), 5, *map_shape)
        return episode_maps
    
    @classmethod
    def from_array(
        cls,
        map_name: str,
        map_shape: tuple[int, int],
        map_pixels_per_meter: float,
        landmark_names: list[str],
        target_name: str,
        object_names: list[str],
        map_data: np.ndarray,
    ):
        nav_map = LandmarkNavMap(map_name, map_shape, map_pixels_per_meter, landmark_names, target_name, object_names)
        nav_map.tracking_map.current_view_area = map_data[0].astype(np.uint8)
        nav_map.tracking_map.explored_area = map_data[1].astype(np.uint8)
        nav_map.landmark_map.landmark_map = map_data[2].astype(np.uint8)
        nav_map.target_map.gsam_map = map_data[3]
        nav_map.surroundings_map.gsam_map = map_data[4]

        return nav_map
    
    def _create_position_marker(self, point: Point2D) -> np.ndarray:
        """创建位置标记图层"""
        return cv2.circle(
            np.zeros(self.shape, dtype=np.float32),
            self.to_row_col(point)[::-1],
            radius=4, color=1, thickness=-1
        )
    def _draw_layer(self, ax, layer, rgba):
        """绘制单个透明图层"""
        color_map = np.zeros((*layer.shape, 4), dtype=np.float32)
        color_map[..., :3] = rgba[:3]
        color_map[..., 3] = layer * rgba[3]
        ax.imshow(color_map, alpha=0.5)

    def _merge_color_layers(self, ax, map_type) -> np.ndarray:
        """优化版图层合并：关键元素优先显示+智能透明度控制"""
        merged = np.zeros((*self.shape, 4), dtype=np.float32)
        if map_type == 'landmark' or map_type == 'TopV':
            # 仅显示地标层
            layers = [
            (self.landmark_map.color_array, 1.0) # 地标层不透明
        ]
        # elif map_type == 'semantic':
        #     layers = [
        #     (self.surroundings_map.color_array, 0.75),  # 环境层中等透明度
        #     (self.target_map.color_array, 0.75)         # 目标层中等透明度
        # ]
        # # 调整叠加顺序：底层元素先绘制，关键元素最后叠加
        else:
            layers = [
                (self.landmark_map.color_array, 1.0),      # 地标层较低透明度
                (self.surroundings_map.color_array, 0.75),  # 环境层中等透明度
                (self.target_map.color_array, 0.75)         # 目标层中等透明度
            ]

        for color_array, alpha in layers:
            if color_array is None:
                continue
            ax.imshow(color_array, alpha=alpha)
    
        return merged

    def _annotate_landmarks(self, ax):
        """添加地标文字标注"""
        for lm in self.landmark_map.landmarks:
            x, y = self.to_row_col(lm.position.xy)
            ax.text(y, x, lm.name, 
                fontsize=10, color='black',
                ha='center', va='center',
                bbox=dict(facecolor='white', 
                            alpha=0.8, 
                            boxstyle='round'))
    
    def _annotate_objects(self, ax, query_engine: QueryEngine, current_pos: Point2D):
        """动态标注物体节点"""
        context = query_engine.get_context(current_pos, radius=500.0)
        for item in context:
            if item['type'] == 'object':  # 只处理物体节点
                obj_id = item.get('id', 'Unknown')
                target = item.get('target', False)
                # 转换物体位置到像素坐标
                pos = query_engine.graph.nodes[obj_id].position
                
                x, y = self.to_row_col(pos)
                if target:
                    ax.text(
                        y, x, 
                        f"{obj_id}",
                        fontsize=10, color='red',
                        ha='center', va='center',
                        bbox=dict(
                            facecolor='white',  # 洋红背景
                            alpha=0.5,
                            boxstyle='round'
                        )
                    )
                else:
                    # 添加标注
                    ax.text(
                        y, x, 
                        f"{obj_id}",
                        fontsize=10, color='black',
                        ha='center', va='center',
                        bbox=dict(
                            facecolor='white',  # 青色背景
                            alpha=0.5,
                            boxstyle='round'
                        )
                    )
    def _create_legend_elements(self) -> list:
        """创建图例元素"""
        elements = []
        
        # 地标图例（灰度轮廓）
        for name, color in self.landmark_map.gray_colors.items():
            # 确保颜色值在0-1范围内
            normalized_color = np.clip(np.array(color[::-1])/255.0, 0, 1)
            elements.append(
                Patch(facecolor=normalized_color,
                    edgecolor='gray',
                    label=name,
                    linestyle='-',
                    linewidth=2)
            )
        
        # 目标/环境图例
        combined_colors = {**self.target_map.semantic_colors, 
                        **self.surroundings_map.semantic_colors}
        for phrase, color in combined_colors.items():
            # 确保颜色值在0-1范围内
            normalized_color = np.clip(np.array(color[:3])/255.0, 0, 1)
            elements.append(
                Patch(facecolor=normalized_color,
                    label=phrase.capitalize())
            )
        
        return elements

    def _render_output(self, fig, map_type) -> Image:
        """渲染输出图像,仅在topdown模式下保存"""
        plot_img = _fig_to_pil(fig)
        sanitized_map_type = map_type.replace(" ", "_").replace("/", "_")
        plt.savefig(self.save_path+f'/{sanitized_map_type}_{self.id}_test_00{self.step}.png')
        return encode_image_from_pil(plot_img)
    
    def plot(
        self,
        map_type,
        query_engine=None,
        current_pos=None
    ):
        # 颜色配置（RGBA格式）
        layer_colors = {
            'current view area': (0.5, 0, 0.5, 0.3),   # 紫色
        }
        
        # 合并地图数据，仅landmark
        map_layers = np.concatenate([
            np.expand_dims(self.to_array()[1],0),
        ])

        # 创建画布
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111)
        # STMR类型特殊处理：添加网格
        if map_type == 'STMR':
            # 绘制网格线
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.6, alpha=0.3)
            ax.set_xticks(np.arange(0, self.shape[1], self.grid_size_pixels))
            ax.set_yticks(np.arange(0, self.shape[0], self.grid_size_pixels))

        # 绘制当前视野区域
        if map_type != 'landmark':
            for idx, (layer_name, rgba) in enumerate(layer_colors.items()):
                self._draw_layer(ax, map_layers[idx], rgba)
        
        # 绘制landmark和semantic图层
        self._merge_color_layers(ax, map_type)
        # 添加landmark标注
        if map_type != 'w/o annotation' and map_type != 'STMR':
            self._annotate_landmarks(ax)
            
        # === 新增轨迹绘制逻辑 ===
        # 绘制方向箭头: topV模式需要
        if map_type != 'w/o annotation':
            arrow_length = 15  # 箭头长度（像素）
            if len(self.trajectory) == 1:
                pose = self.trajectory[0]
                col, row = self.to_row_col(pose.xy)[::-1]
                dx = arrow_length * np.cos(pose.yaw)
                dy = - arrow_length * np.sin(pose.yaw)
                ax.arrow(
                        x=col, y=row,
                        dx=dy, dy=dx,  # 注意坐标轴方向转换
                        width=3, 
                        head_width=8,
                        head_length=10,
                        fc='red',  # 箭头填充色
                        ec='yellow')
            elif len(self.trajectory) >1:
                # 转换轨迹坐标
                trajectory_points = [
                    self.to_row_col(pose.xy)[::-1]  # 转换为(col, row)格式
                    for pose in self.trajectory
                ]
                
                # 绘制轨迹线
                x_coords, y_coords = zip(*trajectory_points)
                if map_type != 'TopV':
                    ax.plot(x_coords, y_coords, color = 'black', linestyle='--', linewidth=1.5, alpha=0.3)

                prev_pose = trajectory_points[-2]
                pose = trajectory_points[-1]
                delta_x = pose[0] - prev_pose[0]
                delta_y = pose[1] - prev_pose[1]
                theta = np.arctan2(delta_y, delta_x)  # 箭头角度是上一位置到当前位置的角度
                col, row = pose
                
                # 计算箭头方向
                dx = arrow_length * np.cos(theta)
                dy = arrow_length * np.sin(theta)
                
                ax.arrow(
                    x=col, y=row,
                    dx=dx, dy=dy,  # 注意坐标轴方向转换
                    width=3, 
                    head_width=8,
                    head_length=10,
                    fc='red',  # 箭头填充色
                    ec='yellow')
            
        # 创建图例，topV模式需要
        if map_type != 'landmark' and map_type != 'STMR':
            self._annotate_objects(ax, query_engine=query_engine, current_pos=current_pos)
            legend_elements = self._create_legend_elements()
            ax.legend(handles=legend_elements, 
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=3)

        # 生成输出图像
        plot_img = self._render_output(fig, map_type)
        plt.close('all')
        
        return plot_img
    def get_semantic_map(self):
        """生成融合target和surroundings的语义地图"""
        # 初始化空白画布（RGB格式）
        sem_map = np.zeros((*self.shape, 3), dtype=np.uint8)
        
        # 叠加surroundings层（洋红色）
        if self.surroundings_map.color_array is not None:
            surroundings_layer = self.surroundings_map.color_array[..., :3].copy().astype(np.uint8)
            sem_map = np.maximum(sem_map, surroundings_layer)  # 直接取最大值，避免颜色重叠
        
        # 叠加target层（青色）
        if self.target_map.color_array is not None:
            target_layer = self.target_map.color_array[..., :3].copy().astype(np.uint8)
            sem_map = np.maximum(sem_map, target_layer)  # 直接取最大值，避免颜色重叠
        
        return sem_map

    def _blend_layers(self, base, layer):
        """图层混合算法（Alpha混合）"""
        alpha = layer[..., 3:]
        blended = base * (1 - alpha) + layer * alpha
        blended[..., 3] = np.minimum(base[..., 3] + layer[..., 3], 1.0)
        return blended
    def vanilla_plot(
        self,
        start_point: Point2D,
        true_goal: Point2D,
        show: bool =False,
    ):
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from PIL import Image
        import numpy as np


        # 定义颜色和透明度 (RGBA 格式: R, G, B, Alpha)
        colors = {
            'current view area': (0, 0, 1, 0.3),  # 蓝色，透明度0.3
            'explored area': (0, 1, 0, 0.3),     # 绿色，透明度0.3
            'landmarks': (1, 0, 0, 0.3),         # 红色，透明度0.3
            'target': (0, 1, 1, 0.5),            # 青色，透明度0.3
            'surroundings': (1, 0, 1, 0.3),      # 洋红色，透明度0.3
            'start point': (1, 1, 0, 1.0),    # 黄色，透明度0.6
            'true goal': (1, 0.5, 0, 1.0)        # 橙色，透明度0.6
        }

        start_point_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(start_point)[::-1],
            radius=4, color=1, thickness=-1
        )
        

        refer_point_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(true_goal)[::-1],
            radius=6, color=1, thickness=1, lineType=cv2.LINE_AA
        )

        # 获取地图数据 (shape 为 [7, 240, 240])
        maps = np.concatenate([self.to_array(), np.stack([start_point_map, refer_point_map])])

        # 创建绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        # fig.suptitle(f"{self.name}: {goal_description}")

        # 绘制每层，并叠加透明度和颜色
        for i, (_, rgba) in enumerate(colors.items()):
            # 为每个数组生成对应的颜色映射
            layer = maps[i]
            color_map = np.zeros((*layer.shape, 4), dtype=np.float32)  # RGBA 图像
            color_map[..., 0] = rgba[0]  # R
            color_map[..., 1] = rgba[1]  # G
            color_map[..., 2] = rgba[2]  # B
            color_map[..., 3] = layer * rgba[3]  # Alpha 透明度与数值强度相关

            ax.imshow(color_map)

        # # ---添加网格线---
        height, width = self.shape
        # # 设置x,y 的刻度位置为网格线间隔
        # x_ticks = np.arange(0, width, grid_size)
        # y_ticks = np.arange(0, height, grid_size)
        # ax.set_xticks(x_ticks, minor=True)
        # ax.set_yticks(y_ticks, minor=True)
        # 绘制网格线
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.6, alpha=0.3)
        ax.set_xticks(np.arange(0, width, self.grid_size_pixels))
        ax.set_yticks(np.arange(0, height, self.grid_size_pixels))
        # ax.set_xticklabels([f'{x:.1f}' for x in ax.get_xticks()])
        # ax.set_yticklabels([f'{y:.1f}' for y in ax.get_yticks()])
        grid_size_pixels = self.grid_size_pixels
        for i in range(0, self.shape[1], grid_size_pixels):
            for j in range(0, self.shape[0], grid_size_pixels):
                # 计算当前网格的字母和数字编号
                col = chr(ord('A') + i // grid_size_pixels)
                row = (j // grid_size_pixels) + 1
                grid_id = f"{col}{row}"
                
                # 在网格中心位置添加文本
                ax.text(
                    x=i + grid_size_pixels/2, 
                    y=j + grid_size_pixels/2, 
                    s=grid_id,
                    color='gray',  # 使用浅色避免喧宾夺主
                    fontsize=8,    # 小字号
                    ha='center', 
                    va='center',
                    alpha=0.9      # 半透明
                )
        
        for landmark in self.landmark_map.landmarks:
            x, y = self.to_row_col(landmark.position.xy)
            ax.text(y, x, landmark.name, fontsize=10, color='black', ha='center', va='center', bbox=dict(facecolor='gray', alpha=0.5, pad=1, boxstyle='round'))

        # 添加图例
        legend_elements = [
            Patch(facecolor=rgba[:3], edgecolor='w', label=title, alpha=rgba[3]) 
            for title, rgba in colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.tight_layout()

        if show:
            plt.show()

        # 将绘制的画布转换为 PIL 图像
        plot_img = _fig_to_pil(fig)
        plot_img = plot_img.resize((224,224))
        plt.savefig(f'results/vanilla/landmap_{self.id}_groundtruth_00{self.step}.png')

        rgb = Image.open(f'results/vanilla/landmap_{self.id}_test_00{self.step}.png')
        # 示例用法
        base64_string = encode_image_from_pil(rgb)
        plt.close(fig)

        return base64_string

    def eva_plot(
        self,
        goal_description: str,
        start_point: Point2D,
        true_goal: Point2D,
        show: bool =False,
    ):
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from PIL import Image
        import numpy as np


        # 定义颜色和透明度 (RGBA 格式: R, G, B, Alpha)
        colors = {
            'current view area': (0, 0, 1, 0.3),  # 蓝色，透明度0.3
            'explored area': (0, 1, 0, 0.3),     # 绿色，透明度0.3
            'landmarks': (1, 0, 0, 0.3),         # 红色，透明度0.3
            'target': (0, 1, 1, 0.5),            # 青色，透明度0.3
            'surroundings': (1, 0, 1, 0.3),      # 洋红色，透明度0.3
            'start point': (1, 1, 0, 1.0),    # 黄色，透明度0.6
            'true goal': (1, 0.5, 0, 1.0)        # 橙色，透明度0.6
        }

        start_point_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(start_point)[::-1],
            radius=4, color=1, thickness=-1
        )
        

        refer_point_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(true_goal)[::-1],
            radius=6, color=1, thickness=1, lineType=cv2.LINE_AA
        )

        # 获取地图数据 (shape 为 [7, 240, 240])
        maps = np.concatenate([self.to_array(), np.stack([start_point_map, refer_point_map])])

        # 创建绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        # fig.suptitle(f"{self.name}: {goal_description}")

        # 绘制每层，并叠加透明度和颜色
        for i, (_, rgba) in enumerate(colors.items()):
            # 为每个数组生成对应的颜色映射
            layer = maps[i]
            color_map = np.zeros((*layer.shape, 4), dtype=np.float32)  # RGBA 图像
            color_map[..., 0] = rgba[0]  # R
            color_map[..., 1] = rgba[1]  # G
            color_map[..., 2] = rgba[2]  # B
            color_map[..., 3] = layer * rgba[3]  # Alpha 透明度与数值强度相关

            ax.imshow(color_map)

        # # ---添加网格线---
        height, width = self.shape
        # # 设置x,y 的刻度位置为网格线间隔
        # x_ticks = np.arange(0, width, grid_size)
        # y_ticks = np.arange(0, height, grid_size)
        # ax.set_xticks(x_ticks, minor=True)
        # ax.set_yticks(y_ticks, minor=True)
        # 绘制网格线
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.6, alpha=0.3)
        ax.set_xticks(np.arange(0, width, self.grid_size_pixels))
        ax.set_yticks(np.arange(0, height, self.grid_size_pixels))
        # ax.set_xticklabels([f'{x:.1f}' for x in ax.get_xticks()])
        # ax.set_yticklabels([f'{y:.1f}' for y in ax.get_yticks()])
        grid_size_pixels = self.grid_size_pixels
        for i in range(0, self.shape[1], grid_size_pixels):
            for j in range(0, self.shape[0], grid_size_pixels):
                # 计算当前网格的字母和数字编号
                col = chr(ord('A') + i // grid_size_pixels)
                row = (j // grid_size_pixels) + 1
                grid_id = f"{col}{row}"
                
                # 在网格中心位置添加文本
                ax.text(
                    x=i + grid_size_pixels/2, 
                    y=j + grid_size_pixels/2, 
                    s=grid_id,
                    color='gray',  # 使用浅色避免喧宾夺主
                    fontsize=8,    # 小字号
                    ha='center', 
                    va='center',
                    alpha=0.9      # 半透明
                )
        
        for landmark in self.landmark_map.landmarks:
            x, y = self.to_row_col(landmark.position.xy)
            ax.text(y, x, landmark.name, fontsize=10, color='black', ha='center', va='center', bbox=dict(facecolor='gray', alpha=0.5, pad=1, boxstyle='round'))

        # 添加图例
        legend_elements = [
            Patch(facecolor=rgba[:3], edgecolor='w', label=title, alpha=rgba[3]) 
            for title, rgba in colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.tight_layout()

        if show:
            plt.show()

        # 将绘制的画布转换为 PIL 图像
        plot_img = _fig_to_pil(fig)
        plot_img = plot_img.resize((224,224))
        plt.savefig(f'results/grid/landmap_{self.id}_groundtruth_00{self.step}.png')

        rgb = Image.open(f'results/grid/landmap_{self.id}_test_00{self.step}.png')
        # 示例用法
        base64_string = encode_image_from_pil(rgb)
        plt.close(fig)

        return base64_string
    
    def extract_json_from_msg(self, msg):
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
    
    def grid_id_to_world_xy(self, grid_id: str) -> Point2D:
        """将网格ID（如C3）转换为对应网格中心的世界坐标
        
        参数：
            grid_id: 网格标识符，格式为字母+数字（如C3）
        
        返回：
            Point2D: 网格中心点的世界坐标
        """
        # 解析网格ID，保留第一个有效的ID
        grid_id = grid_id.split(',')[0].strip()
        
        # 提取列字符和行数字
        col_char = re.match(r'[A-Z]', grid_id).group()
        row_num = int(re.search(r'\d+', grid_id).group())
        
        # 转换为网格索引
        col_idx = ord(col_char) - ord('A')
        row_idx = row_num - 1
        
        # 计算网格左上角坐标（像素）
        grid_size = self.grid_size_pixels
        x_pixel = col_idx * grid_size
        y_pixel = row_idx * grid_size
        
        # 计算网格中心坐标（像素）
        center_col = x_pixel + grid_size // 2  # 列方向（x轴）
        center_row = y_pixel + grid_size // 2  # 行方向（y轴）
        
        # 转换为世界坐标
        return self.to_world_xy(center_row, center_col)