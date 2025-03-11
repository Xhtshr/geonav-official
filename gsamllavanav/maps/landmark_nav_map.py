from typing import Optional
import os
import re
import cv2
import json
from PIL import Image
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


from ggb.QwenAPI import encode_image_from_pil
from gsamllavanav.observation import cropclient
from gsamllavanav.defaultpaths import GSAM_MAPS_DIR
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.dataset.episode import Episode

from .map import Map
from .tracking_map import TrackingMap
from .landmark_map import LandmarkMap
from .gsam_map import GSamMap, GSamParams


class LandmarkNavMap(Map):
    def __init__(
        self,
        map_name: str,
        map_shape: tuple[int, int],
        map_pixels_per_meter: float,
        landmark_names: list[str],
        target_name: str, surroundings_names: list[str],
        gsam_params: GSamParams,
        id: tuple = {},
        grid_size_meters: float = 20.0,
        save_path: str = 'results/geonav',
    ):
        super().__init__(map_name, map_shape, map_pixels_per_meter)
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
        self.target_map = GSamMap(map_name, map_shape, map_pixels_per_meter, [target_name], gsam_params)
        self.surroundings_map = GSamMap(map_name, map_shape, map_pixels_per_meter, surroundings_names, gsam_params)
    
    def update_observations(
        self,
        camera_pose: Pose4D,
        rgb: np.ndarray,
        depth_perspective: Optional[np.ndarray] = None,
        use_gsam_map_cache=True,
    ):
        self.trajectory.append(camera_pose)
        self.tracking_map.mark_current_view_area(camera_pose)
        if use_gsam_map_cache:
            self.target_map.update_from_map_cache(camera_pose)
            self.surroundings_map.update_from_map_cache(camera_pose)
        else:
            self.target_map.update_observation(camera_pose, rgb[..., ::-1], depth_perspective)
            self.surroundings_map.update_observation(camera_pose, rgb[..., ::-1], depth_perspective)

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
        if map_type == 'w/o semantic map':
            layers = [
            (self.landmark_map.color_array, 1.0)
        ]
        elif map_type == 'w/o landmark map':
            layers = [
            (self.surroundings_map.color_array, 0.75),  # 环境层中等透明度
            (self.target_map.color_array, 0.75)         # 目标层中等透明度
        ]
        # 调整叠加顺序：底层元素先绘制，关键元素最后叠加
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
    def _create_legend_elements(self) -> list:
        """创建图例元素"""
        elements = []
        
        # 地标图例（灰度轮廓）
        for name, color in self.landmark_map.gray_colors.items():
            elements.append(
                Patch(facecolor=np.array(color[::-1])/255,
                    edgecolor='gray',
                    label=name,
                    linestyle='-',
                    linewidth=2)
            )
        
        # 目标/环境图例
        for phrase, color in {**self.target_map.semantic_colors, 
                            **self.surroundings_map.semantic_colors}.items():
            elements.append(
                Patch(facecolor=np.array(color[:3])/255,
                    label=phrase.capitalize())
            )
        
        return elements

    def _render_output(self, fig, map_type) -> Image:
        """渲染输出图像"""
        fig.canvas.draw()
        plot_img = Image.frombytes('RGB', 
                                fig.canvas.get_width_height(),
                                fig.canvas.tostring_rgb())
        sanitized_map_type = map_type.replace(" ", "_").replace("/", "_")
        plt.savefig(self.save_path+f'/{sanitized_map_type}_{self.id}_test_00{self.step}.png')
        return encode_image_from_pil(plot_img)
    def plot(
        self,
        map_type
    ):
        
        import numpy as np
        self.step += 1
        # 颜色配置（RGBA格式）
        layer_colors = {
            'current view area': (0.5, 0, 0.5, 0.3),   # 紫色
        }
        
        # 合并地图数据
        map_layers = np.concatenate([
            np.expand_dims(self.to_array()[1],0),
        ])

        # 创建画布
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111)

        # 绘制基础图层

        self._merge_color_layers(ax, map_type)
        # 添加地标标注
        if map_type != 'w/o annotation':
            self._annotate_landmarks(ax)
        
        # 合并颜色层（含灰度地标）
        if map_type != 'w/o semantic map':
            for idx, (layer_name, rgba) in enumerate(layer_colors.items()):
                self._draw_layer(ax, map_layers[idx], rgba)
        
        # === 新增轨迹绘制逻辑 ===
        # 绘制方向箭头
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
                        fc='lime',  # 箭头填充色
                        ec='darkgreen',  # 箭头边缘色
                        alpha=0.8)  # 半透明箭头
            elif len(self.trajectory) >1:
                # 转换轨迹坐标
                trajectory_points = [
                    self.to_row_col(pose.xy)[::-1]  # 转换为(col, row)格式
                    for pose in self.trajectory
                ]
                
                # 绘制轨迹线
                x_coords, y_coords = zip(*trajectory_points)
                ax.plot(x_coords, y_coords, color = 'black', linestyle='--', linewidth=2.5, alpha=0.7)  # 白色虚线轨迹

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
                    fc='lime',  # 箭头填充色
                    ec='darkgreen',  # 箭头边缘色
                    alpha=0.4)  # 半透明箭头
            
        # 创建图例
        if map_type != 'w/o semantic map':
            legend_elements = self._create_legend_elements()
            ax.legend(handles=legend_elements, 
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=3)

        # 生成输出图像
        plot_img = self._render_output(fig, map_type)
        plt.close('all')
        
        return plot_img
    
    def vanilla_plot(
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
        plot_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
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
        plot_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plot_img = plot_img.resize((224,224))
        plt.savefig(f'results/grid/landmap_{self.id}_groundtruth_00{self.step}.png')

        rgb = Image.open(f'results/grid/landmap_{self.id}_test_00{self.step}.png')
        # 示例用法
        base64_string = encode_image_from_pil(rgb)
        plt.close(fig)

        return base64_string
    def pixel_knowledge(self, target_description: str, task_prior_knowledge: dict, rationale: str):
        #'vanilla', 'spatial_rationale', 'ours'
        # 
        if rationale == 'vanilla':
            Easy_prompt = f"""Role: You are a High-Level Decision-Making Agent for drone navigation.Your task is to analyze multi-modal inputs (visual map + textual knowledge) to determine whether to explore unknown areas or focus on target localization. Task: {target_description}"""+ """
    ---
    **Visual Context (Map)**  
    - Grid System: Columns (A-Z), Rows (1-N). Current drone position marked with orientation arrow.  
    - Map Layers:  
        - Landmarks: Predefined static objects with labels  
        - Explored Area: Regions already scanned (green overlay)  
        - Current View: Drone's visible area (blue overlay)  
        - Suspected Targets: Detected objects matching target description (cyan regions)  
        - Surrounding Objects: Other detected objects (magenta regions)  

    **Required Output Format**  
    ```json
    {
    "decision": "Explore|Locate",
    "selected_aoi": "GridID (e.g., C3)",
    "navigation_instruction": "Move [direction] for [X] grids"
    }"""
            return Easy_prompt
        elif rationale == 'space_rationale':
            N = 9
            prompt = f"""Role: You are a High-Level Decision-Making Agent for drone navigation. 
    Your task is to analyze multi-modal inputs (visual map + textual knowledge) to determine whether to explore unknown areas or focus on target localization.

    --- 

    **Contextual Inputs**  
    1. **Mission Objective**  
    - Target: {task_prior_knowledge['Target']}  
    - Key Spatial Constraints:  
        - Landmark Relationships: "{task_prior_knowledge['Relationships with Landmarks']}"  
        - Surroundings: "{task_prior_knowledge['Surrounding']}"  
        - Surroundings-Target Spatial Context: "{task_prior_knowledge['Spatial_Relationships with objects']}"  

    2. **Operational History**  
    - Recent Actions: {', '.join(self.history_actions)}  
    - Previously Selected AOIs: {', '.join(self.history_AOI)}  

    3. **Visual Context (Map)**  
    - Grid System: Columns (A-Z), Rows (1-N). Current drone position marked with orientation arrow.  
    - Map Layers:  
        - Landmarks: Predefined static objects with labels  
        - Explored Area: Regions already scanned (green overlay)  
        - Current View: Drone's visible area (blue overlay)  
        - Suspected Targets: Detected objects matching target description (cyan regions)  
        - Surrounding Objects: Other detected objects (magenta regions)  

    ---

    **Decision-Making Protocol**  
    Analyze the following aspects in sequence:  

    1. **Spatial Reasoning**  
    - Compare target's expected location (from Prior Knowledge) against suspected target positions on the map.  
    - Calculate grid-based distance between drone's current position and potential target AOIs.  

    2. **Exploration-Localization Tradeoff**  
    - IF (Uncertainty > 60%) OR (No high-confidence target detected in viewed areas):  
        → Prioritize exploring unknown grids adjacent to landmark-related AOIs.  
    - ELSE IF (Target-like objects found in grids matching prior knowledge):  
        → Focus on localizing within {N} grids around suspected AOI.  

    3. **AOI Selection Criteria**  
    - Prefer grids that:  
        a) Align with landmark relationships (e.g., "north of Church")  
        b) Maximize coverage of unexplored areas  
        c) Minimize backtracking (avoid revisiting >2 times)  
    """ + """
    ---

    **Required Output Format**  
    ```json
    {
    "decision": "Explore|Locate",
    "selected_aoi": "GridID (e.g., C3)",
    "rationale": {
        "spatial_consistency": "Score 1-5 (How well AOI matches prior knowledge)",
        "exploration_priority": "Score 1-5 (Urgency to scan new areas)",
        "confidence": "0-100% likelihood of correct decision"
    },
    "navigation_instruction": "Move [direction] for [X] grids towards [landmark]"
    }"""
            return prompt
        elif rationale == 'ours':
            N = 9
            prompt = f"""Role: You are a High-Level Decision-Making Agent for drone navigation. 
    Your task is to analyze multi-modal inputs (visual map + textual knowledge) to determine whether to explore unknown areas or focus on target localization.

    --- 

    **Contextual Inputs**  
    1. **Mission**  
    - Task: {target_description}

    2. **Operational History**  
    - Recent Actions: {', '.join(self.history_actions)}  
    - Previously Selected AOIs: {', '.join(self.history_AOI)}  

    3. **Visual Context (Map)**  
    - Grid System: Columns (A-Z), Rows (1-N). Current drone position marked with orientation arrow.  
    - Map Layers:  
        - Current Pose: Drone's current position and orientation (orange arrow)  
        - Start Pose: Drone's start position location (yellow arrow)
        - Landmarks: Predefined static objects with labels  
        - Explored Area: Regions already scanned (green overlay)  
        - Current View: Drone's visible area (blue overlay)  
        - Suspected Targets: Detected objects matching target description (cyan regions)  
        - Surrounding Objects: Other detected objects (magenta regions)  

    ---
    **Decision-Making Protocol**
    Analyze the following aspects in sequence:

    1. **Spatial Reasoning**  
    - Please use your own direction of the arrow as a reference and convert the direction descriptions ("turn left" or "left side") into the left and right areas corresponding grid coordinates (such as "the street on the right side of the arrow").
    - Compare target's expected location (from Prior Knowledge description) against suspected target positions on the map, for example, "off the Church" means target is near the grid of "the Church".

    2. **Exploration-Localization Tradeoff**  
    - IF (Uncertainty > 60%) OR (No high-confidence target detected in viewed areas):  
        → Prioritize exploring unknown grids adjacent to landmark-related AOIs.  
    - ELSE IF (Target-like objects found in grids matching prior knowledge):  
        → Focus on localizing within {N} grids around suspected AOI.  

    3. **AOI Selection Criteria**  
    - Prefer grids that:  
        a) Align with landmark relationships (e.g., "north of Church")  
        b) Maximize coverage of unexplored areas  
        c) Minimize backtracking (avoid revisiting >2 times)  
    
    4. **Notice
    - Do not confuse the starting point with the current position of the drone.
    - Do not output the starting point Grid ID, which may be confused with the Area of Interest's Grid ID.
    
    """+ """
    ---

    **Required Output Format**  
    Thought: [Put your thinking process there. You should think about the location of the target object or the region where it is located. This can be
achieved by reasonably imagining the unseen areas
based on the room layout]
    Answer:
    ```json
    {
    "decision": "Explore|Locate",
    "selected_aoi": "GridID (e.g., C3)", # only output one grid ID here
    "rationale": {
        "spatial_consistency": "Score 1-5 (How well AOI matches prior knowledge)",
        "exploration_priority": "Score 1-5 (Urgency to scan new areas)",
        "confidence": "0-100% likelihood of correct decision"
    },
    "navigation_instruction": "Move [direction] for [X] grids towards [landmark]"
    }"""
            return prompt

    def generate_aoi(self, prompt, image_64):
        
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        response = client.chat.completions.create(
                model="qwen-vl-plus-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                # 需要注意，传入BASE64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                                # PNG图像：  f"data:image/png;base64,{base64_image}"
                                # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                                # WEBP图像： f"data:image/webp;base64,{base64_image}"
                                "image_url": {"url": f"data:image/png;base64,{image_64}"}, 
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        answer = self.extract_json_from_msg(response.choices[0].message.content)

        # 假如文件有内容,接着将response中的内容保存到f'results/landmap_{id}_00{self.step}.txt'
        with open(f'results/finetuned_rationale/landmap_{self.id}_00{self.step}.txt', 'w') as f:
            f.write(prompt + "\n\n")
            json.dump(answer, f, ensure_ascii=False, indent=4)
        self.history_AOI.append(answer["selected_aoi"])
        self.history_actions.append(answer["decision"])
        return answer
    
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