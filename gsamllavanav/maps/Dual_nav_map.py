# high-level map give VLM insight of the scene，提供例如：
# 1. 描述场景布局（街道的布局关系）
# 2. 描述街道的形状
# 3. 描述无人机所在位置，因此该采取的行动
# 4. 描述目标和街道的空间关系（置信度）
# 5. 描述该采取的动作（利用现有信息定位，探索）

# low-level map give VLM the obj loc of the scene,提供例如：
# 6. 物体，以及和目标描述的一致性
# 7. 描述导航路径
# 8. 描述导航路径上的物体关系
# 9. 从high-level map的导入路径在街道的关系
# 10. 描述导航路径上的动作（移动到哪个地点，利用控制器转化为前后左右）
# 11. 描述导航路径上的目标
# 12. 描述导航路径上的房间

from typing import Optional
import os
import re
import json
import numpy as np
from openai import OpenAI


from gsamllavanav.observation import cropclient
from gsamllavanav.defaultpaths import GSAM_MAPS_DIR
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.dataset.episode import Episode

from .map import Map
from .tracking_map import TrackingMap
from .landmark_map import LandmarkMap
from .gsam_map import GSamMap, GSamParams


class GeoNavMap(Map):
    def __init__(
        self,
        id: tuple,
        map_name: str,
        map_shape: tuple[int, int],
        map_pixels_per_meter: float,
        landmark_names: list[str],
        target_name: str, surroundings_names: list[str],
        gsam_params: GSamParams,
        grid_size_meters: float = 20.0,
    ):
        super().__init__(map_name, map_shape, map_pixels_per_meter)
        self.step = 0
        self.id = id
        self.grid_size_meters = grid_size_meters
        self.grid_size_pixels = int(grid_size_meters * map_pixels_per_meter)
        # 预计算网格边界
        self.grid_x_min = - map_shape[1] // 2 / map_pixels_per_meter
        self.grid_y_min = -map_shape[0] // 2 / map_pixels_per_meter

        # 地图历史
        self.history_actions = ['None']
        self.history_AOI = ['None']

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
        nav_map = GeoNavMap(map_name, map_shape, map_pixels_per_meter, landmark_names, target_name, object_names)
        nav_map.tracking_map.current_view_area = map_data[0].astype(np.uint8)
        nav_map.tracking_map.explored_area = map_data[1].astype(np.uint8)
        nav_map.landmark_map.landmark_map = map_data[2].astype(np.uint8)
        nav_map.target_map.gsam_map = map_data[3]
        nav_map.surroundings_map.gsam_map = map_data[4]

        return nav_map
    
    # def plot(
    #     self,
    #     goal_description: str,
    #     predicted_goal: Point2D,
    #     true_goal: Point2D,
    #     show=False,
    # ):
    #     import cv2
    #     self.step += 1
    #     predicted_goal_map = cv2.circle(
    #         img=np.zeros(self.shape, dtype=np.float32),
    #         center=self.to_row_col(predicted_goal)[::-1],
    #         radius=4, color=1, thickness=-1
    #     )

    #     true_goal_map = cv2.circle(
    #         img=np.zeros(self.shape, dtype=np.float32),
    #         center=self.to_row_col(true_goal)[::-1],
    #         radius=4, color=1, thickness=-1
    #     )

    #     titles = ['current view area', 'explored area', 'landmarks', 'target', 'surroundings', 'predicted goal', 'true goal']
    #     maps = np.concatenate([self.to_array(), np.stack([predicted_goal_map, true_goal_map])])

    #     import matplotlib.pyplot as plt
    #     from PIL import Image

    #     fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(35, 5), subplot_kw={'xticks': [], 'yticks': []})
    #     fig.suptitle(f"{self.name}: {goal_description}")

    #     for ax, title, m in zip(axs, titles, maps):
    #         ax.imshow(m, cmap='viridis')
    #         ax.set_title(title)
        
    #     plt.tight_layout()
    #     fig.canvas.draw()

    #     if show:
    #         plt.show()
        
    #     plot_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    #     plt.savefig(f'results/test_landmap_00{self.step}.png')
    #     plt.close(fig)
        
    #     return plot_img

    def plot(
        self,
        type: str,
        start_point: Point2D,
        current_pose: Pose4D,
        with_grid: bool = False
    ):
        import cv2
        import matplotlib.pyplot as plt
        from PIL import Image
        import numpy as np

        self.step += 1

        # 定义颜色和透明度 (RGBA 格式: R, G, B, Alpha)
        colors = {
            'explored area': (0, 0, 1, 0.2),  # 蓝色，透明度0.3
            'current view area': (0, 1, 0, 0.2),     # 绿色，透明度0.3
            'landmarks': (1, 0, 0, 0.3),         # 红色，透明度0.3
            'target': (0, 1, 1, 0.5),            # 青色，透明度0.3
            'surroundings': (1, 0, 1, 0.3),      # 洋红色，透明度0.3
            'start point': (1, 1, 0, 1.0),    # 黄色，透明度0.6
            'current pose': (1, 0.5, 0, 1.0)        # 橙色，透明度0.6
        }

        start_point_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(start_point)[::-1],
            radius=4, color=1, thickness=-1
        )
        
        current_pose_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(current_pose.xy)[::-1],
            radius=4, color=1, thickness=-1
        )
        # 获取地图数据 (shape 为 [7, 240, 240])
        maps = np.concatenate([self.to_array(), np.stack([start_point_map, current_pose_map])])

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
        if with_grid==True:
            # 绘制网格线
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.6, alpha=0.3)
            ax.set_xticks(np.arange(0, width, self.grid_size_pixels))
            ax.set_yticks(np.arange(0, height, self.grid_size_pixels))
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
                        color='gray',  # 使用浅色
                        fontsize=8,    # 小字号
                        ha='center', 
                        va='center',
                        alpha=0.9      # 半透明
                    )
        
        for landmark in self.landmark_map.landmarks:
            x, y = self.to_row_col(landmark.position.xy)
            ax.text(y, x, landmark.name, fontsize=11, color='black', ha='center', va='center', bbox=dict(facecolor='gray', alpha=0.5, pad=1, boxstyle='round'))
        dx, dy = 5 * np.cos(current_pose.yaw), 10 * np.sin(current_pose.yaw) # 无人机方向箭头
        ax.arrow(self.to_row_col(current_pose.xy)[1], self.to_row_col(current_pose.xy)[0], dy, dx, width=0.8, head_width=2, head_length=1, fc='orange', ec='orange')

        plt.tight_layout()

        # 将绘制的画布转换为 PIL 图像
        plot_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plot_img = plot_img.resize((224,224))
        plt.savefig(f'results/finetuned_rationale/landmap_{self.id}_test_00{self.step}.png')
        from ggb.QwenAPI import encode_image_from_pil
        rgb = Image.open(f'results/finetuned_rationale/landmap_{self.id}_test_00{self.step}.png')
        # 示例用法
        base64_string = encode_image_from_pil(rgb)
        plt.close(fig)

        return base64_string

    def integrate_prior_knowledge(self, target_description: str, task_prior_knowledge: dict, rationale: str):
        #'vanilla', 'spatial_rationale', 'ours'
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
        - Explored Area: Regions already scanned (blue overlay) 
        - Current View: Drone's visible area (green overlay)  
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