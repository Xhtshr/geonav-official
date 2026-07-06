from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
import supervision as sv
import torch
from groundingdino.util.inference import Model
from scipy.spatial.transform import Rotation as R

from gsamllavanav.defaultpaths import GDINO_CHECKPOINT_PATH, GDINO_CONFIG_PATH, SAM_CHECKPOINT_PATH, MOBILE_SAM_CHECKPOINT_PATH, GSAM_MAPS_DIR
from gsamllavanav.space import Point2D, Pose4D, view_area_corners, xyxy_to_global_bbox

from .map import Map

GDINO_BOX_TRESHOLD = 0.35
GDINO_TEXT_TRESHOLD = 0.25
GDINO_MAX_BOX_SIZE = 50.
GDINO_MAX_BOX_AREA = 3000.
MAX_DEPTH_METERS = 200.  # depends on airsim simulator settings (airsim default is 100.)

SegmentationModel = Literal['SAM', 'MobileSAM']


@dataclass
class GSamParams:
    use_segmentation_mask: bool
    use_bbox_confidence: bool
    box_threshold: float = GDINO_BOX_TRESHOLD
    text_threshold: float = GDINO_TEXT_TRESHOLD
    max_box_size: float = GDINO_MAX_BOX_SIZE
    max_box_area: float = GDINO_MAX_BOX_AREA


class GSamMap(Map):
    '''A semantic map constructed from the outputs of GSAM segmentation masks'''

    _grounding_dino_model = None
    _sam_predictor = None
    _map_cache = None

    def __init__(
        self,
        map_name: str,
        map_shape: tuple[int, int],
        map_pixels_per_meter: float,
        captions: list[str],
        gsam_params: GSamParams,
        device='cuda',
        layer: str = 'surroundings',
    ):
        super().__init__(map_name, map_shape, map_pixels_per_meter)
        self.layer = layer
        self.semantic_colors = {}

        self.color_cache = {}  # 用于动态颜色分配
        self.color_array = np.zeros((*map_shape, 4), dtype=np.uint8)  # RGB颜色阵列初始化为黑底透明

        self.captions = captions
        self.gsam_map = np.zeros(map_shape, dtype=np.float32)
        self.gsam_params = gsam_params
        self.obj_list = []
        self.draw_list = []
        if GSamMap._grounding_dino_model is None or GSamMap._sam_predictor is None:
            GSamMap._init_models(device=device)

    #为每个语义类别动态分配颜色，优先用预定义色，超出后用黄金角算法生成新色，保证可区分性
    def _get_dynamic_color(self, phrase):
        # 预定义科研配色（RGB格式，alpha=255）
        if self.layer == 'target':
            PREDEFINED_COLORS = [(255, 0, 0),(0, 100, 0),(210, 105, 30),(32, 32, 32)]
        else:
            PREDEFINED_COLORS = [
                (31, 119, 180),   # 蓝色
                (255, 127, 14),   # 橙色
                (44, 160, 44),    # 绿色
                (214, 39, 40),    # 红色
                (148, 103, 189),  # 紫色
                (140, 86, 75),    # 棕色
                (227, 119, 194),  # 粉色
                (127, 127, 127),  # 灰色
                (188, 189, 34),   # 橄榄绿
                (23, 190, 207)    # 青色
            ]

        if phrase not in self.semantic_colors:
            # 排除已使用的预定义颜色
            used_colors = set(self.semantic_colors.values())
            # 顺序分配预定义颜色
            for color in PREDEFINED_COLORS:
                rgba_color = (*color, 255)
                if rgba_color not in used_colors:
                    self.semantic_colors[phrase] = rgba_color
                    break
            else:  # 颜色用尽时回退到黄金角算法
                # 使用黄金角 (约137.5度) 来生成均匀分布的色相
                golden_ratio = (5 ** 0.5 - 1) / 2
                n = len(self.semantic_colors)
                hue = (n * golden_ratio) % 1.0  # 在 [0,1] 范围内
                
                # 固定饱和度和明度以确保颜色可见性
                saturation = 0.8  # 80% 饱和度
                value = 0.9      # 90% 明度
                
                # 转换 HSV 到 RGB
                hsv_color = np.array([[[
                    hue * 180,  # OpenCV 需要 0-180 的色相值
                    saturation * 255,  # OpenCV 需要 0-255 的饱和度值
                    value * 255  # OpenCV 需要 0-255 的明度值
                ]]], dtype=np.float32)
                # 转换 HSV 到 BGR
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
                # 转换 BGR 到 RGB
                rgb_color = bgr_color[0, 0, ::-1]
                # 将颜色值转换为整数并添加 alpha 通道
                rgba_color = (*[int(c) for c in rgb_color], 255)
                self.semantic_colors[phrase] = rgba_color
            
        return self.semantic_colors[phrase]
    
    #将检测到的bbox（像素坐标）转换为全局物理坐标（Point2D），用于后续目标定位
    def bbox_to_global_pos(self, bboxes):
        poses = []
        for bbox in bboxes:
            bbox_g = xyxy_to_global_bbox(bbox, self.image_bgr.shape[:2], self.pose, self.ground_level)
            center_x = sum(point.x for point in bbox_g) / 4
            center_y = sum(point.y for point in bbox_g) / 4
            # 添加到 poses 列表
            poses.append(Point2D(center_x, center_y))
        return poses
    #根据检测到的bbox、短语和置信度，生成目标列表（obj_list），便于后续处理
    def detect_list(self, bboxes):
        self.obj_list = []
        poses = self.bbox_to_global_pos(bboxes)
        for pos, phrase, confidence in zip(poses, self.phrases, self.detections.confidence):
            if phrase == '' and len(self.captions)==1:
                phrase = self.captions[0]
            self.obj_list.append((pos, phrase, confidence))

    #观测更新函数
    def update_observation(
        self,
        camera_pose: Pose4D,
        image_bgr: np.ndarray,
        depth_perspective: Optional[np.ndarray] = None,
        flip_depth=True,
        max_depth_meters=MAX_DEPTH_METERS,
        strategy=''
    ):
        if depth_perspective is not None and flip_depth:  # 是否翻转深度图（因传感器常输出上下颠倒的图）
            depth_perspective = np.flip(depth_perspective, axis=0)

        self.pose = camera_pose
        self.image_bgr = image_bgr

        if (image_bgr > 0).mean() < 0.15:  # skip empty image
            self.detections = None
            self.phrases = None
            return self
        # 在这里不兼容旧方法了（目标检测）
        self.detections, self.phrases = GSamMap._gdino_predict_bboxes(
            image_bgr, self.captions, image_bgr.shape[0] / (2 * (camera_pose.z - self.ground_level)),
            self.gsam_params.box_threshold, self.gsam_params.text_threshold, self.gsam_params.max_box_size, self.gsam_params.max_box_area
        )
        # store obj nodes
        self.obj_list = []
        if self.detections:
            # 生成语义颜色层
            if self.gsam_params.use_segmentation_mask:
                self.detections.mask, bboxes = GSamMap._sam_segment(image_bgr, self.detections)
                if strategy == 'Search':
                    self.detect_list(bboxes)
            else:
                self.detections.mask = np.stack([
                    cv2.rectangle(np.zeros(image_bgr.shape[:2], dtype=np.uint8), (x1, y1), (x2, y2), True, -1).astype(bool)
                    for x1, y1, x2, y2 in self.detections.xyxy.astype(np.int32)
                ])
            
            if depth_perspective is None:
                #平面投影
                new_gsam_map = self._gsam_map_from_planar_projection(self.detections, camera_pose, image_bgr.shape[:2])
            else:
                #透视投影
                new_gsam_map = self._gsam_map_from_perspective_projection(self.detections, camera_pose, depth_perspective, max_depth_meters)
            
            self.gsam_map = np.maximum(self.gsam_map, new_gsam_map)

        return self
    
    #从缓存中加载预先生成的地图片段，快速更新当前视野区域的gsam_map，提升效率
    def update_from_map_cache(self, camera_pose: Pose4D):

        rows, cols = self.to_rows_cols(view_area_corners(camera_pose, self.ground_level))
        view_corners = np.stack((cols, rows)).T
        view_area_mask = cv2.fillConvexPoly(np.zeros(self.shape, dtype=np.uint8), view_corners, color=1).astype(bool)

        altitude = round(camera_pose.z - self.ground_level)
        # params = (altitude, self.shape[0], round(self.size_meters))
        params = (100, self.shape[0], round(self.size_meters))
        
        if GSamMap._map_cache is None:
            GSamMap._map_cache = dict(np.load(GSAM_MAPS_DIR/f'full_scan_{params}.npz'))

        for caption in self.captions:
            caption = caption.replace('/', ' ')
            map_cache = GSamMap._map_cache[f'{self.name}-{caption}'.lower()][0]
            self.gsam_map[view_area_mask] = np.maximum(self.gsam_map[view_area_mask], map_cache[view_area_mask])
            #采用最大值融合
        return self

    #将gsam_map转为标准数组格式，便于后续处理或可视化
    def to_array(self, dtype=np.float32) -> np.ndarray:
        gsam_map = self.gsam_map if self.gsam_params.use_bbox_confidence else self.gsam_map > 0
        return gsam_map[np.newaxis].astype(dtype)
    
    @property#返回当前观测中置信度最高的目标的全局bbox
    def max_confidence_bbox(self):

        row, col, channel = self.image_bgr.shape

        xyxy = self.detections.xyxy[np.argmax(self.detections.confidence)] if self.detections else (0, 0, col-1, row-1)
        
        return xyxy_to_global_bbox(xyxy, (row, col), self.pose, self.ground_level)

    #可视化检测到的目标框，支持显示标签和置信度
    def plot_bboxes(self, plot_size=(16, 16), show=True):
        if self.detections:
            labels = [f"{phrase} {confidence:0.2f}" for confidence, phrase in zip(self.detections.confidence, self.phrases)]
            annotated_frame = sv.BoxAnnotator().annotate(self.image_rgb.copy(), self.detections, labels)
        else:
            annotated_frame = self.image_rgb

        if show:
            sv.plot_image(annotated_frame, plot_size)

        return annotated_frame
    
    #可视化分割掩码，叠加在原图上
    def plot_segmentation_masks(self, plot_size=(16, 16), show=True):
        if self.detections:
            self.detections.class_id = np.arange(len(self.detections))
            box_annotated_frame = self.plot_bboxes(plot_size, show=False)
            annotated_frame = sv.MaskAnnotator().annotate(box_annotated_frame, self.detections)
        else:
            annotated_frame = self.image_rgb

        if show:
            sv.plot_image(annotated_frame, plot_size)
        
        if self.detections:
            self.detections.class_id = None

        return annotated_frame
    
    #直接可视化gsam_map
    def plot(self, plot_size=(16, 16)):
        gsam_map = self.to_array()
        sv.plot_image(gsam_map, plot_size)
        return gsam_map
    
    @property#将BGR格式的图像转为RGB，便于可视化
    def image_rgb(self):
        return self.image_bgr[..., ::-1]
    
    @classmethod
    @torch.no_grad()
    def _init_models(cls, segmentation_model: SegmentationModel = 'MobileSAM', device='cuda'):
        cls._grounding_dino_model = Model(GDINO_CONFIG_PATH, GDINO_CHECKPOINT_PATH, device)
        if segmentation_model == 'SAM':
            from segment_anything import SamPredictor, sam_model_registry
            cls._sam_predictor = SamPredictor(sam_model_registry["vit_h"](SAM_CHECKPOINT_PATH).to(device=device).eval())
        if segmentation_model == 'MobileSAM':
            from mobile_sam import SamPredictor, sam_model_registry
            cls._sam_predictor = SamPredictor(sam_model_registry["vit_t"](MOBILE_SAM_CHECKPOINT_PATH).to(device=device).eval())
        
        assert not cls._grounding_dino_model.model.training and not cls._sam_predictor.model.training
    
    @classmethod
    @torch.no_grad()
    def _gdino_predict_bboxes(
        cls, image_bgr: np.ndarray, captions: list[str], pixels_per_meter: float,
        box_threshold:float, text_threshold: float,
        max_box_size: float, max_box_area: float,
    ) -> tuple[sv.Detections, list[str]]:
        '''Predicts bounding boxes from `image_bgr` matching `caption`
        
        Refer to https://github.com/IDEA-Research/GroundingDINO for details.
        Bounding boxes exceeding any of `max_box_width`, `max_box_height`, `max_box_area`
        are filtered out of the result.
        '''
        # predict bounding boxes
        detections, phrases = cls._grounding_dino_model.predict_with_caption(
            image_bgr, '.'.join(captions), box_threshold, text_threshold
        )
        #print("Raw phrases:", phrases)
        
        # === 关键修复：用标准化后的 phrase 检查是否在 caption_set 中 ===
        # Step 1: 如果没有检测结果，直接返回空
        if len(phrases) == 0:
            return detections, phrases  # detections 也是空的 sv.Detections
        # Step 2: 标准化输入 captions（转小写、去空格）
        caption_set = set(c.lower().strip() for c in captions)
        # Step 3: 检查是否在 caption_set 中
        valid_phrases_mask = np.array([p.lower().strip() in caption_set for p in phrases])
        #print("Valid mask:", valid_phrases_mask)
        #print("Filtered phrases:", np.array(phrases)[valid_phrases_mask].tolist())
        # Step 4: 过滤 detections 和 phrases
        detections = detections[valid_phrases_mask]
        phrases = np.array(phrases)[valid_phrases_mask].tolist()
        # === 关键修复：用标准化后的 phrase 检查是否在 caption_set 中 ===
        
        # filter out boxes with invalid size, xyxy: left,down,right,up
        # unit is meter
        box_widths = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) / pixels_per_meter
        box_heights = (detections.xyxy[:, 3] - detections.xyxy[:, 1]) / pixels_per_meter
        box_area = box_widths * box_heights
        valid_mask = (box_widths < max_box_size) & (box_heights < max_box_size) & (box_area < max_box_area)
        
        return detections[valid_mask], np.array(phrases)[valid_mask].tolist()

    @classmethod
    @torch.no_grad()
    def _sam_segment(cls, image_bgr: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        使用 SAM 模型生成分割掩码，并计算每个掩码的精确边界框。
        
        返回：
            masks: 分割掩码数组，形状为 (n_boxes, height, width)，dtype=bool。
            refined_bboxes: 精确的 bbox 列表，格式为 [(x1, y1, x2, y2), ...]。
        """
        image_rgb = image_bgr[..., ::-1] # bgr 2 rgb
        cls._sam_predictor.set_image(image_rgb)

        def top_score_mask(masks, scores, logits):
            return masks[np.argmax(scores)]
        # 存储分割掩码和精化后的 bbox
        masks = []
        refined_bboxes = []
        for box in detections.xyxy:
            # 使用 SAM 预测分割掩码
            mask, scores, logits = cls._sam_predictor.predict(box=box, multimask_output=True)
            best_mask = top_score_mask(mask, scores, logits)
            masks.append(best_mask)

            # 计算掩码的边界
            rows, cols = np.where(best_mask)
            if len(rows) == 0 or len(cols) == 0:
                # 如果掩码为空，保留原始 bbox
                refined_bboxes.append(tuple(box.astype(int)))
            else:
                # 计算新的 bbox 边界
                y1, y2 = rows.min(), rows.max()
                x1, x2 = cols.min(), cols.max()
                refined_bboxes.append((x1, y1, x2, y2))

        # 将 masks 转换为 numpy 数组
        masks = np.array(masks)

        return masks, refined_bboxes
    
    #将分割掩码投影到地图平面，融合置信度，更新gsam_map和颜色层
    def _gsam_map_from_planar_projection(self, detections: sv.Detections, camera_pose: Pose4D, image_shape: tuple[int, int]):
        img_row, img_col = image_shape

        resized_masks = _resize_mask(detections.mask, (img_row, img_col))
        confidence_wighted_mask = (resized_masks * detections.confidence.reshape(-1, 1, 1)).max(axis=0)

        # camera xys
        offset_from_center = np.mgrid[-1:1:2/img_row, -1:1:2/img_col] + np.array([1/img_row, 1/img_col]).reshape(2, 1, 1)
        camera_xys = offset_from_center * (camera_pose.z - self.ground_level)

        # world xys
        cos, sin = np.cos(camera_pose.yaw), np.sin(camera_pose.yaw)
        r = np.array([[cos, -sin], [sin, cos]])
        world_xys = (-r @ camera_xys.reshape(2, -1)) + np.array(camera_pose.xy).reshape(2, 1)
        world_xys = world_xys.reshape(2, img_row, img_col).transpose(1, 2, 0)

        # draw map
        map_rows, map_cols = self.to_rows_cols(world_xys.reshape(-1, 2))
        map_rows = map_rows.clip(0, self.shape[0] - 1)
        map_cols = map_cols.clip(0, self.shape[1] - 1)
        gsam_map = np.zeros(self.shape, dtype=np.float32)
        gsam_map[map_rows, map_cols] = confidence_wighted_mask.flatten()

        self._update_color_layer(
            resized_masks=resized_masks,
            world_xys=world_xys,
            phrases=self.phrases,
            img_shape=image_shape
        )
        
        return gsam_map
    
    #根据分割掩码和短语，更新地图的颜色层，实现语义可视化
    def _update_color_layer(self, resized_masks, world_xys, phrases, img_shape):
        """核心颜色更新逻辑"""
        # 将世界坐标转换为地图网格坐标
        rows, cols = self.to_rows_cols(world_xys.reshape(-1, 2))
        map_coords = np.stack([rows, cols], axis=1).reshape(*img_shape, 2).transpose(2,0,1)
        # 创建临时颜色层
        temp_color = np.zeros_like(self.color_array)
        for idx in range(len(resized_masks)):
            mask = resized_masks[idx]
            phrase = phrases[idx].lower()
            if phrase == '':
                continue
            color = self.semantic_colors.get(phrase, self._get_dynamic_color(phrase))
            # 生成掩码区域坐标
            rows, cols = map_coords[0][mask], map_coords[1][mask]
            valid = (rows >= 0) & (rows < self.shape[0]) & (cols >= 0) & (cols < self.shape[1])
            r, c = rows[valid].astype(int), cols[valid].astype(int)
            # 直接覆盖颜色（无透明度混合）
            temp_color[r, c] = color
        # 保留最强颜色（不叠加）
        self.color_array = np.maximum(self.color_array, temp_color)
    
    #利用深度图，将分割掩码投影到三维世界坐标，再映射到地图，实现更精确的地图融合
    def _gsam_map_from_perspective_projection(self, detections: sv.Detections, camera_pose: Pose4D, depth_perspective: np.ndarray, max_depth_meters: float):
        depth_n_row, depth_n_col = depth_perspective.shape

        resized_masks = _resize_mask(detections.mask, (depth_n_row, depth_n_col))
        confidence_wighted_mask = (resized_masks * detections.confidence.reshape(-1, 1, 1)).max(axis=0)
        
        xyz_world_view = _perspective_depth_to_world_xyz(depth_perspective, camera_pose, max_depth_meters)
        
        assert confidence_wighted_mask.shape == (depth_n_row, depth_n_col)
        assert xyz_world_view.shape == (depth_n_row, depth_n_col, 3)

        map_rows, map_cols = self.to_rows_cols(xyz_world_view.reshape(-1, 3)[:, :2])
        map_rows = map_rows.clip(0, self.shape[0] - 1)
        map_cols = map_cols.clip(0, self.shape[1] - 1)
        gsam_map = np.zeros(self.shape, dtype=np.float32)
        gsam_map[map_rows, map_cols] = confidence_wighted_mask.flatten()
        # 新增颜色处理（与平面投影相同逻辑）
        self._update_color_layer(
            resized_masks=resized_masks,
            world_xys=xyz_world_view[..., :2],  # 使用3D坐标的XY平面
            phrases=self.phrases,
            img_shape=(depth_n_row, depth_n_col)
        )

        return gsam_map

#调整掩码尺寸，保证与输入图像一致
def _resize_mask(masks: np.ndarray, resized_shape: tuple[int, int]):
    assert masks.dtype == bool
    masks = masks.view(np.uint8).transpose(1, 2, 0)
    masks = cv2.resize(masks, resized_shape, interpolation=cv2.INTER_NEAREST)
    masks = masks[..., np.newaxis] if masks.ndim == 2 else masks  # cv2.resize() drops the channel dim if n_channel == 1
    masks = masks.view(bool).transpose(2, 0, 1)
    return masks

#将深度图像素点转换为世界坐标（xyz），用于三维投影
def _perspective_depth_to_world_xyz(perspective_depth: np.ndarray, camera_pose: Pose4D, max_depth_meters: float):
    '''converts pinhole depth image to world xyz coords: (row, col, depth) -> (row, col, xyz)'''
    r, c = perspective_depth.shape
    offset_from_center = np.mgrid[-1:1:2/r, -1:1:2/c] + np.array([1/r, 1/c]).reshape(2, 1, 1)
    
    # to planar depth
    ## AirSim's DepthPerspectiveImage returns the distance from the camera pinhole to the pointcloud.
    ## Refer to https://github.com/microsoft/AirSim/discussions/3955#discussioncomment-1159638 for details.
    distance_from_pinhole = np.linalg.norm(offset_from_center, axis=0)
    distance_from_camera_plane = (1 + distance_from_pinhole**2)**(0.5)
    planar_depth = perspective_depth / distance_from_camera_plane

    # to camera xyz
    z = -max_depth_meters * planar_depth
    x, y = z * offset_from_center
    xyz = np.stack([x, y, z])
    assert xyz.shape == (3, r, c)

    # to world xyz
    rotation_matrix = R.from_euler('z', camera_pose.yaw).as_matrix()  # TODO: check if the sign of the yaw is correct
    xyz_world_view = (rotation_matrix @ xyz.reshape(3, -1)) + np.array(camera_pose.xyz).reshape(3, 1)
    xyz_world_view = xyz_world_view.reshape(xyz.shape).transpose(1, 2, 0)  # (3, r, c) -> (r, c, 3)
    assert xyz_world_view.shape == (r, c, 3)

    return xyz_world_view
