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
    ):
        super().__init__(map_name, map_shape, map_pixels_per_meter)
        # 新增颜色映射属性
        self.semantic_colors = {
            'car': (255, 0, 0, 255),    # 红色
            'building': (210, 105, 30, 255),  # 棕色
            'tree': (0, 100, 0, 255),   # 绿色
            'other': (32, 32, 32, 255)  # 黑色
        }
        self.color_cache = {}  # 用于动态颜色分配
        self.color_array = np.zeros((*map_shape, 4), dtype=np.uint8)  # RGB颜色阵列初始化为黑底透明

        self.captions = captions
        self.gsam_map = np.zeros(map_shape, dtype=np.float32)
        self.gsam_params = gsam_params
        self.obj_list = []
        
        if GSamMap._grounding_dino_model is None or GSamMap._sam_predictor is None:
            GSamMap._init_models(device=device)

    def _get_dynamic_color(self, phrase):
        """动态生成高对比度颜色（优化HSV色轮算法）"""
        if phrase not in self.semantic_colors:
            # 避免与预定义颜色冲突
            predefined_hues = {color[0]: True for color in self.semantic_colors.values()}
            
            # 黄金角分割算法 (137.5度 -> OpenCV的HSV范围0-180对应到68.75)
            golden_angle = 68.75  
            hue = (len(self.color_cache) * golden_angle) % 180
            
            # 跳过红色区域（0-15度）和邻近色
            while any(abs(hue - h) < 20 for h in predefined_hues):  
                hue = (hue + 30) % 180
            
            # 调整饱和度与明度增强对比度
            saturation = 220 + (len(self.color_cache) % 3)*15  # 220-250
            value = 200 + (len(self.color_cache) % 4)*15       # 200-245
            
            # 生成RGB颜色
            rgb_color = cv2.cvtColor(
                np.uint8([[[hue, saturation, value]]]),
                cv2.COLOR_HSV2RGB
            )[0,0].tolist()
            
            self.color_cache[phrase] = (*rgb_color, 255)
            self.semantic_colors[phrase] = self.color_cache[phrase]
        
        return self.semantic_colors[phrase]
    
    def update_observation(
        self,
        camera_pose: Pose4D,
        image_bgr: np.ndarray,
        depth_perspective: Optional[np.ndarray] = None,
        flip_depth=True,
        max_depth_meters=MAX_DEPTH_METERS,
    ):
        if depth_perspective is not None and flip_depth:  # depth sensors often produce vertically flipped images
            depth_perspective = np.flip(depth_perspective, axis=0)

        self.pose = camera_pose
        self.image_bgr = image_bgr

        if (image_bgr > 0).mean() < 0.15:  # skip empty image
            self.detections = None
            self.phrases = None
            return self
        # 在这里不兼容旧方法了
        self.detections, self.phrases = GSamMap._gdino_predict_bboxes(
            image_bgr, self.captions, image_bgr.shape[0] / (2 * (camera_pose.z - self.ground_level)),
            self.gsam_params.box_threshold, self.gsam_params.text_threshold, self.gsam_params.max_box_size, self.gsam_params.max_box_area
        )
        # store obj nodes
        self.obj_list = []
        if self.detections:
            # 生成语义颜色层
            poses = []
            for bbox in self.detections.xyxy:
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                camera_xys = np.array([((x_center/image_bgr.shape[0]) *2 - 1), ((y_center/image_bgr.shape[0]) * 2-1)]) * (camera_pose.z - self.ground_level)
                cos, sin = np.cos(camera_pose.yaw), np.sin(camera_pose.yaw)
                r = np.array([[cos, -sin], [sin, cos]])
                world_xys = (-r @ camera_xys) + np.array(camera_pose.xy)
                poses.append(Point2D(world_xys[0],world_xys[1]))
            for pos, phrase, confidence in zip(poses, self.phrases, self.detections.confidence):
                if phrase == '':
                    continue
                self.obj_list.append((pos, phrase, confidence))
            if self.gsam_params.use_segmentation_mask:
                self.detections.mask = GSamMap._sam_segment(image_bgr, self.detections)
            else:
                self.detections.mask = np.stack([
                    cv2.rectangle(np.zeros(image_bgr.shape[:2], dtype=np.uint8), (x1, y1), (x2, y2), True, -1).astype(bool)
                    for x1, y1, x2, y2 in self.detections.xyxy.astype(np.int32)
                ])

            if depth_perspective is None:
                new_gsam_map = self._gsam_map_from_planar_projection(self.detections, camera_pose, image_bgr.shape[:2])
            else:
                new_gsam_map = self._gsam_map_from_perspective_projection(self.detections, camera_pose, depth_perspective, max_depth_meters)
            
            self.gsam_map = np.maximum(self.gsam_map, new_gsam_map)
            
        return self
    
    
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
        
        return self

    def to_array(self, dtype=np.float32) -> np.ndarray:
        gsam_map = self.gsam_map if self.gsam_params.use_bbox_confidence else self.gsam_map > 0
        return gsam_map[np.newaxis].astype(dtype)
    
    @property
    def max_confidence_bbox(self):

        row, col, channel = self.image_bgr.shape

        xyxy = self.detections.xyxy[np.argmax(self.detections.confidence)] if self.detections else (0, 0, col-1, row-1)
        
        return xyxy_to_global_bbox(xyxy, (row, col), self.pose, self.ground_level)

    
    def plot_bboxes(self, plot_size=(16, 16), show=True):
        if self.detections:
            labels = [f"{phrase} {confidence:0.2f}" for confidence, phrase in zip(self.detections.confidence, self.phrases)]
            annotated_frame = sv.BoxAnnotator().annotate(self.image_rgb.copy(), self.detections, labels)
        else:
            annotated_frame = self.image_rgb

        if show:
            sv.plot_image(annotated_frame, plot_size)

        return annotated_frame

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
    
    def plot(self, plot_size=(16, 16)):
        gsam_map = self.to_array()
        sv.plot_image(gsam_map, plot_size)
        return gsam_map
    
    @property
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
        image_rgb = image_bgr[..., ::-1] # bgr 2 rgb
        cls._sam_predictor.set_image(image_rgb)

        def top_score_mask(masks, scores, logits):
            return masks[np.argmax(scores)]
        
        return np.array([
            top_score_mask(
                *cls._sam_predictor.predict(box=box, multimask_output=True)
            )
            for box in detections.xyxy
        ])
    
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


def _resize_mask(masks: np.ndarray, resized_shape: tuple[int, int]):
    assert masks.dtype == bool
    masks = masks.view(np.uint8).transpose(1, 2, 0)
    masks = cv2.resize(masks, resized_shape, interpolation=cv2.INTER_NEAREST)
    masks = masks[..., np.newaxis] if masks.ndim == 2 else masks  # cv2.resize() drops the channel dim if n_channel == 1
    masks = masks.view(bool).transpose(2, 0, 1)
    return masks


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
