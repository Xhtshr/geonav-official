import cv2
import numpy as np
import Levenshtein

from gsamllavanav.cityreferobject import get_landmarks, remove_duplicate_landmarks_by_area

from .map import Map


class LandmarkMap(Map):
    _landmarks_cache = None
    _landmark_segmentations = None

    def __init__(
        self,
        map_name: str,
        map_shape: tuple[int, int],
        pixels_per_meter: float,
        landmark_names: list[str],
    ):
        super().__init__(map_name, map_shape, pixels_per_meter)
        self.landmark_names = landmark_names
        self.landmarks = LandmarkMap._search_landmarks_by_name(map_name, landmark_names)

        # 新增灰度颜色配置
        self.gray_colors = self._generate_gray_colors()
        self.color_array = np.zeros((*map_shape, 4), dtype=np.uint8)  # 改为四通道

        self.landmark_map = np.zeros(map_shape, dtype=np.uint8)
        for lm, color in zip(self.landmarks, self.gray_colors.values()):
            # 转换轮廓坐标为图像坐标
            pts = np.stack(self.to_rows_cols(lm.contour))[::-1].T  # [N,2]数组
            #print(pts)
            # 绘制彩色轮廓
            cv2.polylines(
            img=self.color_array,
            pts=[pts.astype(np.int32)],
            isClosed=True,
            color=(*color, 255),  # 添加alpha通道
            thickness=2
            )
    def _generate_gray_colors(self):
        """为每个地标生成唯一颜色（HSV色轮算法）"""
        """生成渐变灰度色（从浅灰到深灰）"""
        base_gray = 200  # 提高基础灰度值
        gray_step = 30   # 减小级差
        return {
            name: (base_gray - i*gray_step, )*3
            for i, name in enumerate(self.landmark_names)
        }
    
    def to_array(self, dtype=np.float32) -> np.ndarray:
        return self.landmark_map[np.newaxis].astype(dtype)
    
    @classmethod
    def _search_landmarks_by_name(cls, map_name: str, query_names: list[str]):
        # load landmark data
        if cls._landmarks_cache is None:
            cls._landmarks_cache = remove_duplicate_landmarks_by_area(get_landmarks())

        landmarks = cls._landmarks_cache[map_name].values()

        return [
            min(landmarks, key=lambda lm, q=query: Levenshtein.distance(lm.name, q))
            for query in query_names
        ]
