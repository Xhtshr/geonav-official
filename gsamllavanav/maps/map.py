import numpy as np

from gsamllavanav.mapdata import MAP_BOUNDS, GROUND_LEVEL
from gsamllavanav.space import Point2D


class Map:

    def __init__(self, map_name: str, shape: tuple[int, int], pixels_per_meter: float):
        self.name = map_name
        self.shape = shape
        self.pixels_per_meter = pixels_per_meter
    
    @property#返回当前地图的边界（x_min, y_min, x_max, y_max），用于坐标转换
    def bounds(self):
        return MAP_BOUNDS[self.name]
    
    @property#返回当前地图的地面高度（z轴），用于高度相关计算
    def ground_level(self):
        return GROUND_LEVEL[self.name]
    
    @property#返回地图的实际物理尺寸（以米为单位）
    def size_meters(self):
        return self.shape[0] / self.pixels_per_meter
    
    #将世界坐标（x, y，单位米）转换为地图像素坐标（row, col），适用于单个点
    def to_row_col(self, world_xy: Point2D) -> tuple[int, int]:
        x, y = world_xy
        x_min, y_min, x_max, y_max = self.bounds

        col = round((x - x_min) * self.pixels_per_meter)
        row = round((y_max - y) * self.pixels_per_meter)

        return row, col
    
    #将一组世界坐标批量转换为像素坐标（row, col），返回两个 numpy 数组，适用于批量点
    def to_rows_cols(self, world_xys: list[Point2D]) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.array(world_xys).T
        x_min, y_min, x_max, y_max = self.bounds

        col = np.round((x - x_min) * self.pixels_per_meter).astype(int)
        row = np.round((y_max - y) * self.pixels_per_meter).astype(int)

        return row, col
    
    #将像素坐标（row, col）转换为世界坐标（x, y，单位米），适用于单个点
    def to_world_xy(self, row: int, col: int) -> Point2D:
        x_min, y_min, x_max, y_max = self.bounds

        x = x_min + col / self.pixels_per_meter
        y = y_max - row / self.pixels_per_meter

        return Point2D(x, y)

    #将一组像素坐标批量转换为世界坐标（x, y，单位米），返回一个 Point2D 列表，适用于批量点
    def to_world_xys(self, rows: np.ndarray, cols: np.ndarray) -> list[Point2D]:
        x_min, y_min, x_max, y_max = self.bounds

        xs = x_min + cols / self.pixels_per_meter
        ys = y_max - rows / self.pixels_per_meter
        xys = [Point2D(x, y) for x, y in zip(xs, ys)]

        return xys

    #根据给定的高度 z（单位米），计算从地面到该高度的像素半径，常用于视野范围、感知半径等场景
    def view_radius_pixels(self, z: float) -> int:
        altitude_from_ground = z - self.ground_level
        return round(altitude_from_ground * self.pixels_per_meter)