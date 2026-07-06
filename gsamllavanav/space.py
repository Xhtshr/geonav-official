from __future__ import annotations

from math import pi, ceil, floor
from typing import NamedTuple

import numpy as np
from shapely.geometry import Polygon


class Point2D(NamedTuple):
    x: float
    y: float

    def dist_to(self, other: Point2D):
        return np.linalg.norm(np.array(self) - np.array(other))


class Point3D(NamedTuple):
    x: float
    y: float
    z: float

    @property
    def xy(self):
        return Point2D(self.x, self.y)

    def dist_to(self, other: Point3D):
        return np.linalg.norm(np.array(self) - np.array(other))


class Pose4D(NamedTuple):
    x: float
    y: float
    z: float
    yaw: float

    @property
    def xyz(self):
        return Point3D(self.x, self.y, self.z)
    
    @property
    def xy(self):
        return Point2D(self.x, self.y)
    
    def with_z(self, new_z: float):
        return Pose4D(self.x, self.y, new_z, self.yaw)

class Pose5D(NamedTuple):
    x: float
    y: float
    z: float
    yaw: float
    pitch: float

    @classmethod
    def from_direction_vector(cls, x: float, y: float, z: float, dx: float, dy: float, dz: float):
        yaw = np.arctan2(dy, dx)#偏转角
        pitch = np.arctan2(dz, np.sqrt(dx**2 + dy**2))#俯仰角
        return Pose5D(x, y, z, yaw, pitch)
    
    @property
    def xyzyaw(self):
        return Pose4D(self.x, self.y, self.z, self.yaw)
    
    @property
    def xyz(self):
        return Point3D(self.x, self.y, self.z)


def bbox_corners_to_position(corners: list[Point2D], ground_level: float):
    
    corners = np.array(corners)
    x, y = corners.mean(axis=0)
    
    box_width = np.linalg.norm(corners[0] - corners[1])
    box_height = np.linalg.norm(corners[0] - corners[-1])
    z = max(box_width, box_height) + ground_level

    return Point3D(x, y, z)


def bbox_IoU(bbox1: list[Point2D], bbox2: list[Point2D]) -> float:
    
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    return intersection / union


def crwh_to_global_bbox(
    crwh: tuple[int, int, int, int],
    image_size: tuple[int, int],
    pose: Pose4D,
    ground_level: float
):
    """converts bbox of format (column, row, width, height) to global xy coordinates"""
    c, r, w, h = crwh
    
    x1 = c - w / 2
    x2 = c + w / 2
    y1 = r - h / 2
    y2 = r + w / 2
    
    return xyxy_to_global_bbox((x1, y1, x2, y2), image_size, pose, ground_level)

#将图像边界框转换到世界坐标系
def xyxy_to_global_bbox(
    xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
    pose: Pose4D,
    ground_level: float
):
    x1, y1, x2, y2 = xyxy
    n_rows, n_cols = image_size

    # 坐标系修正关键步骤
    # 将像素坐标转换为以图像中心为原点的坐标系
    center_col = n_cols / 2
    center_row = n_rows / 2
    
    # 转换为相对于图像中心的坐标（单位：像素）
    bbox_corners_col_row = np.array([
        [x1 - center_col, y1 - center_row],
        [x2 - center_col, y1 - center_row],
        [x2 - center_col, y2 - center_row],
        [x1 - center_col, y2 - center_row],
    ])

    # 计算物理参数
    cos, sin = np.cos(pose.yaw), np.sin(pose.yaw)
    front = np.array([cos, sin])   # 相机前向方向
    left = np.array([-sin, cos])   # 相机左向方向
    view_area_size = pose.z - ground_level
    
    # 计算每个像素对应的物理尺寸（米/像素）
    meter_per_pixel_col = view_area_size / n_cols  # 列方向（左右）
    meter_per_pixel_row = view_area_size / n_rows  # 行方向（前后）

    # 构建坐标变换矩阵
    transform_matrix = np.stack([
        meter_per_pixel_col * left,    # x 轴分量（左方向）
        -meter_per_pixel_row * front   # y 轴分量（前方向反，因为图像行坐标向下增长）
    ], axis=1)

    # 执行坐标变换
    global_points = pose.xy + bbox_corners_col_row @ transform_matrix
    return [Point2D(x, y) for x, y in global_points]


def modulo_radians(theta: float):
    ''' projects radians to range [-pi, pi) '''
    return (theta + pi) % (2*pi) - pi


def view_area_corners(pose: Pose4D, ground_level: float):
    
    cos, sin = np.cos(pose.yaw), np.sin(pose.yaw)
    front = np.array([cos, sin])
    left = np.array([-sin, cos])
    center = np.array([pose.x, pose.y])

    # compute view area corners
    altitude_from_ground = pose.z - ground_level
    view_area_corners_xy = [
        center + altitude_from_ground * (front + left),
        center + altitude_from_ground * (front - left),  # front right
        center + altitude_from_ground * (-front - left),  # back right
        center + altitude_from_ground * (-front + left),  # back left
    ]
    #print(f"View area corners (xy): {view_area_corners_xy}")
    return [Point2D(x, y) for x, y in view_area_corners_xy]