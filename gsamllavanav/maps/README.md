**Maps Overview**

本目录包含用于地标导航与语义地图的几种 Map 实现。本文档简要说明每种地图的职责、数据接口、通道顺序以及常见使用方式和注意事项，方便团队调用与调试。

Files:
- `map.py` — 基类 `Map`，提供世界坐标 ↔ 像素坐标转换方法与地图元信息。
- `tracking_map.py` — `TrackingMap`：当前视野与累积已探索区域（2 通道）。
- `landmark_map.py` — `LandmarkMap`：城市地标轮廓与灰度/颜色层（1 通道 + color array）。
- `gsam_map.py` — `GSamMap`：基于 GroundingDINO + SAM 的视觉语义映射（1 通道置信度 + color array）。
- `landmark_nav_map.py` — `LandmarkNavMap`：上层组合器，将上述子地图拼接为导航用多通道地图并负责可视化。

核心概念
-----------
- Map（基类）
  - 提供 `to_row_col`, `to_rows_cols`, `to_world_xy` 等方法，统一世界坐标与像素坐标系。依赖项目中的 `MAP_BOUNDS` 与 `GROUND_LEVEL`。

- TrackingMap
  - 成员: `current_view_area`, `explored_area`（uint8 mask）
  - 方法: `mark_current_view_area(pose)` 把当前相机视野写入 `current_view_area` 并累加进 `explored_area`。
  - `to_array()` 返回 shape (2, H, W)。

- LandmarkMap
  - 从城市地标数据库匹配地标（按名称模糊匹配），生成 `landmark_map`（单通道）和 `color_array`（RGBA 可视化）。
  - `to_array()` 返回 shape (1, H, W)。

- GSamMap
  - 使用 GroundingDINO + SAM（或 MobileSAM）在单帧图像上检测/分割短语对应的物体，并把分割掩码或 bbox 投影到地图平面，生成 `gsam_map`（float32 置信度图）。
  - 支持两种更新方式：实时推理 `update_observation(...)`（代价高）或从预计算缓存 `update_from_map_cache(...)`（速度快，适合评估）。
  - `to_array()` 返回 shape (1, H, W)。`color_array` 用于可视化，不一定作为训练输入。

- LandmarkNavMap
  - 组合器，将子图层合并为一个多通道地图，并负责轨迹记录及绘图。
  - `to_array()` 返回 shape (5, H, W)，通道顺序（非常重要）：
    0. tracking_map.current_view_area
    1. tracking_map.explored_area
    2. landmark_map.landmark_map
    3. target_map.gsam_map
    4. surroundings_map.gsam_map
  - `update_observations(camera_pose, rgb, depth_perspective=None, use_gsam_map_cache=True)`：首选缓存模式以加速评估；交互式或在线运行可选择实时推理。

使用示例（Python 片段）
-------------------------
从 `LandmarkNavMap` 获取多通道地图并查看单通道：

```python
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap, GSamParams

# 假设已有 episode/map_shape/pixels_per_meter
nav_map = LandmarkNavMap(map_name, map_shape, pixels_per_meter, landmark_names, target_name, surroundings_names, GSamParams(...))
nav_map.update_observations(camera_pose, rgb_image, depth_perspective=None, use_gsam_map_cache=True)
arr = nav_map.to_array()  # shape (5, H, W)
current_view = arr[0]     # 当前视野
landmarks = arr[2]        # 地标掩码
```

可视化（使用 color_array 或 plot 接口）：

```python
img_b64 = nav_map.plot('TopV', query_engine=query_engine, current_pos=current_pos)
# 或使用 vanilla_plot / eva_plot 返回 base64 编码
```

注意与建议（调试要点）
-------------------------
- 确保 `MAP_BOUNDS` 与 `pixels_per_meter` 一致，否则世界 ↔ 像素的转换会出错。
- `GSamMap` 首次调用会加载大模型（GroundingDINO/SAM），建议使用 GPU 并避免在每帧重复加载（类级缓存已实现）。
- 对大规模评估强烈建议事先生成 `GSAM_MAPS_DIR` 的缓存并在 `LandmarkNavMap.update_observations(..., use_gsam_map_cache=True)` 中使用。
- headless 服务器请设置 matplotlib 使用 `Agg` backend，并确保绘图后调用 `plt.close(fig)` 释放内存。
- `LandmarkNavMap.plot()` 中曾发现若干可修复点（例如重复创建 figure、保存目录未确保存在、`vanilla_plot` 中的文件名不一致）。若需我可以自动应用小补丁。

如何生成 GSAM 缓存（建议）
-----------------------------
常见做法是遍历整张大地图在若干飞行高度/视野参数下，使用 `GSamMap._gsam_map_from_planar_projection` 生成切片并保存为 npz，后续评估直接载入 `GSAM_MAPS_DIR` 下的 `full_scan_{params}.npz`。具体实现会依赖你的场景与采样策略，如需我可以提供脚本模板。

常见问题与快速排查
-------------------
- 若可视化报错：检查 matplotlib backend、确认 `self.save_path` 目录存在。
- 若 GSAM 没有检测到任何物体：检查 GroundingDINO/SAM checkpoint 路径是否正确（`gsamllavanav/defaultpaths.py`）。
- 若目标坐标偏移：检查 `camera_pose.z` 与 `ground_level` 是否一致及 `pixels_per_meter` 是否正确。

下一步建议
----------------
- 我可以立即提交小补丁修复 `landmark_nav_map.py` 中的几个明显 bug（重复创建 figure、目录检查、文件名不一致），或
- 帮你编写生成 GSAM 缓存的脚本模板。
