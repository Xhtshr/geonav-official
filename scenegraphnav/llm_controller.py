from openai import OpenAI
from json import dumps as json_dumps  # 添加在文件顶部
from scenegraphnav.parser import parse_args

from shapely.geometry import Point
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose
from gsamllavanav.observation import cropclient
from scenegraphnav.city_scene_graph import GeoNode, ObjectNode, KnowledgeGraph, QueryEngine
from scenegraphnav.prompt.geonav import LOCAL_GRAPH_PROMPT


import os
import json, math
import numpy as np

class LLMController:
    def __init__(self, args: ExperimentArgs, pose: Pose4D):
        self.args = args
        self.pose = Pose4D(pose.x, pose.y, pose.z, 1.5708)
        self.timestep = 0
        # 初始化知识图谱实例
        self.scene_graph = KnowledgeGraph()

    def perceive(self, pose: Pose4D, map_name: str):
        # 使用感知模块处理RGB和深度图像
        rgb = cropclient.crop_image(map_name, pose, (1024, 1024), 'rgb')
        depth = cropclient.crop_image(map_name, pose, (1024, 1024), 'depth') / self.args.max_depth
        self.timestep += 1
        print(f"Timestep {self.timestep}")
        return rgb, depth

    def understand(self, image_64: str, epi: Episode):
        # 从RGB中提取地标和目标的位置信息，使用自然语言处理来存储json格式
        PROMPT = LOCAL_GRAPH_PROMPT.format(objects=[epi.description_target]+epi.description_surroundings)

        client = OpenAI(
            api_key='sk-ca477c37e2214255a5498915ea609ae5',
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        response = client.chat.completions.create(
            model="qwen-vl-max-latest",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional urban scene analyzer with commonsense and strong ability to infer the spatial relationships."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{image_64}"
                        },
                        {"type": "text", "text": PROMPT}
                    ]
                }
            ]
        )

        return json.loads(response.choices[0].message.content)

    def act(self, pose: Pose4D, actions: list, more_info=False):
        new_pose = pose
        pose_list = []
        for action in actions:
            # 执行动作并更新位置
            new_pose = _moved_pose(new_pose, *action.value)
            pose_list.append(new_pose)
        if more_info:
            return pose_list
        return new_pose

    def reached_target(self, pose: Pose4D, target: Point2D):
        # 判断是否到达目标
        print('dist:', pose.xy.dist_to(target.xy))
        return pose.xy.dist_to(target.xy) < self.args.success_dist #-10.0
    
    def build_geo_nodes(self, landmarks):
        for landmark in landmarks:
            self.scene_graph.add_geo_node(landmark)
        self.query_engine = QueryEngine(self.scene_graph)
        landmark = self.query_engine.is_within_geo_node(self.pose.xy)+ self.query_engine.get_enhanced_geo_relation(self.pose.xy)
        return landmark

    def build_scene_graph(self, subgraph, gsm):
        # 过滤node_data.get("bbox", [])==[]的节点
        subgraph["nodes"] = [node for node in subgraph["nodes"] if node.get("bbox", [])]
        id_mapping = {}
        bboxes = [node_data.get("bbox", []) for node_data in subgraph["nodes"]]
        
        # pos, class, confidence, timestamp
        poses = gsm.bbox_to_global_pos(bboxes)
        for pos, node in zip(poses, subgraph["nodes"]):
            # 提取节点的所有属性
            node_attrs = {
                'obj_class': node['object_type'],
                'confidence': 1.0,
                'timestamp': self.timestep
            }
            
            # 添加可选属性
            if 'color' in node:
                node_attrs['color'] = node['color']
            
            # 添加其他可能的属性
            for key, value in node.items():
                if key not in ['id', 'object_type', 'bbox']:
                    node_attrs[key] = value
            
            # 创建节点并添加到场景图中
            global_id = self.scene_graph.add_object_node_with_attrs(pos, node_attrs)
            
            if global_id is not None:
                id_mapping[node['id']] = global_id

                parent_geo, relation_type = self._find_parent_geo(pos)
                if parent_geo:
                    # 添加基于空间关系的边
                    self.scene_graph.add_edge(
                        parent_geo.id, 
                        global_id, 
                        relation_type
                    )
                    
                    # 如果不是 "contains" 关系，还可以添加更详细的空间描述
                    if relation_type != "contains":
                        spatial_rel = self._describe_spatial_relation(
                            parent_geo, 
                            self.scene_graph.nodes[global_id]
                        )
                        self.scene_graph.add_edge(
                            parent_geo.id,
                            global_id,
                            spatial_rel
                        )

        # 处理边
        for edge in subgraph.get("edges", []):
            source = id_mapping.get(edge["source"])
            target = id_mapping.get(edge["target"])
            if source is None or target is None: 
                continue
            
            # 提取边的属性
            relationship = edge["relationship"]
            edge_attrs = {}
            
            # 添加其他可能的属性
            for key, value in edge.items():
                if key not in ['source', 'target', 'relationship']:
                    edge_attrs[key] = value
            
            # 添加边到场景图中
            self.scene_graph.add_edge(source, target, relationship, **edge_attrs)
        
        self.query_engine = QueryEngine(self.scene_graph) # 用于查询场景图
        
    def build_scene_nodes(self, targets, surroundings, show=True, time_window=3):
        for object in targets:
            self.scene_graph.add_object_node(object[0], object[1], object[2], self.timestep, target=True)
        for object in surroundings:
            self.scene_graph.add_object_node(object[0], object[1], object[2], self.timestep)

        # Query scenary context
        self.query_engine = QueryEngine(self.scene_graph)
        current_pos = self.pose.xy
        context = self.query_engine.get_context(self.pose.xy, radius=30.0)

        surrounding = ''
        recent_objects = self.query_engine .get_recent_objects(current_timestamp=self.timestep, time_window=time_window)
        surrounding += f"Current Position is {(round(current_pos.x,2), round(current_pos.y,2))}"
        for item in context:
            surrounding += f"The {item['type'].upper()} Node {item.get('name','')}{item.get('class','')} is {item['distance']} to the {item['direction']}."

        # 输出当前graph规模和节点信息
        print(f"Current graph size: {len(self.scene_graph.nodes)} nodes")
        return surrounding, recent_objects

    def parse_instruction(self, instruction: str, landmarks):
        # 解析指令，提取地标和目标
        from scenegraphnav.prompt.instruction import create_prompt, gpt_api_call
        from scenegraphnav.agent import extract_json_from_msg
        prompt = create_prompt(instruction, landmarks)
        response = gpt_api_call(prompt)
        self.intr_knowledge = extract_json_from_msg(response)
        return self.intr_knowledge
    
    def _calc_distance(self, p1: Point2D, p2: Point2D):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    def _get_direction(self, src: Point2D, target: Point2D):
        DIRECTION_NAMES = ["East", "Northeast", "North", "Northwest", "West", "Southwest", "South", "Southeast"]
        dx = target.x - src.x
        dy = target.y - src.y
        angle = math.degrees(math.atan2(dy, dx)) % 360
        return DIRECTION_NAMES[round(angle / 45) % 8]
    def _find_parent_geo(self, position: Point2D):
        """查找与当前坐标相关的地理父节点，考虑多种空间关系"""
        point = Point(position.x, position.y)
        
        # 存储候选地理节点及其关系类型和距离
        candidates = []
        
        for node in self.scene_graph.nodes.values():
            if not isinstance(node, GeoNode):
                continue
            
            polygon = node.contour_polygon
            distance = self._calc_distance(node.position, position)
            
            # 情形1：在轮廓多边形内部（最高优先级）
            if polygon.contains(point):
                return node, "contains"
            
            # 情形2：接近多边形边界（次高优先级）
            if polygon.distance(point) < 10.0:  # 10米内视为接近边界
                candidates.append((node, "adjacent_to", distance))
                continue
            
            # 情形3：接近多边形顶点（角落检测）
            closest_corner_dist = float('inf')
            for coord in polygon.exterior.coords[:-1]:  # 排除重复的闭合点
                corner_dist = math.hypot(position.x-coord[0], position.y-coord[1])
                if corner_dist < closest_corner_dist:
                    closest_corner_dist = corner_dist
                
            if closest_corner_dist < 15.0:  # 15米内视为接近角落
                candidates.append((node, "near_corner", distance))
                continue
            
            # 情形4：在一定距离内的普通关系
            if distance < 50.0:  # 50米内视为相关
                direction = self._get_direction(position, node.position)
                candidates.append((node, f"{direction.lower()}_of", distance))
        
        # 按距离排序，选择最近的候选节点
        if candidates:
            candidates.sort(key=lambda x: x[2])  # 按距离排序
            return candidates[0][0], candidates[0][1]
        
        return None, None
    
    def _describe_spatial_relation(self, parent: GeoNode, child: ObjectNode):
        """生成更细致的自然语言描述的空间关系"""
        dx = child.position.x - parent.position.x
        dy = child.position.y - parent.position.y
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360  # 确保角度为正值
        
        relations = {
            (337.5, 22.5): "directly in front of",
            (22.5, 67.5): "to the top right of",
            (67.5, 112.5): "to the right of",
            (112.5, 157.5): "to the bottom right of",
            (157.5, 202.5): "directly behind",
            (202.5, 247.5): "to the bottom left of",
            (247.5, 292.5): "to the left of",
            (292.5, 337.5): "to the top left of"
        }
        
        for (start, end), desc in relations.items():
            if start <= angle < end:
                return desc
        return "near"
    