from openai import OpenAI
from utils.tools import get_direction

from shapely.geometry import Point
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.space import Point2D, Pose4D
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

        for attempt in range(2):  # 尝试3次
            try:
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

                # 检查 response 的结构
                if not response or not hasattr(response, "choices") or not response.choices:
                    print(f"完整的 API 返回内容: {response}")
                    raise ValueError("API 返回的 response 结构无效或为空")

                # 检查 choices[0] 是否存在
                if not response.choices[0] or not hasattr(response.choices[0], "message") or not response.choices[0].message:
                    print(f"完整的 API 返回内容: {response}")
                    raise ValueError("API 返回的 choices[0].message 结构无效或为空")

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("API 返回的 message.content 为空")

                # 清理返回内容
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]

                # 尝试解析 JSON
                try:
                    json_data = json.loads(content)
                    return json_data
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败: {e}")
                    print(f"返回内容: {content}")
                    raise e

            except Exception as e:
                print(f"调用 API 时发生错误: {str(e)}")
                if attempt == 1:  # 如果是最后一次尝试，抛出异常
                    raise e

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
                    if relation_type != "contains" and relation_type != "adjacent_to":
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
    
    def query_scene_graph(self, instruction: str, debug=False):
        """使用增强版的robust_subgraph_query查询场景图"""
        from scenegraphnav.prompt.geonav import QUERY_OPERATION_CHAIN_PROMPT
        import json
        
        # 使用LLM生成操作链
        prompt = QUERY_OPERATION_CHAIN_PROMPT.format(instruction=instruction)
        client = OpenAI(
            api_key= os.environ.get("OPENAI_API_KEY", 'sk-xHX92exOc6iulrMz8q8BGcXOveU8qVgpfDkvdXdbctOA4rOr'),
            base_url= os.environ.get("OPENAI_BASE_URL", 'https://api.chatanywhere.tech'),
        )
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional query planner."},
                {"role": "user", "content": prompt}
            ]
        )
        
        try:
            operation_chain = json.loads(response.choices[0].message.content)
            if debug:
                print(f"生成的操作链: {json.dumps(operation_chain, indent=2, ensure_ascii=False)}")
                
            # 使用robust_subgraph_query执行查询
            results = self.query_engine.robust_subgraph_query(
                operation_chain, fallback=True, min_results=1, debug=debug
            )
            
            return results
        except Exception as e:
            print(f"查询场景图时发生错误: {str(e)}")
            return []
    
    def complex_query_scene_graph(self, instruction: str, debug=False):
        """处理复杂查询，将其分解为多个子查询并验证结果之间的关系"""
        # 分解复杂查询为多个子查询
        sub_queries = self.decompose_complex_query(instruction, debug)
        if not sub_queries:
            # 如果无法分解，则使用普通查询
            if debug:
                print("无法分解查询，使用普通查询")
            return self.query_scene_graph(instruction, debug)
        
        # 执行每个子查询
        sub_results = []
        for i, sub_query in enumerate(sub_queries):
            if debug:
                print(f"子查询 {i+1}: {sub_query}")
            
            # 执行子查询
            results = self.query_scene_graph(sub_query, debug=debug)
            sub_results.append((sub_query, results))
            
            if debug:
                print(f"子查询 {i+1} 返回 {len(results)} 个结果")
        
        # 验证结果之间的关系
        final_results = self.verify_relation_constraints(instruction, sub_results, debug)
        
        return final_results
    
    def decompose_complex_query(self, instruction: str, debug=False):
        """将复杂查询分解为多个简单子查询"""
        prompt = f"""
        将以下复杂指令分解为多个简单查询步骤:
        
        指令: "{instruction}"
        
        每个步骤应关注一个具体物体及其特征。例如，指令"白色车前面有灰色车，后面是另一辆白色车"可以分解为:
        1. 查找白色车
        2. 查找灰色车在白色车前面
        3. 查找另一辆白色车在第一辆白色车后面
        
        请输出分解后的简单查询列表，每个查询应该是完整的句子。
        """
        
        client = OpenAI(
            api_key= os.environ.get("OPENAI_API_KEY", 'sk-xHX92exOc6iulrMz8q8BGcXOveU8qVgpfDkvdXdbctOA4rOr'),
            base_url= os.environ.get("OPENAI_BASE_URL", 'https://api.chatanywhere.tech'),
        )
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a professional query analyzer."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        
        # 提取子查询列表
        import re
        sub_queries = []
        
        # 匹配格式为 "1. xxx" 或 "1) xxx" 或 "- xxx" 的文本行
        pattern = r'(?:\d+[\.\)]\s+|\-\s+)(.+)'
        matches = re.findall(pattern, result)
        
        if matches:
            sub_queries = [match.strip() for match in matches if match.strip()]
        
        if debug:
            print(f"分解查询结果: {sub_queries}")
        
        return sub_queries
    
    def verify_relation_constraints(self, original_instruction, sub_results, debug=False):
        """验证多个查询结果之间的关系约束"""
        if not sub_results or any(not results for _, results in sub_results):
            if debug:
                print("子查询结果为空，无法验证关系")
            # 返回第一个非空结果，或空列表
            for _, results in sub_results:
                if results:
                    return results
            return []
        
        # 构建结果描述
        results_desc = []
        for i, (query, nodes) in enumerate(sub_results):
            nodes_desc = []
            for j, node in enumerate(nodes[:5]):  # 限制每个查询最多描述5个结果
                node_desc = f"节点{j+1} (ID: {node.id}, "
                if hasattr(node, 'obj_class'):
                    node_desc += f"类型: {node.obj_class}, "
                if hasattr(node, 'color'):
                    node_desc += f"颜色: {node.color}, "
                node_desc += f"位置: ({node.position.x:.1f}, {node.position.y:.1f}))"
                nodes_desc.append(node_desc)
            
            results_desc.append(f"查询 {i+1}: {query}\n结果: {'; '.join(nodes_desc)}")
        
        joined_results = "\n\n".join(results_desc)
        prompt = f"""
        分析以下多个查询结果，确定哪些对象组合最符合原始指令要求。

        原始指令: "{original_instruction}"

        查询结果:
        {joined_results}

        请确定哪些对象组合最可能满足原始指令的所有关系约束，并给出理由。
        返回JSON格式:
        {{
            "best_match": [
                {{"query_index": 查询索引, "node_index": 节点索引}},
                ...
            ],
            "reason": "选择理由"
        }}

        如果没有完全符合的组合，返回最接近的组合。
        """
        
        client = OpenAI(
            api_key= os.environ.get("OPENAI_API_KEY", 'sk-xHX92exOc6iulrMz8q8BGcXOveU8qVgpfDkvdXdbctOA4rOr'),
            base_url= os.environ.get("OPENAI_BASE_URL", 'https://api.chatanywhere.tech'),
        )
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional spatial relation analyzer."},
                {"role": "user", "content": prompt}
            ]
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            
            if debug:
                print(f"关系验证结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if 'best_match' in result and result['best_match']:
                # 提取最佳匹配节点
                best_nodes = []
                for match in result['best_match']:
                    query_idx = match.get('query_index')
                    node_idx = match.get('node_index')
                    
                    if query_idx is not None and node_idx is not None:
                        try:
                            query_idx = int(query_idx) - 1  # 调整为0-索引
                            node_idx = int(node_idx) - 1    # 调整为0-索引
                            
                            if 0 <= query_idx < len(sub_results):
                                _, nodes = sub_results[query_idx]
                                if 0 <= node_idx < len(nodes):
                                    best_nodes.append(nodes[node_idx])
                        except (ValueError, IndexError) as e:
                            if debug:
                                print(f"索引解析错误: {str(e)}")
                
                if best_nodes:
                    if debug:
                        print(f"找到 {len(best_nodes)} 个最佳匹配节点")
                    return best_nodes
        
        except Exception as e:
            if debug:
                print(f"关系验证错误: {str(e)}")
        
        # 如果无法确定最佳组合，返回第一个查询的结果
        if debug:
            print("无法确定最佳组合，返回第一个查询的结果")
        return sub_results[0][1] if sub_results else []
    
    def _calc_distance(self, p1: Point2D, p2: Point2D):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
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
            if polygon.distance(point) < 20.0:  # 10米内视为接近边界
                candidates.append((node, "adjacent_to", distance))
                continue
            
            # 情形3：接近多边形顶点（角落检测）
            closest_corner_dist = float('inf')
            for coord in polygon.exterior.coords[:-1]:  # 排除重复的闭合点
                corner_dist = math.hypot(position.x-coord[0], position.y-coord[1])
                if corner_dist < closest_corner_dist:
                    closest_corner_dist = corner_dist
                
            if closest_corner_dist < 35.0:  # 15米内视为接近角落
                candidates.append((node, "near_corner", distance))
                continue
            
            # 情形4：在一定距离内的普通关系
            if distance < 80.0:  # 50米内视为相关
                direction = get_direction(position, node.position)
                candidates.append((node, f"{direction.lower()}_of", distance))
        
        # 按距离排序，选择最近的候选节点
        if candidates:
            candidates.sort(key=lambda x: x[2])  # 按距离排序
            return candidates[0][0], candidates[0][1]
        
        return None, None
    
    def _describe_spatial_relation(self, parent: GeoNode, child: ObjectNode):
        """生成一致的空间关系描述"""
        dx = child.position.x - parent.position.x
        dy = child.position.y - parent.position.y
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360  # 确保角度为正值
        
        # relations = {
        #     (337.5, 22.5): "north_of",      # 正北
        #     (22.5, 67.5): "northeast_of",   # 东北
        #     (67.5, 112.5): "east_of",       # 正东
        #     (112.5, 157.5): "southeast_of", # 东南
        #     (157.5, 202.5): "south_of",     # 正南
        #     (202.5, 247.5): "southwest_of", # 西南
        #     (247.5, 292.5): "west_of",      # 正西
        #     (292.5, 337.5): "northwest_of"  # 西北
        # }
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
        return "adjacent_to"  # 默认关系保持不变
    