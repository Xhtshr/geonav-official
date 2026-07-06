from openai import OpenAI
from utils.tools import get_direction

from shapely.geometry import Point
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.space import Point2D, Pose4D, xyxy_to_global_bbox
from gsamllavanav.teacher.trajectory import _moved_pose
from gsamllavanav.observation import cropclient
from scenegraphnav.city_scene_graph import GeoNode, ObjectNode, KnowledgeGraph, QueryEngine
from scenegraphnav.prompt.geonav_cot import LOCAL_GRAPH_PROMPT
from scenegraphnav.prompt.geonav_cot import LOCAL_GRAPH_PROMPT_V2
from scenegraphnav.prompt.config import VLM_NAME, LLM_NAME

import os, re, ast, json
import json, math
import numpy as np
import logging 
from functools import wraps

logger = logging.getLogger(__name__)

def validate_edge(edge_data):
    """Verify the integrity of the edge data"""
    required_fields = ['source', 'target']
    return all(field in edge_data for field in required_fields)

def safe_graph_operation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper


class LLMController:
    def __init__(self, args: ExperimentArgs, pose: Pose4D, vlm: OpenAI, llm: OpenAI):
        self.args = args
        self.pose = Pose4D(pose.x, pose.y, pose.z, 1.5708)
        self.timestep = 0
        # 初始化知识图谱实例
        self.scene_graph = KnowledgeGraph()
        self.vlm_client = vlm
        self.llm_client = llm

    def perceive(self, pose: Pose4D, map_name: str):
        # 使用感知模块处理RGB和深度图像
        rgb = cropclient.crop_image(map_name, pose, (1024, 1024), 'rgb')
        depth = cropclient.crop_image(map_name, pose, (1024, 1024), 'depth') / self.args.max_depth
        self.timestep += 1
        print(f"Timestep {self.timestep}")
        return rgb, depth

    def understand(self, image_64: str, epi: Episode):
        # 从RGB中提取周围环境和目标的位置信息（没有地标），使用自然语言处理来存储json格式
        # 定性空间关系表示：两级空间关系建立HSG
        PROMPT = LOCAL_GRAPH_PROMPT.format(instruction=epi.target_description, objects=[epi.description_target]+epi.description_surroundings)
        #PROMPT = LOCAL_GRAPH_PROMPT.format(objects=epi.description_landmarks+[epi.description_target]+epi.description_surroundings)

        for attempt in range(2):  # 尝试2次
            try:
                response = self.vlm_client.chat.completions.create(
                    model=VLM_NAME,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a geospatial scene graph extractor analyzing a north-aligned satellite image."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_64}"
                                    }
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

                # 处理可能的多余内容
                if "```json" in content:
                    content = content.split("```json")[1]
                if "```" in content:
                    content = content.split("```")[0]
                content = content.strip()

                # 尝试修复常见的 JSON 格式问题
                content = content.replace("\n", "").replace("\r", "")  # 移除换行符
                content = re.sub(r',(\s*[}\]])', r'\1', content)  # 移除尾随逗号
                # 删除JSON中的注释行（以//开头的行）
                content = re.sub(r'//.*?(?=[\r\n}])', '', content)
                
                # 增强对尾随逗号的处理，处理对象和数组内的尾随逗号
                content = re.sub(r',(\s*[}\]])', r'\1', content)

                # 尝试解析 JSON
                try:
                    json_data = json.loads(content)
                    print(f"成功解析 JSON 数据: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
                    return json_data
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败: {e}")
                    print(f"返回内容: {content}")
                    # 不再重新抛出异常，而是继续循环尝试或者返回默认值

            except Exception as e:
                print(f"调用 API 时发生错误: {str(e)}")
                # 在最后一次尝试时，不抛出异常而是返回默认值
                if attempt == 1:  # 如果是最后一次尝试，返回默认值而不是抛出异常
                    print("API调用失败，返回空的场景图")
                    return {
                        "nodes": [],
                        "edges": []
                    }
        # 如果所有尝试都失败了，返回默认值
        return {
            "nodes": [],
            "edges": []
        }

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
        return pose.xy.dist_to(target.xy) <= self.args.success_dist #-10.0
    
    def build_geo_nodes(self, landmarks):
        for landmark in landmarks:
            self.scene_graph.add_geo_node(landmark)
        #self.query_engine = QueryEngine(self.scene_graph)
        self.query_engine = QueryEngine(self.scene_graph, controller=self)
        landmark_describe = self.query_engine.is_within_geo_node(self.pose.xy)+ self.query_engine.get_enhanced_geo_relation(self.pose.xy)
        return landmark_describe
    
    def build_geo_nodes_describe(self, landmarks):
        for landmark in landmarks:
            self.scene_graph.add_geo_node(landmark)
        #self.query_engine = QueryEngine(self.scene_graph)
        self.query_engine = QueryEngine(self.scene_graph, controller=self)
        landmark_describe = self.query_engine.describe_landmark_positions()
        return landmark_describe
    
    def polygon_to_polygon_distance(self, poly1, poly2):
        """
        计算两个多边形之间的最小距离（边界到边界的距离）
        如果相交则返回0
        """
        if poly1.intersects(poly2):
            return 0.0
        # 计算边界之间的最小距离
        min_dist = float('inf')
        # 多边形的每条边
        for edge1 in poly1.exterior.coords[:-1]:
            for edge2 in poly2.exterior.coords[:-1]:
                dist = math.hypot(edge1[0] - edge2[0], edge1[1] - edge2[1])
                if dist < min_dist:
                    min_dist = dist
        return min_dist
    
    def get_object_geo_relations(self, object_pos, object_bbox_corners, geo_node: GeoNode):
        """
        判断 ObjectNode 和 GeoNode 之间的所有关系
        参数：
            object_pos: ObjectNode 的位置 (Point2D)
            object_bbox_corners: 物体 bbox 的四个角点在全局坐标系下的坐标列表 [(x,y), (x,y), (x,y), (x,y)]
            geo_node: GeoNode 对象
        返回：
            list: 关系类型列表 ["contains", "north_of", ...]
        """
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        relations = []
        polygon = geo_node.contour_polygon
        '''
        # 安全检查
        if not polygon:
            direction = get_direction(geo_node.position, object_pos)
            print("landmark is invalid")
            print(f"landmark polygon: {polygon}")
            relations.append(f"{direction.lower()}_of")
            return relations
        # 如果 bbox_corners 无效，只返回方向关系
        if not object_bbox_corners or len(object_bbox_corners) != 4:
            print(f"Warning: invalid bbox_corners (len={len(object_bbox_corners) if object_bbox_corners else None})")
            print(f"object_bbox_corners: {object_bbox_corners}")
            direction = get_direction(geo_node.position, object_pos)
            relations.append(f"{direction.lower()}_of")
            return relations
        '''
        
        # 新版地标节点与对象节点的连接
        
        # 构建物体的 bbox 多边形
        obj_polygon = Polygon(object_bbox_corners)
        # 确保多边形有效
        if not obj_polygon.is_valid:
            obj_polygon = make_valid(obj_polygon)
        '''
        # 1. 判断空间拓扑关系
        min_distance = self.polygon_to_polygon_distance(obj_polygon, polygon)
        # contains: 物体完全在地标轮廓内
        if polygon.contains(obj_polygon):
            relations.append("contains")
        # overlaps: 物体与地标部分重叠或相切
        elif polygon.intersects(obj_polygon):
            relations.append("overlaps")
        # separates: 物体与地标相离，且最近距离小于50米
        elif min_distance < 50.0:
            relations.append("separates")
            # 2. 八方位计算 (子节点相对于父节点的方位)
            direction = get_direction(geo_node.position, object_pos)
            relations.append(f"{direction.lower()}_of")
        '''    
        # 师兄原版地标节点与对象节点的连接
        
        # 1. 判断空间拓扑关系
        min_distance = self.polygon_to_polygon_distance(obj_polygon, polygon)
        # contains: 物体完全在地标轮廓内
        if polygon.contains(obj_polygon):
            relations.append("contains")
        elif min_distance < 20.0:
            relations.append("adjacent_to")
        elif min_distance < 35.0:
            relations.append("near_corner")
        else:
            direction = get_direction(geo_node.position, object_pos)
            relations.append(f"{direction.lower()}_of")
        
        return relations
    
    @safe_graph_operation
    def build_scene_graph(self, subgraph, gsm):
        if not isinstance(subgraph, dict):
            logger.error("Invalid subgraph format")
            return
        # 过滤有效节点
        subgraph["nodes"] = [
            node for node in subgraph.get("nodes", []) 
            if node.get("bbox", []) and isinstance(node.get("object_type"), str)
        ]
        id_mapping = {}
        bboxes = [node_data.get("bbox", []) for node_data in subgraph["nodes"]]
        #将bbox转换为全局坐标
        bbox_corners_list = [
            xyxy_to_global_bbox(bbox, gsm.image_bgr.shape[:2], gsm.pose, gsm.ground_level)
            for bbox in bboxes
        ]
        #将边界框坐标转换为全局地理坐标，并计算中心点坐标
        poses = gsm.bbox_to_global_pos(bboxes)
        # 处理节点
        for pos, corners, node in zip(poses, bbox_corners_list, subgraph["nodes"]):
            try:
                node_attrs = {
                    'obj_class': node.get('object_type', 'unknown'),
                    'confidence': 1.0,
                    'timestamp': self.timestep
                }
                node_attrs.update({
                    k: v for k, v in node.items() 
                    if k not in ['id', 'object_type', 'bbox']
                })
                global_id = self.scene_graph.add_object_node_with_attrs(pos, node_attrs)
                if global_id:
                    id_mapping[node.get('id')] = global_id
                    # 找到与该 ObjectNode 相关的 GeoNode，并添加多条边
                    for gn in self.scene_graph.nodes.values():
                        if isinstance(gn, GeoNode):
                            # 传递 bbox 四个角点的坐标进行关系判断
                            corners_coords = [(p.x, p.y) for p in corners]
                            relations = self.get_object_geo_relations(pos,corners_coords, gn)
                            for relation in relations:
                                self.scene_graph.add_edge(gn.id, global_id, relation, **{})
                            
            except Exception as e:
                logger.error(f"Error processing node: {str(e)}")
                continue
        # 处理边
        for edge in subgraph.get("edges", []):
            try:
                if not validate_edge(edge):
                    logger.warning(f"Invalid edge data: {edge}")
                    continue
                    
                source = id_mapping.get(edge["source"])
                target = id_mapping.get(edge["target"])
                
                if not (source and target):
                    continue
                    
                relationship = edge.get("relationship", "related_to")
                edge_attrs = {
                    k: v for k, v in edge.items() 
                    if k not in ['source', 'target', 'relationship']
                }
                
                self.scene_graph.add_edge(source, target, relationship, **edge_attrs)
                
            except Exception as e:
                logger.error(f"Error processing edge: {str(e)}")
                continue
        #self.query_engine = QueryEngine(self.scene_graph)
        self.query_engine = QueryEngine(self.scene_graph, controller=self)
        self.scene_graph.print_summary()
        return True
        
    def build_scene_nodes(self, targets, surroundings, show=True, time_window=3):
        for object in targets:
            self.scene_graph.add_object_node(object[0], object[1], object[2], self.timestep, target=True)
        for object in surroundings:
            self.scene_graph.add_object_node(object[0], object[1], object[2], self.timestep)

        # Query scenary context
        #self.query_engine = QueryEngine(self.scene_graph)
        self.query_engine = QueryEngine(self.scene_graph, controller=self)
        current_pos = self.pose.xy
        context = self.query_engine.get_context(self.pose.xy, radius=30.0)

        surrounding = ''
        recent_objects = self.query_engine.get_recent_objects(current_timestamp=self.timestep, time_window=time_window)
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
    
    def robust_json_loads(self, text: str):
        """
        尝试从任意文本中提取并修复 JSON 内容。
        支持：
            - Markdown code block (```json ... ```)
            - 单引号转双引号
            - 补全缺失的括号（启发式）
            - 移除尾随逗号
        返回解析后的 Python 对象，或抛出 ValueError。
        """
        # 1. 尝试直接解析（最快路径）
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 2. 提取 ```json ... ``` 或 ``` ... ``` 中的内容
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if code_block_match:
            text = code_block_match.group(1).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        # 3. 尝试找到最外层的 [...] 或 {...} 结构（贪心匹配）
        # 先找数组
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        # 再找对象（优先级低于数组，因为你的例子是 list of dict）
        obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        candidate = None
        if array_match:
            candidate = array_match.group(0)
        elif obj_match:
            candidate = obj_match.group(0)
        if candidate:
            text = candidate
        # 4. 修复常见问题
        # 替换单引号为双引号（但保留字符串内的单引号）
        # 注意：不能简单全局替换，会破坏字符串内容
        # 更安全的方式：用 ast.literal_eval 如果结构简单，否则用正则+json5
        try:
            # 先尝试用 ast（支持单引号、True/False 等）
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass
        # 5. 手动修复：替换明显错误
        original = text
        # 移除尾随逗号（在 ] 或 } 前）
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        # 替换中文标点（如果存在）
        text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        # 尝试补全缺失的括号（仅限简单情况）
        brace_diff = text.count('{') - text.count('}')
        bracket_diff = text.count('[') - text.count(']')
        if brace_diff > 0:
            text += '}' * brace_diff
        elif brace_diff < 0:
            text = '{' * (-brace_diff) + text
        if bracket_diff > 0:
            text += ']' * bracket_diff
        elif bracket_diff < 0:
            text = '[' * (-bracket_diff) + text
        # 6. 再次尝试解析
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # 最后手段：打印原始内容供调试
            raise ValueError(f"Failed to parse JSON after robust recovery. Original text:\n{text[:500]}...\nError: {e}")
    
    def query_scene_graph(self, instruction: str, debug=False):
        """robust_subgraph_query"""
        from scenegraphnav.prompt.geonav_cot import QUERY_OPERATION_CHAIN_PROMPT
        import json
        # 使用LLM生成操作链
        prompt = QUERY_OPERATION_CHAIN_PROMPT.format(instruction=instruction)
        operation_chain = None
        raw_content = None
        for attempt in range(2):  # 尝试2次
            try:
                response = self.llm_client.chat.completions.create(
                    model= LLM_NAME, #"gpt-4-turbo"
                    response_format={"type": "json_object"},
                    max_tokens=2048,
                    messages=[
                        {"role": "system", "content": "You are a professional query planner."},
                        {"role": "user", "content": prompt}
                    ]
                )
                raw_content = response.choices[0].message.content
                operation_chain = self.robust_json_loads(raw_content)
                if debug:
                    print(f"Generated operation chain {json.dumps(operation_chain, indent=2, ensure_ascii=False)}")
                break  # 成功则退出循环
            except Exception as e:
                if attempt == 0:
                    print(f"First attempt failed: {str(e)}, retrying...")
                    continue
                else:
                    print(f"Second attempt failed: {str(e)}")
                    return [], None
        else:
            # 如果循环正常结束但没有成功（不应该发生）
            return [], None
        if operation_chain is None:
            return [], None
            # 使用robust_subgraph_query执行查询
        results, query_logs = self.query_engine.robust_subgraph_query(
            operation_chain, fallback=True, min_results=1, debug=debug
        )
        # 添加操作链信息到日志
        query_info = {
            "instruction": instruction,
            "operation_chain": operation_chain,
            "step_logs": query_logs
        }
        return results, query_info
    
    # 复杂查询分解与关系验证
    def complex_query_scene_graph(self, instruction: str, debug=False):
        """break complex queries down into multiple sub-queries"""
        query_logs = []  # 收集所有子查询的日志
        # break down
        sub_queries = self.decompose_complex_query(instruction, debug)
        if not sub_queries:
            # decomposition failed
            if debug:
                print("Decompose complex query is failed. Use a regular query.")
            results, query_info = self.query_scene_graph(instruction, debug)
            if query_info:
                query_logs.append(query_info)
            return results, query_logs
        sub_results = []
        for i, sub_query in enumerate(sub_queries):
            if debug:
                print(f"subquery {i+1}: {sub_query}")
            
            try:
                results, query_info = self.query_scene_graph(sub_query, debug)
                if query_info:
                    query_logs.append(query_info)
                sub_results.append((sub_query, results))
            except Exception as e:
                print(f"Error in subquery {i+1}: {e}")
                sub_results.append((sub_query, []))
            
            if debug:
                result_len = len(sub_results[-1][1]) if sub_results[-1][1] else 0
                print(f"subquery {i+1} return {result_len} results")
        
        # 验证关系约束
        try:
            final_results, verify_log = self.verify_relation_constraints(instruction, sub_results, debug)
            if verify_log:
                query_logs.append({"verify_relation_constraints": verify_log})
        except Exception as e:
            print(f"Error in verify_relation_constraints: {e}")
            # 返回第一个非空结果
            for _, results in sub_results:
                if results:
                    final_results = results
                    break
            else:
                final_results = []
        
        return final_results, query_logs
    
    def decompose_complex_query(self, instruction: str, debug=False):
        """Decompose Complex Queries into Multiple Simple Queries"""
        prompt = f"""
        Decompose the following complex instructions into multiple simple query steps:
        instruction: "{instruction}"
        
        Each step should focus on a specific object and its characteristics. For example, the instruction "There is a gray car in front of the white car and another white car behind it" can be decomposed as:
        1. Look for white cars
        2. Find the grey car in front of the white one
        3. Look for another white car behind the first one
        Please output the decomposed list of simple queries. Each query should be a complete sentence.
        """
        
        response = self.llm_client.chat.completions.create(
            model= LLM_NAME, #"gpt-4-turbo",
            max_tokens=2048,
            messages=[
                {"role": "system", "content": "You are a professional query analyzer."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        
        # 提取子查询列表
        sub_queries = []
        # 匹配格式为 "1. xxx" 或 "1) xxx" 或 "- xxx" 的文本行
        pattern = r'(?:\d+[\.\)]\s+|\-\s+)(.+)'
        matches = re.findall(pattern, result)
        
        if matches:
            sub_queries = [match.strip() for match in matches if match.strip()]
        
        if debug:
            print(f"Decompose Query Result: {sub_queries}")
        
        return sub_queries
    
    def decompose_spatial_relation(self, instruction: str, target: str, debug=False):
        """Decompose Complex Queries into Multiple Simple Queries"""
        prompt = f"""
        Your goal is to find the target {target}, and the description of it is {instruction}.
        You need to verify the spatial relationship between the surrounding objects and the target. 
        Decompose complex description into multiple simple query steps.
        
        Pay attention to the spatial relationships between the surrounding objects and the target, select the most important spatial relationships for verification, and do not exceed five.
        Each step should focus on the spatial relationship between a specific object with its characteristics and the target. Use "it" to refer to the target. 
        For example, the description "One black car in between a dark blue car and a gray car that also has a blue car parts behind it off Woodvale Lodge facing a Grassfield" can be decomposed as:
        1. A dark blue car is on one side of it
        2. A gray car is on the other side of it
        3. A blue car is behind it
        Please output the decomposed list of simple queries. Each query should be a complete sentence.
        """

        for attempt in range(2):
            try:
                response = self.llm_client.chat.completions.create(
                    model=LLM_NAME,
                    max_tokens=2048,
                    messages=[
                        {"role": "system", "content": "You are a professional query analyzer."},
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.choices[0].message.content
                break
            except Exception as e:
                if attempt == 0:
                    print(f"First attempt failed: {str(e)}, retrying...")
                    continue
                else:
                    print(f"Second attempt failed: {str(e)}")
                    return []

        # 提取子查询列表
        sub_queries = []
        # 匹配格式为 "1. xxx" 或 "1) xxx" 或 "- xxx" 的文本行
        pattern = r'(?:\d+[\.\)]\s+|\-\s+)(.+)'
        matches = re.findall(pattern, result)
        if matches:
            sub_queries = [match.strip() for match in matches if match.strip()]
        if debug:
            print(f"Decompose Query Result: {sub_queries}")

        return sub_queries
    
    def verify_relation_constraints(self, original_instruction, sub_results, debug=False):
        """Verify multiple query results based on relation constraints
        Returns:
            (best_nodes, log_entry): 结果节点列表，以及包含 LLM 响应的日志字典
        """
        log_entry = {
            "original_instruction": original_instruction,
            "sub_results_count": len(sub_results) if sub_results else 0,
            "sub_results_desc": [],
            "llm_response": None,
            "parsed_result": None,
            "selected_node_ids": []
        }

        if not sub_results or any(not results for _, results in sub_results):
            if debug:
                print("Some subquery results are empty, unable to construct a complete operation chain. Return the first non-null result.")
            # 构建子结果描述（用于日志）
            for query, nodes in (sub_results or []):
                log_entry["sub_results_desc"].append({
                    "query": query,
                    "node_count": len(nodes),
                    "node_ids": [n.id for n in nodes[:5]]
                })
            # 返回第一个非空结果，或空列表
            for _, results in sub_results:
                if results:
                    log_entry["selected_node_ids"] = [n.id for n in results]
                    return results, log_entry
            return [], log_entry

        # 构建结果描述
        results_desc = []
        for i, (query, nodes) in enumerate(sub_results):
            node_ids_sample = [n.id for n in nodes[:5]]
            log_entry["sub_results_desc"].append({
                "query": query,
                "node_count": len(nodes),
                "node_ids": node_ids_sample
            })
            nodes_desc = []
            for j, node in enumerate(nodes[:5]):  # 限制每个查询最多描述5个结果
                node_desc = f"Node{j+1} (ID: {node.id}, "
                if hasattr(node, 'obj_class'):
                    node_desc += f"Class: {node.obj_class}, "
                if hasattr(node, 'color'):
                    node_desc += f"color: {node.color}, "
                node_desc += f"Position: ({node.position.x:.1f}, {node.position.y:.1f}))"
                nodes_desc.append(node_desc)

            results_desc.append(f"Query {i+1}: {query}\nResult: {'; '.join(nodes_desc)}")

        joined_results = "\n\n".join(results_desc)
        prompt = f"""
        Analyze the following multiple query results to determine which object combinations best meet the requirements of the original instruction.
        Original instruction: "{original_instruction}"

        Query Result:
        {joined_results}

        Determine which object combinations are most likely to satisfy all the relational constraints of the original instruction and provide reasons.
        Return JSON format:
        {{
            "best_match": [
                {{"query_index": query_index, "node_index": node_index}},
                ...
            ],
            "reason": "reason for the best match"
        }}

        If there is no exactly matching, return the closest combination
        """

        response = self.llm_client.chat.completions.create(
            model= LLM_NAME, #"gpt-4-turbo",
            response_format={"type": "json_object"},
            max_tokens=2048,
            messages=[
                {"role": "system", "content": "You are a professional spatial relation analyzer."},
                {"role": "user", "content": prompt}
            ]
        )

        raw_response = response.choices[0].message.content
        log_entry["llm_response"] = raw_response

        try:
            result = json.loads(raw_response)
            log_entry["parsed_result"] = result

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
                    log_entry["selected_node_ids"] = [n.id for n in best_nodes]
                    return best_nodes, log_entry

        except Exception as e:
            if debug:
                print(f"关系验证错误: {str(e)}")

        # 如果无法确定最佳组合，返回第一个查询的结果
        if debug:
            print("无法确定最佳组合，返回第一个查询的结果")
        fallback = sub_results[0][1] if sub_results else []
        log_entry["selected_node_ids"] = [n.id for n in fallback]
        return fallback, log_entry
    
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
        """Generate consistent spatial relationship descriptions"""
        dx = child.position.x - parent.position.x
        dy = child.position.y - parent.position.y
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360  # 确保角度为正值
        
        relations = {
            (337.5, 22.5): "north_of",      # 正北
            (22.5, 67.5): "northeast_of",   # 东北
            (67.5, 112.5): "east_of",       # 正东
            (112.5, 157.5): "southeast_of", # 东南
            (157.5, 202.5): "south_of",     # 正南
            (202.5, 247.5): "southwest_of", # 西南
            (247.5, 292.5): "west_of",      # 正西
            (292.5, 337.5): "northwest_of"  # 西北
        }
        # relations = {
        #     (337.5, 22.5): "directly in front of",
        #     (22.5, 67.5): "to the top right of",
        #     (67.5, 112.5): "to the right of",
        #     (112.5, 157.5): "to the bottom right of",
        #     (157.5, 202.5): "directly behind",
        #     (202.5, 247.5): "to the bottom left of",
        #     (247.5, 292.5): "to the left of",
        #     (292.5, 337.5): "to the top left of"
        # }
        for (start, end), desc in relations.items():
            if start <= angle < end:
                return desc
        return "adjacent_to"  # 默认关系保持不变
    