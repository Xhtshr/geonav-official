from utils.tools import get_direction
import math
import networkx as nx
import numpy as np
from collections import defaultdict
from rtree import index  # pip install rtree

import sys
sys.path.append("/data1/XHT/citynav/")
from gsamllavanav.space import Point2D
from gsamllavanav.cityreferobject import CityReferObject
from shapely.geometry import Point
from utils.text_cosine import semantic_similarity

from scenegraphnav.prompt.config import LLM_NAME
from scenegraphnav.prompt.geonav_cot import QUERY_OPERATION_SUBCHAIN_PROMPT


# ================== Knowledge Graph ==================
class KnowledgeNode:
    def __init__(self, node_id, node_type, position, timestamp):
        self.id = node_id
        self.type = node_type
        self.position = position
        self.timestamp = timestamp

class GeoNode(KnowledgeNode):
    def __init__(self, city_obj: CityReferObject):
        super().__init__(
            node_id=f"{city_obj.object_type}_{city_obj.name}",
            node_type='geo',
            position=city_obj.position,
            timestamp=-1  # 地理数据永不过期
        )
        self.name = city_obj.name
        self.object_type = city_obj.object_type
        self.contour_polygon = city_obj.contour_polygon
        self.child_objects = []  # 子对象ID列表

class ObjectNode(KnowledgeNode):
    def __init__(self, detection_id, position, obj_class, confidence, timestamp=None, target=False, **attrs):
        super().__init__(
            node_id=detection_id,
            node_type='object',
            position=position,
            timestamp=timestamp
        )
        self.obj_class = obj_class
        self.confidence = confidence
        self.target = target
        self.attrs = attrs
        
        # 添加其他属性
        for key, value in attrs.items():
            setattr(self, key, value)

# ================== Spatial Index ==================
class SpatialIndex:
    def __init__(self):
        self.idx = index.Index()
        self.nodes = {}
    
    def insert(self, node: KnowledgeNode):
        # 使用(x_min, y_min, x_max, y_max)作为边界框
        hashed_id = hash(node.id)  # 哈希值作为 R-tree 的 ID
        self.idx.insert(id=hashed_id, 
                       coordinates=(node.position.x, node.position.y, 
                                     node.position.x, node.position.y))
        self.nodes[hashed_id] = node
    
    def query_radius(self, center: Point2D, radius: float):
        candidates = self.idx.intersection(
            (center.x - radius, center.y - radius,
             center.x + radius, center.y + radius))
        return [self.nodes[n] for n in candidates]

class KnowledgeGraph:
    def __init__(self):
        self.spatial_index = SpatialIndex()
        self.nodes = {}
        self.temp_nodes = {}
        self.class_counters = defaultdict(int)
        
        # NetworkX
        self.graph_nx = nx.Graph()
        self.digraph_nx = nx.DiGraph()
        self.semantic_similarity = semantic_similarity()
        self.alpha = 0.95  # 空间和语义相似度的权重
    def update_nodes(self, current_timestamp: int, decay_rate=0.1):
        """
        更新节点置信度并清理过期节点
        Args:
            current_timestamp: 当前时间步
            decay_rate: 每个时间步的置信度衰减率(默认0.1)
        """
        to_remove = []
        
        for node_id, node in self.nodes.items():
            if isinstance(node, ObjectNode):
                # 计算存活时间步数
                time_alive = current_timestamp - node.timestamp
                
                # 指数衰减公式：confidence * (1 - decay_rate)^t
                node.confidence *= (1 - decay_rate) ** time_alive
                
                # 标记需要移除的节点
                if node.confidence < 0.05:  # 设置最小阈值
                    to_remove.append(node_id)
        
        # 移除低置信度节点
        for node_id in to_remove:
            node = self.nodes.pop(node_id)
            hashed_id = hash(node_id)
            self.spatial_index.idx.delete(hashed_id, None)
            del self.spatial_index.nodes[hashed_id]
    
    def add_edge(self, source_id: str, target_id: str, relation_type: str, **attrs):
        """添加带有属性的边（支持同节点对多条边）"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Cannot create edge between non-existing nodes")
        # 检查是否已存在边
        if self.graph_nx.has_edge(source_id, target_id):
            # 如果边已存在，将新关系添加到列表中
            existing_relations = self.graph_nx[source_id][target_id].get('relations', [])
            if isinstance(existing_relations, str):
                existing_relations = [existing_relations]
            #对边的关系进行去重
            if relation_type not in existing_relations:
                existing_relations.append(relation_type)
            self.graph_nx[source_id][target_id]['relations'] = existing_relations
            self.digraph_nx[source_id][target_id]['relations'] = existing_relations
        else:
            # 新建边
            self.graph_nx.add_edge(source_id, target_id, relations=[relation_type])
            self.digraph_nx.add_edge(source_id, target_id, relations=[relation_type])
        
        # 添加其他属性
        for key, value in attrs.items():
            self.graph_nx[source_id][target_id][key] = value
            self.digraph_nx[source_id][target_id][key] = value

    def add_geo_node(self, city_obj: CityReferObject):
        node = GeoNode(city_obj)
        self._add_node(node, 'geo')
    
    def add_object_node_with_attrs(self, position, attrs):
        """添加带有多个属性的对象节点"""
        obj_class = attrs.pop('obj_class')
        confidence = attrs.pop('confidence')
        timestamp = attrs.pop('timestamp')
        target = attrs.pop('target', False)
        
        # 基本添加节点
        current_count = self.class_counters[obj_class]  # 获取当前计数
        node_id = f"{obj_class}_{current_count}"
        node = ObjectNode(node_id, position, obj_class, confidence, timestamp, target, **attrs)
        Flag = self._add_node_v2(node, obj_class)
        if Flag:
            return Flag
        return node_id

    def _add_node(self, node, obj_class):
        # 简单去重策略：相同位置同类型视为同一对象
        existing = self.spatial_index.query_radius(node.position, 20.0)
        for n in existing:
            if n.type == node.type and self._distance(n.position, node.position) < 5.0:
                return n.id
        self.graph_nx.add_node(node.id)
        self.digraph_nx.add_node(node.id)
        self.class_counters[obj_class] += 1
        self.nodes[node.id] = node
        self.spatial_index.insert(node)
    
    def _add_node_v2(self, node, obj_class, query_radius=20.0, similarity_threshold=0.8):
        # 使用相似性度量来判断是否为同一对象
        self.spatial_threshold = query_radius / 5.0
        existing = self.spatial_index.query_radius(node.position, query_radius)
        # 遍历邻近节点，寻找可合并的候选
        merge_candidate = None
        max_score = -1
        for n in existing:
            current_score = self.alpha * self._calculate_spatial_similarity(n.position, node.position) \
                + (1 - self.alpha) * self._calculate_semantic_similarity(n.type, node.type)
            if current_score > max_score and current_score >= similarity_threshold:
                max_score = current_score
                merge_candidate = n
        if merge_candidate:
            return merge_candidate.id
        self.graph_nx.add_node(node.id)
        self.digraph_nx.add_node(node.id)
        self.class_counters[obj_class] += 1
        self.nodes[node.id] = node
        self.spatial_index.insert(node)
    
    def _calculate_spatial_similarity(self, p1: Point2D, p2: Point2D):
        distance = self._distance(p1, p2)
        return np.exp(-distance / self.spatial_threshold)
    
    def _calculate_semantic_similarity(self, type1, type2):
        """计算语义相似性（类别相同为1，否则可扩展为类别间相似度）"""
        return 1.0 if type1 == type2 else self.semantic_similarity.sentence_similarity(type1, type2)

    @staticmethod
    def _distance(p1: Point2D, p2: Point2D):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def format_summary(self, sample_geo=None, sample_object=None, sample_edge=None, as_dict=True):
        """
        输出 KnowledgeGraph 的摘要信息，便于调试或在其他地方调用。
        参数：
            sample_geo: GeoNode 展示数量上限，None 表示全部
            sample_object: ObjectNode 展示数量上限，None 表示全部
            sample_edge: Edge 展示数量上限，None 表示全部
            as_dict: 是否返回字典格式（用于JSON保存）
        返回：
            str 或 dict: 格式化的摘要字符串或字典
        """
        # 节点统计
        geo_nodes = [n for n in self.nodes.values() if isinstance(n, GeoNode)]
        object_nodes = [n for n in self.nodes.values() if isinstance(n, ObjectNode)]
        n_geo, n_obj = len(geo_nodes), len(object_nodes)
        n_total = len(self.nodes)
        # 边统计
        n_undirected = self.graph_nx.number_of_edges()
        n_directed = self.digraph_nx.number_of_edges()
        # 按类别统计 object
        by_class = defaultdict(int)
        n_target = 0
        for n in object_nodes:
            by_class[n.obj_class] += 1
            if getattr(n, 'target', False):
                n_target += 1
        # 关系类型统计（支持新旧格式）
        rel_count = defaultdict(int)
        for u, v, data in self.digraph_nx.edges(data=True):
            relations = data.get('relations', data.get('relation_type', 'related_to'))
            if isinstance(relations, list):
                for r in relations:
                    rel_count[r] += 1
            else:
                rel_count[relations] += 1
        if as_dict:
            # 返回字典格式（用于JSON保存）
            summary = {
                "nodes": {"total": n_total, "geo": n_geo, "object": n_obj},
                "edge_status": {"undirected": n_undirected, "directed": n_directed},
                "objects_by_class": dict(by_class),
                "target_object_count": n_target,
                "relations": dict(rel_count),
                "geo_nodes": [],
                "object_nodes": [],
                "edges": []
            }
            # 添加 GeoNodes
            for i, node in enumerate(geo_nodes):
                if sample_geo is not None and i >= sample_geo:
                    break
                name = getattr(node, 'name', node.id)
                otype = getattr(node, 'object_type', 'unknown')
                summary["geo_nodes"].append({
                    "id": node.id,
                    "name": name,
                    "type": otype,
                    "position": {"x": round(node.position.x, 1), "y": round(node.position.y, 1)}
                })
            # 添加 ObjectNodes
            for i, node in enumerate(object_nodes):
                if sample_object is not None and i >= sample_object:
                    break
                summary["object_nodes"].append({
                    "id": node.id,
                    "class": node.obj_class,
                    "timestamp": getattr(node, 'timestamp', '-'),
                    "position": {"x": round(node.position.x, 1), "y": round(node.position.y, 1)},
                    "attributes": node.attrs
                })
            # 添加 Edges
            edges_list = list(self.digraph_nx.edges(data=True))
            for i, (u, v, data) in enumerate(edges_list):
                if sample_edge is not None and i >= sample_edge:
                    break
                relations = data.get('relations', data.get('relation_type', 'related_to'))
                if isinstance(relations, list):
                    relations_str = ', '.join(relations)
                else:
                    relations_str = relations
                summary["edges"].append({
                    "relation": relations_str,
                    "from": u,
                    "to": v
                })
            return summary
        # 返回字符串格式（用于控制台输出）
        lines = []
        # 节点统计
        geo_nodes = [n for n in self.nodes.values() if isinstance(n, GeoNode)]
        object_nodes = [n for n in self.nodes.values() if isinstance(n, ObjectNode)]
        n_geo, n_obj = len(geo_nodes), len(object_nodes)
        n_total = len(self.nodes)
        # 边统计
        n_undirected = self.graph_nx.number_of_edges()
        n_directed = self.digraph_nx.number_of_edges()
        # 按类别统计 object
        by_class = defaultdict(int)
        n_target = 0
        for n in object_nodes:
            by_class[n.obj_class] += 1
            if getattr(n, 'target', False):
                n_target += 1
        # 关系类型统计（支持新旧格式）
        rel_count = defaultdict(int)
        for u, v, data in self.digraph_nx.edges(data=True):
            relations = data.get('relations', data.get('relation_type', 'related_to'))
            if isinstance(relations, list):
                for r in relations:
                    rel_count[r] += 1
            else:
                rel_count[relations] += 1
        # ----- Summary -----
        lines.append("# ----- Scene Graph: KnowledgeGraph Summary ----- #")
        lines.append(f"--Nodes: {n_total} (geo={n_geo}, object={n_obj})")
        lines.append(f"--Edges: {n_undirected} undirected, {n_directed} directed")
        lines.append(f"--Objects (by class): {dict(by_class)}")
        lines.append(f"--target_object: {n_target}")
        lines.append("--Relations:")
        for rel, cnt in sorted(rel_count.items(), key=lambda x: -x[1]):
            lines.append(f"  --{rel}: {cnt}")
        # ----- GeoNodes -----
        lines.append("")
        lines.append(f"GeoNodes (sample {min(len(geo_nodes), sample_geo or len(geo_nodes))}/{len(geo_nodes)})")
        for i, node in enumerate(geo_nodes):
            if sample_geo is not None and i >= sample_geo:
                break
            name = getattr(node, 'name', node.id)
            otype = getattr(node, 'object_type', 'unknown')
            x, y = node.position.x, node.position.y
            lines.append(f"--{node.id}, geo_name='{name}', type='{otype}', @({x:.1f},{y:.1f})")
        # ----- ObjectNodes -----
        lines.append("")
        lines.append(f"ObjectNodes (sample {min(len(object_nodes), sample_object or len(object_nodes))}/{len(object_nodes)})")
        for i, node in enumerate(object_nodes):
            if sample_object is not None and i >= sample_object:
                break
            conf = getattr(node, 'confidence', 0)
            ts = getattr(node, 'timestamp', '-')
            x, y = node.position.x, node.position.y
            lines.append(f"--{node.id}, obj class='{node.obj_class}', conf={conf:.3f}, ts={ts}, @({x:.1f},{y:.1f})")
        # ----- Edges -----
        edges_list = list(self.digraph_nx.edges(data=True))
        n_show = min(len(edges_list), sample_edge or len(edges_list))
        lines.append("")
        lines.append(f"Edges (sample {n_show}/{len(edges_list)})")
        for i, (u, v, data) in enumerate(edges_list):
            if sample_edge is not None and i >= sample_edge:
                break
            relations = data.get('relations', data.get('relation_type', 'related_to'))
            if isinstance(relations, list):
                rt_str = ', '.join(relations)
            else:
                rt_str = relations
            lines.append(f"--{rt_str} : {u} --> {v}")
        return "\n".join(lines)

    def print_summary(self, sample_geo=None, sample_object=None, sample_edge=None):
        """打印 KnowledgeGraph 摘要到标准输出"""
        print(self.format_summary(sample_geo, sample_object, sample_edge))

# ================== 查询引擎 ==================
class QueryEngine:
    '''原版方法
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph'''
    
    def __init__(self, graph: KnowledgeGraph, controller=None):
        self.graph = graph
        self.controller = controller
        
    def get_context(self, position: Point2D, radius=10.0):
        nodes = self.graph.spatial_index.query_radius(position, radius)
        return [self._describe_node(n, position) for n in nodes]
    
    def _describe_node(self, node, query_pos):
        desc = {
            'type': node.type,
            'distance': f"{self._calc_distance(node.position, query_pos):.1f} meters",
            'direction': get_direction(query_pos, node.position)
        }
        if isinstance(node, GeoNode):
            desc['name'] = node.name
            desc['category'] = node.object_type
        elif isinstance(node, ObjectNode):
            desc['id'] = node.id
            desc['class'] = node.obj_class
            desc['confidence'] = node.confidence
            desc['target'] = node.target
        return desc
    
# ===============dummy 查询 ==================
    def get_recent_objects(self, current_timestamp: int, time_window: int):
        """获取指定时间窗口内的动态物体信息
        Args:
            current_timestamp: 当前仿真时间戳（0-20的整数）
            time_window: 要查询的时间窗口长度（正整数）
        """
        threshold = max(0, current_timestamp - time_window)
        
        recent = []
        for node in self.graph.nodes.values():
            if isinstance(node, ObjectNode) and node.timestamp is not None:
                if threshold <= node.timestamp <= current_timestamp:
                    recent.append({
                        'id': f"{node.id}",
                        'type': node.obj_class,
                        'pos': (node.position.x, node.position.y),
                        'age': current_timestamp - node.timestamp
                    })
        
        if not recent:
            return f"No objects has been detected in the past {time_window}"
            
        desc = [f"Detected {len(recent)} objects in the past {time_window} timesteps:"]
        for obj in sorted(recent, key=lambda x: x['age']):
            desc.append(
                f"{obj['type']} (ID:{obj['id']}) at ({obj['pos'][0]:.1f}, {obj['pos'][1]:.1f})"
                f", appeared {obj['age']} steps ago"
            )
        
        return '\n'.join(desc)
    
# ================== 查询原子操作 ==================
    def subgraph_query(self, operation_chain: list):
        """链式查询框架"""
        current_nodes = None
        for op in operation_chain:
            method_name = op['method']
            func = getattr(self, method_name)
            
            # 检查是否需要 parent_node 参数
            if method_name == 'get_child_nodes':
                if current_nodes is None or len(current_nodes) == 0:
                    raise ValueError("No parent node available for get_child_nodes operation")
                # 假设使用第一个节点作为 parent_node
                #parent_node = current_nodes[0]
                # 传递所有父节点，而不是只取第一个
                parent_node = current_nodes
                current_nodes = func(parent_node, **op.get('kwargs', {}), candidates=current_nodes)
            elif method_name == 'get_geonode_by_name':
                # 特殊处理 get_geonode_by_name，当 args 为空时使用所有地理节点
                args = op.get('args', [])
                if not args or (len(args) == 1 and args[0] == ""):
                    # 如果没有提供名称模式，则返回所有地理节点
                    current_nodes = [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]
                else:
                    current_nodes = func(*args, candidates=current_nodes)
            elif method_name in ['filter_by_class', 'filter_by_attribute']:
                # 确保 current_nodes 不为 None
                if current_nodes is None:
                    current_nodes = list(self.graph.nodes.values())
                current_nodes = func(*op.get('args', []), **op.get('kwargs', {}), candidates=current_nodes)
            else:
                # 其他方法
                current_nodes = func(*op.get('args', []), **op.get('kwargs', {}), candidates=current_nodes)
            
            if not current_nodes:
                break
        return current_nodes
    
    def robust_subgraph_query(self, operation_chain: list, fallback=True, min_results=1, debug=False):
        """增强版链式查询框架，具有容错和回退机制
        
        Args:
            operation_chain: 查询操作链
            fallback: 是否启用回退机制
            min_results: 最小结果数量，少于此数量将触发回退
            debug: 是否输出调试信息
            
        Returns:
            查询结果节点列表
        """
        if not operation_chain:
            return []
            
        current_nodes = None
        all_results = []  # 存储每步查询的结果
        fallback_modes = []  # 存储每步是否使用了回退模式
        
        for i, op in enumerate(operation_chain):
            method_name = op['method']
            step_name = f"步骤{i+1}: {method_name}"
            
            try:
                # 尝试获取方法
                if not hasattr(self, method_name):
                    if debug:
                        print(f"警告: 方法 '{method_name}' 不存在，跳过此步骤")
                    fallback_modes.append(True)
                    continue
                    
                func = getattr(self, method_name)
                
                # 执行查询，根据方法类型调整参数
                fallback_step = False
                
                if method_name == 'get_child_nodes':
                    if current_nodes is None or len(current_nodes) == 0:
                        if not fallback:
                            if debug:
                                print(f"{step_name}: 没有可用的父节点，无法继续查询")
                            break
                        else:
                            # 回退：获取所有地理节点
                            if debug:
                                print(f"{step_name}: 没有可用的父节点，回退到所有地理节点")
                            current_nodes = [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]
                            fallback_step = True
                    
                    if not fallback_step:
                        # 使用第一个节点作为父节点
                        parent_node = current_nodes[0]
                        temp_nodes = func(parent_node, **op.get('kwargs', {}), candidates=current_nodes)
                        
                        # 检查结果是否为空
                        if not temp_nodes or len(temp_nodes) < min_results:
                            if fallback and i > 0:
                                if debug:
                                    print(f"{step_name}: The number of results is insufficient ({len(temp_nodes) if temp_nodes else 0}). Keep the current results.")
                                fallback_step = True
                            else:
                                current_nodes = temp_nodes
                        else:
                            current_nodes = temp_nodes
                
                elif method_name == 'get_geonode_by_name':
                    args = op.get('args', [])
                    if not args or (len(args) == 1 and not args[0]):
                        # 如果没有提供名称模式，则返回所有地理节点
                        current_nodes = [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]
                    else:
                        temp_nodes = func(*args, candidates=current_nodes)
                        
                        # 检查结果是否为空
                        if not temp_nodes or len(temp_nodes) < min_results:
                            if fallback:
                                # 返回所有地理节点
                                if debug:
                                    print(f"{step_name}: No matching geographical nodes were found. Retrying with all geographical nodes.")
                                current_nodes = [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]
                                fallback_step = True
                            else:
                                current_nodes = temp_nodes
                        else:
                            current_nodes = temp_nodes
                
                elif method_name in ['filter_by_class', 'filter_by_attribute']:
                    # 确保 current_nodes 不为 None
                    if current_nodes is None:
                        current_nodes = list(self.graph.nodes.values())
                    
                    temp_nodes = func(*op.get('args', []), **op.get('kwargs', {}), candidates=current_nodes)
                    
                    # 检查结果是否为空
                    if not temp_nodes or len(temp_nodes) < min_results:
                        if fallback:
                            if debug:
                                print(f"{step_name}: filtered result is empty or insufficient. Maintain the current result.")
                            # 保持当前结果不变
                            fallback_step = True
                        else:
                            current_nodes = temp_nodes
                    else:
                        current_nodes = temp_nodes
                
                else:
                    # 其他方法
                    temp_nodes = func(*op.get('args', []), **op.get('kwargs', {}), candidates=current_nodes)
                    
                    # 检查结果是否为空
                    if not temp_nodes or len(temp_nodes) < min_results:
                        if fallback and i > 0:
                            if debug:
                                print(f"{step_name}: the result is empty or insufficient, maintain the current result.")
                            # 保持当前结果不变
                            fallback_step = True
                        else:
                            current_nodes = temp_nodes
                    else:
                        current_nodes = temp_nodes
                
                # 保存当前步骤的结果状态
                all_results.append(current_nodes)
                fallback_modes.append(fallback_step)
                
                if debug:
                    print(f"{step_name}: Find {len(current_nodes) if current_nodes else 0} result nodes" +
                         (f"Backtracking mode" if fallback_step else ""))
            
            except Exception as e:
                if debug:
                    print(f"{step_name}: 发生错误 - {str(e)}")
                
                if fallback:
                    # 错误发生时，保持当前结果不变
                    if i > 0 and all_results:
                        if debug:
                            print(f"{step_name}error occurred. Keep the result of the previous step.")
                        fallback_step = True
                    else:
                        # 如果是第一步，或者没有前一步结果，使用所有节点
                        if debug:
                            print(f"{step_name} error occurred. Use all nodes.")
                        current_nodes = list(self.graph.nodes.values())
                        fallback_step = True
                    
                    fallback_modes.append(fallback_step)
                    all_results.append(current_nodes)
                else:
                    # 不使用回退机制，直接中断查询
                    break
        
        # 查询完成后，如果结果为空且允许回退
        if (not current_nodes or len(current_nodes) < min_results) and fallback and all_results:
            # 找到最后一个非回退步骤的结果
            for i in range(len(all_results)-1, -1, -1):
                if not fallback_modes[i] and all_results[i] and len(all_results[i]) >= min_results:
                    current_nodes = all_results[i]
                    if debug:
                        print(f"final result is empty or insufficient, revert to the result of step {i+1}. There are {len(current_nodes)} nodes in total.")
                    break
        
        if debug:
            print(f"Query complete. {len(current_nodes) if current_nodes else 0} result nodes are returned.")
        
        return current_nodes or []
    
    # def robust_subgraph_query(self, operation_chain: list, fallback=True, min_results=1, debug=False):
    #     """增强版链式查询框架，具有容错和回退机制
    #     Args:
    #         operation_chain: 查询操作链
    #         fallback: 是否启用回退机制
    #         min_results: 最小结果数量，少于此数量将触发回退
    #         debug: 是否输出调试信息
    #     Returns:
    #         查询结果节点列表, 查询日志列表
    #     """
    #     if not operation_chain:
    #         return [], []
            
    #     current_nodes = None
    #     all_results = []  # 存储每步查询的结果
    #     fallback_modes = []  # 存储每步是否使用了回退模式
    #     query_logs = []  # 存储查询日志
        
    #     for i, op in enumerate(operation_chain):
    #         method_name = op['method']
    #         step_name = f"步骤{i+1}: {method_name}"
    #         # 记录日志
    #         log_entry = {
    #             "step": i + 1,
    #             "method": method_name,
    #             "args": op.get('args', []),
    #             "kwargs": op.get('kwargs', {}),
    #             "parent_nodes_count": len(current_nodes) if current_nodes else 0,
    #             "fallback": False,
    #             "result_count": 0,
    #             "result_ids": []
    #         }
    #         try:
    #             # 尝试获取方法
    #             if not hasattr(self, method_name):
    #                 if debug:
    #                     print(f"警告: 方法 '{method_name}' 不存在，跳过此步骤")
    #                 fallback_modes.append(True)
    #                 continue
    #             func = getattr(self, method_name)
    #             # 执行查询，根据方法类型调整参数
    #             fallback_step = False
    #             if method_name == 'get_child_nodes':
    #                 if current_nodes is None or len(current_nodes) == 0:
    #                     if not fallback:
    #                         if debug:
    #                             print(f"{step_name}: 没有可用的父节点，无法继续查询")
    #                         break
    #                     else:
    #                         # 回退：获取所有地理节点
    #                         if debug:
    #                             print(f"{step_name}: 没有可用的父节点，回退到所有地理节点")
    #                         current_nodes = [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]
    #                         fallback_step = True
                    
    #                 if not fallback_step:
    #                     # 传递所有父节点，而不是只取第一个
    #                     parent_node = current_nodes
    #                     temp_nodes = func(parent_node, **op.get('kwargs', {}), candidates=current_nodes)
                        
    #                     # 检查结果是否为空
    #                     if not temp_nodes or len(temp_nodes) < min_results:
    #                         if fallback and i > 0:
    #                             if debug:
    #                                 print(f"{step_name}: The number of results is insufficient ({len(temp_nodes) if temp_nodes else 0}). Keep the current results.")
    #                             fallback_step = True
    #                         else:
    #                             current_nodes = temp_nodes
    #                     else:
    #                         current_nodes = temp_nodes
                
    #             elif method_name == 'get_geonode_by_name':
    #                 args = op.get('args', [])
    #                 if not args or (len(args) == 1 and not args[0]):
    #                     # 如果没有提供名称模式，则返回所有地理节点
    #                     current_nodes = [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]
    #                 else:
    #                     temp_nodes = func(*args, candidates=current_nodes)
                        
    #                     # 检查结果是否为空
    #                     if not temp_nodes or len(temp_nodes) < min_results:
    #                         if fallback:
    #                             # 返回所有地理节点
    #                             if debug:
    #                                 print(f"{step_name}: No matching geographical nodes were found. Retrying with all geographical nodes.")
    #                             current_nodes = [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]
    #                             fallback_step = True
    #                         else:
    #                             current_nodes = temp_nodes
    #                     else:
    #                         current_nodes = temp_nodes
                
    #             elif method_name in ['filter_by_class', 'filter_by_attribute']:
    #                 # 确保 current_nodes 不为 None
    #                 if current_nodes is None:
    #                     current_nodes = list(self.graph.nodes.values())
                    
    #                 temp_nodes = func(*op.get('args', []), **op.get('kwargs', {}), candidates=current_nodes)
                    
    #                 # 检查结果是否为空
    #                 if not temp_nodes or len(temp_nodes) < min_results:
    #                     if fallback:
    #                         if debug:
    #                             print(f"{step_name}: filtered result is empty or insufficient. Maintain the current result.")
    #                         # 保持当前结果不变
    #                         fallback_step = True
    #                     else:
    #                         current_nodes = temp_nodes
    #                 else:
    #                     current_nodes = temp_nodes
                
    #             else:
    #                 # 其他方法
    #                 temp_nodes = func(*op.get('args', []), **op.get('kwargs', {}), candidates=current_nodes)
                    
    #                 # 检查结果是否为空
    #                 if not temp_nodes or len(temp_nodes) < min_results:
    #                     if fallback and i > 0:
    #                         if debug:
    #                             print(f"{step_name}: the result is empty or insufficient, maintain the current result.")
    #                         # 保持当前结果不变
    #                         fallback_step = True
    #                     else:
    #                         current_nodes = temp_nodes
    #                 else:
    #                     current_nodes = temp_nodes
                
    #             # 保存当前步骤的结果状态
    #             all_results.append(current_nodes)
    #             fallback_modes.append(fallback_step)
                
    #             # 记录日志
    #             log_entry["fallback"] = fallback_step
    #             log_entry["result_count"] = len(current_nodes) if current_nodes else 0
    #             log_entry["result_ids"] = [n.id for n in current_nodes] if current_nodes else []
    #             query_logs.append(log_entry)
                
    #             if debug:
    #                 print(f"{step_name}: Find {len(current_nodes) if current_nodes else 0} result nodes." +
    #                      (f" Backtracking mode" if fallback_step else ""))
            
    #         except Exception as e:
    #             if debug:
    #                 print(f"{step_name}: 发生错误 - {str(e)}")
                
    #             if fallback:
    #                 # 错误发生时，保持当前结果不变
    #                 if i > 0 and all_results:
    #                     if debug:
    #                         print(f"{step_name}error occurred. Keep the result of the previous step.")
    #                     fallback_step = True
    #                 else:
    #                     # 如果是第一步，或者没有前一步结果，使用所有节点
    #                     if debug:
    #                         print(f"{step_name} error occurred. Use all nodes.")
    #                     current_nodes = list(self.graph.nodes.values())
    #                     fallback_step = True
                    
    #                 fallback_modes.append(fallback_step)
    #                 all_results.append(current_nodes)
    #             else:
    #                 # 不使用回退机制，直接中断查询
    #                 break
        
    #     # 查询完成后，如果结果为空且允许回退
    #     if (not current_nodes or len(current_nodes) < min_results) and fallback and all_results:
    #         # 找到最后一个非回退步骤的结果
    #         for i in range(len(all_results)-1, -1, -1):
    #             if not fallback_modes[i] and all_results[i] and len(all_results[i]) >= min_results:
    #                 current_nodes = all_results[i]
    #                 if debug:
    #                     print(f"final result is empty or insufficient, revert to the result of step {i+1}. There are {len(current_nodes)} nodes in total.")
    #                 break
        
    #     if debug:
    #         print(f"Query complete. {len(current_nodes) if current_nodes else 0} result nodes are returned.")
        
    #     return current_nodes or [], query_logs

    def query_operation(self, operation_chain: list, target_description: str = "", debug=False):
        """执行操作链查询，封装了操作链执行的完整逻辑
        Args:
            operation_chain: 查询操作链
            target_description: 目标描述，用于 verify_spatial_relation
            debug: 是否输出调试信息
        Returns:
            (结果节点列表, 额外信息字典)
            - 结果节点列表: 查询到的节点或 fallback 节点
            - 额外信息字典: {"step_logs": [...], "verify_log": {...}}
        """
        if not operation_chain or not isinstance(operation_chain, list):
            if debug:
                print("Operation chain is empty or invalid")
            return [], {}

        current_nodes = None
        verify_candidates = None
        #last_valid_nodes = None
        step_logs = []
        verify_log_info = {}

        for i, op in enumerate(operation_chain):
            method_name = op.get('method')
            if debug:
                print(f"Step {i+1}: Executing {method_name}")

            if not hasattr(self, method_name):
                if debug:
                    print(f"Method '{method_name}' not found, skipping")
                continue

            func = getattr(self, method_name)

            try:
                if method_name == 'get_geonode_by_name':
                    current_nodes = func(*op.get('args', []), candidates=None)
                    if debug and current_nodes:
                        print(f"Found {len(current_nodes)} geo nodes")
                    if current_nodes:
                        last_valid_nodes = current_nodes
                    else:
                        # 如果没有提供名称模式，则返回所有地理节点
                        current_nodes = [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]
                        print(f"Have not find any candidates. Return all geo nodes")

                elif method_name == 'get_child_nodes':
                    if current_nodes is None:
                        current_nodes = [n for n in self.graph.nodes.values()
                                        if hasattr(n, 'object_type')]
                    parent = current_nodes if isinstance(current_nodes, list) else [current_nodes]
                    current_nodes = func(parent, **op.get('kwargs', {}), candidates=current_nodes)
                    # if current_nodes:
                    #     last_valid_nodes = current_nodes
                    if debug and current_nodes:
                        print(f"Found {len(current_nodes)} child nodes")

                elif method_name in ['filter_by_class', 'filter_by_attribute']:
                    if current_nodes is None:
                        current_nodes = list(self.graph.nodes.values())
                    current_nodes = func(*op.get('args', []), **op.get('kwargs', {}), candidates=current_nodes)
                    # if current_nodes:
                    #     last_valid_nodes = current_nodes
                    if debug and current_nodes:
                        print(f"After {method_name}: {len(current_nodes)} nodes")

                elif method_name == 'verify_spatial_relation':
                    if current_nodes:
                        verify_candidates = current_nodes
                    target = op.get('kwargs', {}).get('target', '')
                    try:
                        current_nodes = func(target_description, verify_candidates, target, debug=debug, log_info=verify_log_info)
                    except Exception as verify_err:
                        if debug:
                            print(f"verify_spatial_relation failed: {verify_err}")
                        current_nodes = []
                    # if current_nodes:
                    #     last_valid_nodes = current_nodes
                    if debug and current_nodes:
                        print(f"After verify_spatial_relation: {len(current_nodes)} nodes")

                else:
                    current_nodes = func(*op.get('args', []), candidates=current_nodes, **op.get('kwargs', {}))
                    # if current_nodes:
                    #     last_valid_nodes = current_nodes

            except Exception as e:
                if debug:
                    print(f"Error executing {method_name}: {e}")
                current_nodes = []

            node_ids = [n.id for n in current_nodes] if current_nodes else []
            step_logs.append({
                "step": i + 1,
                "method": method_name,
                "args": op.get('args', []),
                "kwargs": op.get('kwargs', {}),
                "result_count": len(node_ids),
                "result_ids": node_ids
            })
            if debug:
                print(f"Step {i+1} result IDs: {node_ids}")

            if current_nodes is None or (isinstance(current_nodes, list) and len(current_nodes) == 0):
                break

        extra_info = {
            "step_logs": step_logs,
            "verify_log": verify_log_info
        }

        if not current_nodes or (isinstance(current_nodes, list) and len(current_nodes) == 0):
            for op in operation_chain:
                if op.get('method') == 'get_geonode_by_name':
                    args = op.get('args', [])
                    if args:
                        geo_nodes = self.get_geonode_by_name(*args, candidates=None)
                        if geo_nodes:
                            if debug:
                                print(f"Fallback to first geo node: {geo_nodes[0].id}")
                        else:
                            # 如果提供名称与landmark不匹配，则返回所有地理节点
                            geo_nodes = [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]
                            print(f"Have not find any candidates. Return all geo nodes")
                    break
            current_nodes= geo_nodes if geo_nodes else []    
            #last_valid_nodes= geo_nodes if geo_nodes else []
            
        # 如果有多个候选节点，调用 verify_relation_constraints 选择最合适的一个
        if isinstance(current_nodes, list) and len(current_nodes) > 1 and target_description:
            try:
                sub_results = [(target_description, current_nodes)]
                verified, verify_log = self.controller.verify_relation_constraints(
                    target_description, sub_results, debug=debug
                )
                if verify_log:
                    verify_log_info["relation_constraint_verification"] = verify_log
                if verified:
                    if debug:
                        print(f"verify_relation_constraints selected {len(verified)} best node(s): {[n.id for n in verified]}")
                    current_nodes = verified
                else:
                    if debug:
                        print("verify_relation_constraints returned empty, keeping all candidates")
            except Exception as verify_err:
                if debug:
                    print(f"verify_relation_constraints failed: {verify_err}, keeping all candidates")

        return current_nodes if isinstance(current_nodes, list) else [current_nodes], extra_info

    # def get_geonode_by_name(self, name_pattern: str, candidates=None):
    #     """名称匹配地理节点"""
    #     nodes = candidates if candidates else self.graph.nodes.values()
    #     return [n for n in nodes if isinstance(n, GeoNode)
    #            and name_pattern.lower() in n.name.lower()]
    def get_geonode_by_name(self, name_pattern: str, candidates=None, threshold=0.8):
        """模糊名称匹配地理节点"""
        from difflib import SequenceMatcher
        
        def similarity(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()
        
        nodes = candidates if candidates else self.graph.nodes.values()
        results = []
        
        for n in nodes:
            if isinstance(n, GeoNode):
                if name_pattern.lower() in n.name.lower():  # 精确包含匹配
                    results.append((n, 1.0))
                else:  # 模糊匹配
                    sim = similarity(name_pattern, n.name)
                    if sim >= threshold:
                        results.append((n, sim))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in results]
    # 方向关系的反向映射
    DIRECTION_REVERSE_MAP = {
        "north_of": "south_of",
        "south_of": "north_of",
        "east_of": "west_of",
        "west_of": "east_of",
        "northeast_of": "southwest_of",
        "southwest_of": "northeast_of",
        "southeast_of": "northwest_of",
        "northwest_of": "southeast_of",
        "overlaps": "overlaps",
        "separates": "separates",
    }
    def get_child_nodes(self, parent_node, relation_type, candidates=None, bidirectional=True):
        """获取指定关系的子节点
        
        Args:
            parent_node: 单个节点或节点列表
            relation_type: 关系类型，支持三种格式：
                - 单个字符串: "contains"
                - 列表: ["separates", "overlaps"]
                - 逗号分隔字符串: "separates,overlaps"
            bidirectional: 是否双向查询，默认False。
                为True时，会同时查询入边（反向关系），适用于"找某节点的X方向有什么"，
                即使边是反方向存储的（如A在B东边，边为B→A relation="east_of"）也能查到。
        """
        # 支持多个父节点
        if isinstance(parent_node, list):
            parent_nodes = parent_node
        else:
            parent_nodes = [parent_node]
        
        # 验证关系类型是否有效
        valid_relations = [
            "contains", "overlaps", "separates",
            "north_of", "south_of", "east_of", "west_of",
            "northeast_of", "northwest_of", "southeast_of", "southwest_of"
        ]
        
        # 解析 relation_type：支持列表、逗号分隔字符串或单个字符串
        if isinstance(relation_type, list):
            target_relations = relation_type
        elif isinstance(relation_type, str) and ',' in relation_type:
            target_relations = [r.strip() for r in relation_type.split(',')]
        else:
            target_relations = [relation_type]
        
        # 验证所有关系类型是否有效
        invalid_relations = [r for r in target_relations if r not in valid_relations]
        if invalid_relations:
            print(f"Warning: Invalid relation type(s) {invalid_relations}. Using 'contains' instead.")
            target_relations = ["contains"]
        
        # 分离方向关系和非方向关系
        directional_relations = [r for r in target_relations if r in self.DIRECTION_REVERSE_MAP]
        # 计算方向关系的反向关系
        reverse_relations = [self.DIRECTION_REVERSE_MAP[r] for r in directional_relations]
        
        all_children = []
        seen_ids = set()  # 用于去重
        
        # 对每个父节点执行操作
        for parent in parent_nodes:
            # 查询出边：匹配目标关系（方向关系 + 非方向关系）
            for edge in self.graph.digraph_nx.out_edges(parent.id, data=True):
                child = self._match_edge_relation(edge, target_relations, seen_ids, return_source=False)
                if child:
                    all_children.append(child)
            # 双向查询：查询入边（只用于方向关系），匹配反向关系，返回source节点
            if bidirectional and reverse_relations:
                for edge in self.graph.digraph_nx.in_edges(parent.id, data=True):
                    child = self._match_edge_relation(edge, reverse_relations, seen_ids, return_source=True)
                    if child:
                        all_children.append(child)
        return all_children
    
    def _match_edge_relation(self, edge, target_relations, seen_ids, return_source=False):
        """检查边的关系是否匹配目标关系，返回匹配的节点
        Args:
            edge: 边 tuple (source, target, attrs)
            target_relations: 目标关系列表
            seen_ids: 已访问的节点ID集合（用于去重）
            return_source: 是否返回source节点（用于反向查询）
        """
        edge_relations = edge[2].get('relations', edge[2].get('relation_type', []))
        if isinstance(edge_relations, list):
            edge_rel_set = set(edge_relations)
        else:
            edge_rel_set = {edge_relations}
            
        if edge_rel_set & set(target_relations):
            # 根据 return_source 决定返回 source 还是 target
            node_id = edge[0] if return_source else edge[1]
            child = self.graph.nodes[node_id]
            if child.id not in seen_ids:
                seen_ids.add(child.id)
                return child
        return None
    def filter_by_class(self, obj_class: str, candidates):
        """按类别过滤物体节点""" 
        return [n for n in candidates if isinstance(n, ObjectNode) 
               and n.obj_class == obj_class]
    # def filter_by_attribute(self, key: str, value: str, candidates):
    #     """按属性过滤物体节点"""
    #     return [n for n in candidates if isinstance(n, ObjectNode) 
    #            and hasattr(n, key) and getattr(n, key) == value]
    def filter_by_attribute(self, key: str, value: str, candidates, threshold=0.7):
        """模糊属性匹配"""
        from difflib import SequenceMatcher
        
        def similarity(a, b):
            return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
        
        results = []
        for n in candidates:
            if isinstance(n, ObjectNode):
                if hasattr(n, key):
                    attr_value = getattr(n, key)
                    # 精确匹配
                    if str(attr_value).lower() == str(value).lower():
                        results.append((n, 1.0))
                    # 模糊匹配
                    else:
                        sim = similarity(attr_value, value)
                        if sim >= threshold:
                            results.append((n, sim))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in results]
    
    def verify_spatial_relation(self, instruction: str, candidates: list, target: str, debug=False, log_info: dict = None):
        """验证候选节点是否满足空间关系约束
        Args:
            instruction: 任务描述
            candidates: 候选节点列表
            target: 目标对象
            debug: 是否输出调试信息
            log_info: 用于存储验证过程的日志（会追加记录）
        Returns:
            通过所有子关系验证的候选节点列表
        """
        if log_info is None:
            log_info = {}

        if not candidates:
            print("Warning: No candidates provided for spatial relation verification")
            log_info["error"] = "No candidates provided"
            return []

        # 初始化日志结构
        if "verification_rounds" not in log_info:
            log_info["verification_rounds"] = []

        # 获取子关系描述
        sub_queries = self.controller.decompose_spatial_relation(instruction, target, debug)
        if not sub_queries:
            if debug:
                print("Warning: Failed to decompose instruction into sub-queries")
            log_info["error"] = "Failed to decompose spatial relation"
            return []

        if debug:
            print(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

        # 一次性生成所有子查询的操作链，确保所有候选节点使用相同的操作链
        sub_query_chains = {}
        for sq in sub_queries:
            prompt = QUERY_OPERATION_SUBCHAIN_PROMPT.format(instruction=sq)
            try:
                response = self.controller.llm_client.chat.completions.create(
                    model=LLM_NAME,
                    max_tokens=2048,
                    messages=[
                        {"role": "system", "content": "You are a professional query planner."},
                        {"role": "user", "content": prompt}
                    ]
                )
                raw_content = response.choices[0].message.content
                operation_chain = self.controller.robust_json_loads(raw_content)
            except Exception as e:
                if debug:
                    print(f"Error generating operation chain for sub-query '{sq}': {e}")
                sub_query_chains[sq] = []  # 记录为空链，待查

            if not operation_chain or not isinstance(operation_chain, list):
                if debug:
                    print(f"Invalid operation chain for sub-query '{sq}': {operation_chain}")
                sub_query_chains[sq] = []
            else:
                sub_query_chains[sq] = operation_chain

        if debug:
            for sq, chain in sub_query_chains.items():
                print(f"Sub-query: '{sq}' -> Operation chain: {chain}")

        # 记录当前轮的汇总信息
        round_summary = {
            "instruction": instruction,
            "target": target,
            "sub_queries": sub_queries,
            "input_candidates": [c.id for c in candidates],
            "sub_query_chains": {sq: chain for sq, chain in sub_query_chains.items()},
            "candidates_detail": []
        }

        # 对每个候选节点应用相同的操作链进行验证
        verified_candidates = []
        for candidate in candidates:
            if debug:
                print(f"Verifying candidate: {candidate.id}")

            candidate_detail = {
                "candidate_id": candidate.id,
                "sub_query_results": []
            }

            is_valid = True
            failed_reason = None

            for sub_query in sub_queries:
                if not is_valid:
                    # 记录未执行的子查询（前置子查询已失败）
                    candidate_detail["sub_query_results"].append({
                        "sub_query": sub_query,
                        "skipped": True,
                        "reason": failed_reason or "previous sub-query failed"
                    })
                    continue

                operation_chain = sub_query_chains[sub_query]
                current_nodes = [candidate]
                sub_query_result = {
                    "sub_query": sub_query,
                    "operation_chain": operation_chain,
                    "steps": [],
                    "passed": False,
                    "error": None
                }

                # 执行操作链中的每一步
                for op in operation_chain:
                    step_result = {
                        "step_index": len(sub_query_result["steps"]) + 1,
                        "method": op.get('method'),
                        "args": op.get('args', []),
                        "kwargs": op.get('kwargs', {}),
                        "input_count": len(current_nodes) if current_nodes else 0,
                        "input_ids": [n.id for n in current_nodes] if current_nodes else [],
                        "output_count": 0,
                        "output_ids": [],
                        "error": None
                    }

                    if not current_nodes:
                        step_result["error"] = "No input nodes (previous step failed)"
                        sub_query_result["steps"].append(step_result)
                        is_valid = False
                        failed_reason = f"sub-query '{sub_query}', step '{op.get('method')}' had no input nodes"
                        break

                    method_name = op.get('method')
                    if not method_name:
                        step_result["error"] = "No method name specified"
                        sub_query_result["steps"].append(step_result)
                        continue

                    if not hasattr(self, method_name):
                        step_result["error"] = f"Method '{method_name}' not found"
                        sub_query_result["steps"].append(step_result)
                        continue

                    func = getattr(self, method_name)
                    try:
                        if method_name == 'get_child_nodes':
                            parent_node = current_nodes if isinstance(current_nodes, list) else [current_nodes]
                            current_nodes = func(parent_node, **op.get('kwargs', {}), candidates=current_nodes)
                        elif method_name == 'filter_by_class':
                            args = op.get('args', [])
                            if args:
                                current_nodes = func(args[0], candidates=current_nodes)
                            else:
                                step_result["error"] = "filter_by_class missing args"
                        elif method_name == 'filter_by_attribute':
                            args = op.get('args', [])
                            if len(args) >= 2:
                                current_nodes = func(args[0], args[1], candidates=current_nodes)
                            else:
                                step_result["error"] = "filter_by_attribute missing sufficient args"
                        else:
                            args = op.get('args', [])
                            kwargs = op.get('kwargs', {})
                            current_nodes = func(*args, candidates=current_nodes, **kwargs)
                    except Exception as e:
                        step_result["error"] = str(e)
                        current_nodes = None

                    # 记录输出
                    step_result["output_count"] = len(current_nodes) if current_nodes else 0
                    step_result["output_ids"] = [n.id for n in current_nodes] if current_nodes else []
                    sub_query_result["steps"].append(step_result)

                    if debug:
                        print(f"  Candidate {candidate.id} | sub-query '{sub_query}' | step {step_result['step_index']} ({method_name}): "
                              f"in={step_result['input_count']} -> out={step_result['output_count']}"
                              + (f" ERROR: {step_result['error']}" if step_result['error'] else ""))

                    # 如果该步骤输出为空，标记失败
                    if not current_nodes or (isinstance(current_nodes, list) and len(current_nodes) == 0):
                        is_valid = False
                        failed_reason = f"sub-query '{sub_query}', step '{method_name}' returned empty result"
                        break

                # 判断该子查询是否通过（is_valid 仍为 True 且 current_nodes 非空）
                if is_valid and current_nodes:
                    sub_query_result["passed"] = True
                else:
                    if is_valid:
                        sub_query_result["error"] = "Empty result after operation chain"
                    # is_valid 和 failed_reason 已在循环中设置
                    sub_query_result["final_nodes_count"] = len(current_nodes) if current_nodes else 0
                    sub_query_result["final_nodes_ids"] = [n.id for n in current_nodes] if current_nodes else []

                candidate_detail["sub_query_results"].append(sub_query_result)

                if debug and not sub_query_result["passed"]:
                    print(f"Candidate {candidate.id} failed at sub-query: {sub_query} | reason: {failed_reason}")

            candidate_detail["verified"] = is_valid
            candidate_detail["failed_reason"] = failed_reason
            round_summary["candidates_detail"].append(candidate_detail)

            if is_valid:
                verified_candidates.append(candidate)

        round_summary["output_candidates"] = [c.id for c in verified_candidates]
        round_summary["passed_count"] = len(verified_candidates)
        round_summary["failed_count"] = len(candidates) - len(verified_candidates)

        log_info["verification_rounds"].append(round_summary)

        if debug:
            print(f"Verified {len(verified_candidates)}/{len(candidates)} candidates")

        return verified_candidates
    
    def filter_by_relative_position(self, reference_node, direction: str, distance: float, candidates=None):
        """根据相对位置过滤节点"""
        if candidates is None:
            candidates = list(self.graph.nodes.values())
        
        filtered = []
        for node in candidates:
            if isinstance(node, ObjectNode):
                dx = node.position.x - reference_node.position.x
                dy = node.position.y - reference_node.position.y
                dist = math.hypot(dx, dy)
                
                if dist <= distance:
                    angle = math.degrees(math.atan2(dy, dx)) % 360
                    if self._is_direction_match(angle, direction):
                        filtered.append(node)
        return filtered

    def _is_direction_match(self, angle: float, direction: str) -> bool:
        """判断角度是否匹配指定方向"""
        direction_map = {
            "north": (337.5, 22.5),
            "northeast": (22.5, 67.5),
            "east": (67.5, 112.5),
            "southeast": (112.5, 157.5),
            "south": (157.5, 202.5),
            "southwest": (202.5, 247.5),
            "west": (247.5, 292.5),
            "northwest": (292.5, 337.5)
        }
        
        if direction not in direction_map:
            return False
        
        start, end = direction_map[direction]
        return start <= angle < end
    @staticmethod
    def _calc_distance(p1: Point2D, p2: Point2D):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    
    def is_within_geo_node(self, position: Point2D):
         point = Point(position.x, position.y)
         name_list = []
         for node in self.graph.nodes.values():
             if isinstance(node, GeoNode):
                 polygon = node.contour_polygon
                 #print(f'地标轮廓为 {polygon}')
                 if polygon.contains(point):
                     name_list.append(node.name)
         if name_list:
             return f' in {", ".join(name_list)}'
         return f' not in any landmarks. '
     
    #描述地标的轮廓坐标和质心坐标
    def describe_landmark_positions(self) -> str:
        entries = []
        for node in self.graph.nodes.values():
            if isinstance(node, GeoNode):
                pos = node.position
                entries.append(f"{node.name}'s centroid coordinates is: ({pos.x:.1f}, {pos.y:.1f})")
        if not entries:
            return "the position of landmarks is unknown"
        return " | ".join(entries)
    
    def get_enhanced_geo_relation(self, position: Point2D) -> str:
        """增强版地理关系描述：综合包含性、边界角和距离信息"""
        point = Point(position.x, position.y)
        descriptions = []
        
        for node in self.graph.nodes.values():
            if not isinstance(node, GeoNode):
                continue
                
            # 基础信息计算
            distance = self._calc_distance(node.position, position)
            direction = get_direction(position, node.position)
            polygon = node.contour_polygon
            
            # 情形1：在轮廓多边形内部
            if polygon.contains(point):
                # 计算相对于多边形中心的方位
                centroid = polygon.centroid
                intra_direction = get_direction(
                    Point2D(centroid.x, centroid.y), 
                    position
                )
                descriptions.append(f"You are inside {node.name} ({intra_direction} area)")
                continue
                
            # 情形2：接近多边形顶点（角落检测）
            closest_corner, min_corner_dist = None, float('inf')
            for coord in polygon.exterior.coords[:-1]:  # 排除重复的闭合点
                corner_dist = math.hypot(position.x-coord[0], position.y-coord[1])
                if corner_dist < min_corner_dist:
                    min_corner_dist = corner_dist
                    closest_corner = Point2D(coord[0], coord[1])
                    
            if min_corner_dist < 5.0:  # 5米内视为接近角落
                corner_dir = get_direction(closest_corner, position)
                descriptions.append(
                    f"near {node.name}'s {corner_dir} corner "
                    f"({min_corner_dist:.1f}m)")
                continue
                
            # 情形3：外部普通方位
            descriptions.append(
                f"{node.name} is {distance:.1f}m {direction} of you")
        '''
        if not descriptions:
            return "No nearby landmarks detected"
        print("Enhanced Geo-Relation Descriptions:")
        for desc in descriptions:
            print(f"  - {desc}")
        '''
        return " | ".join(sorted(descriptions, key=lambda x: len(x)))

    # 示例用法
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 创建地理参考对象
    road = CityReferObject(
        map_name="birmingham_block_1",
        obj_id=659,
        name="Leslie Road",
        obj_type="TrafficRoad",
        position=Point2D(380.6, 448.5),
        dimension=Point2D(38.5, 97.1),
        contour=[Point2D(359.0, 400.0), Point2D(363.9, 415.6)]
    )

    # 初始化系统
    graph = KnowledgeGraph()
    graph.add_geo_node(road)

    # 模拟无人机检测到物体
    graph.add_object_node(Point2D(382.0, 450.0), "汽车", 0.95)
    graph.add_object_node(Point2D(378.0, 445.0), "行人", 0.87)
    graph.add_object_node(Point2D(378.1, 445.1), "自行车", 0.92)  # 应被去重

    # 执行查询
    query_engine = QueryEngine(graph)
    current_pos = Point2D(380.0, 449.0)
    context = query_engine.get_context(current_pos, radius=15.0)
    contain = query_engine.get_enhanced_geo_relation(current_pos)

    # 打印查询结果
    print("Current Position is", (current_pos.x, current_pos.y))
    if contain:
        print(f"Current Position is in landmark {contain}")
    for item in context:
        print(f"- {item['type'].upper()}: {item.get('name','')}{item.get('class','')}"
                f" {item['distance']} {item['direction']} direction")