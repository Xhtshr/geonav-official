from utils.tools import get_direction
import math
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass
from rtree import index  # pip install rtree

import sys
sys.path.append("/data1/XHT/citynav/")
from gsamllavanav.space import Point2D
from gsamllavanav.cityreferobject import CityReferObject
from shapely.geometry import Point, Polygon


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
        """添加带有属性的边"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Cannot create edge between non-existing nodes")
        
        # 添加基本边
        self.graph_nx.add_edge(source_id, target_id, relation_type=relation_type)
        self.digraph_nx.add_edge(source_id, target_id, relation_type=relation_type)
        
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
        Flag = self._add_node(node, obj_class)
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
    
    @staticmethod
    def _distance(p1: Point2D, p2: Point2D):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

# ================== 查询引擎 ==================
class QueryEngine:
    def __init__(self, graph: KnowledgeGraph):
        
        self.graph = graph
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
                parent_node = current_nodes[0]
                current_nodes = func(parent_node, **op.get('kwargs', {}), candidates=current_nodes)
            elif method_name == 'get_geonode_by_name':
                # 特殊处理 get_geonode_by_name，当 args 为空时使用所有地理节点
                args = op.get('args', [])
                if not args or (len(args) == 1 and not args[0]):
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
                                    print(f"{step_name}: 结果数量不足({len(temp_nodes) if temp_nodes else 0})，维持当前结果")
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
                                    print(f"{step_name}: 没有找到匹配的地理节点，回退到所有地理节点")
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
                                print(f"{step_name}: 过滤后结果为空或不足，维持当前结果")
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
                                print(f"{step_name}: 结果为空或不足，维持当前结果")
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
                    print(f"{step_name}: 找到 {len(current_nodes) if current_nodes else 0} 个结果节点" +
                         (f" (回退模式)" if fallback_step else ""))
            
            except Exception as e:
                if debug:
                    print(f"{step_name}: 发生错误 - {str(e)}")
                
                if fallback:
                    # 错误发生时，保持当前结果不变
                    if i > 0 and all_results:
                        if debug:
                            print(f"{step_name}: 发生错误，维持前一步结果")
                        fallback_step = True
                    else:
                        # 如果是第一步，或者没有前一步结果，使用所有节点
                        if debug:
                            print(f"{step_name}: 发生错误，使用所有节点")
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
                        print(f"最终结果为空或不足，回退到步骤{i+1}的结果，共{len(current_nodes)}个节点")
                    break
        
        if debug:
            print(f"查询完成，返回 {len(current_nodes) if current_nodes else 0} 个结果节点")
        
        return current_nodes or []

    # def get_geonode_by_name(self, name_pattern: str, candidates=None):
    #     """名称匹配地理节点"""
    #     nodes = candidates if candidates else self.graph.nodes.values()
    #     return [n for n in nodes if isinstance(n, GeoNode) 
    #            and name_pattern.lower() in n.name.lower()]
    def get_geonode_by_name(self, name_pattern: str, candidates=None, threshold=0.6):
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
    def get_child_nodes(self, parent_node, relation_type: str, candidates=None):
        """获取指定关系的子节点"""
        # 验证关系类型是否有效
        valid_relations = [
            "contains", "adjacent_to", "near_corner", 
            "north_of", "south_of", "east_of", "west_of",
            "northeast_of", "northwest_of", "southeast_of", "southwest_of"
        ]
        
        if relation_type not in valid_relations:
            print(f"Warning: Invalid relation type '{relation_type}'. Using 'contains' instead.")
            relation_type = "contains"
        
        children = []
        for edge in self.graph.digraph_nx.out_edges(parent_node.id, data=True):
            child = self.graph.nodes[edge[1]]
            children.append(child)
            # if edge[2]['relation_type'] == relation_type:
            #     child = self.graph.nodes[edge[1]]
            #     children.append(child)
        return children
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
                 if polygon.contains(point):
                     name_list.append(node.name)
         if name_list:
             return f' in {", ".join(name_list)}'
         return f' not in any landmarks. '
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
        
        if not descriptions:
            return "No nearby landmarks detected"
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