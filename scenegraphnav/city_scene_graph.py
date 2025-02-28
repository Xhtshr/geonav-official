import time
import math
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass
from rtree import index  # 需要提前安装：pip install rtree

import sys
sys.path.append("/data1/XHT/citynav/")
from gsamllavanav.space import Point2D
from gsamllavanav.cityreferobject import CityReferObject
from shapely.geometry import Point, Polygon


# ================== 知识图谱节点 ==================
class KnowledgeNode:
    def __init__(self, node_id, node_type, position, timestamp):
        self.id = node_id
        self.type = node_type
        self.position = position
        self.timestamp = timestamp
        self.relations = {
            'spatial': defaultdict(list),
            'semantic': defaultdict(list),
            'temporal': defaultdict(list)
        }

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

class ObjectNode(KnowledgeNode):
    def __init__(self, detection_id, position, obj_class, confidence):
        super().__init__(
            node_id=f"obj_{detection_id}",
            node_type='object',
            position=position,
            timestamp=time.time()
        )
        self.obj_class = obj_class
        self.confidence = confidence

# ================== 空间索引与图谱 ==================
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
    
    def add_geo_node(self, city_obj: CityReferObject):
        node = GeoNode(city_obj)
        self._add_node(node)
    
    def add_object_node(self, position, obj_class, confidence):
        node_id = f"obj_{len(self.nodes)}"
        node = ObjectNode(node_id, position, obj_class, confidence)
        self._add_node(node)
    
    def _add_node(self, node):
        # 简单去重策略：相同位置同类型视为同一对象
        existing = self.spatial_index.query_radius(node.position, 1.0)
        for n in existing:
            if n.type == node.type and self._distance(n.position, node.position) < 1.0:
                return  # 跳过重复对象
        self.nodes[node.id] = node
        self.spatial_index.insert(node)
    
    @staticmethod
    def _distance(p1: Point2D, p2: Point2D):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

# ================== 查询引擎 ==================
class QueryEngine:
    DIRECTION_NAMES = ["East", "Northeast", "North", "Northwest", "West", "Southwest", "South", "Southeast"]
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
    
    def get_context(self, position: Point2D, radius=10.0):
        nodes = self.graph.spatial_index.query_radius(position, radius)
        return [self._describe_node(n, position) for n in nodes]
    
    def _describe_node(self, node, query_pos):
        desc = {
            'type': node.type,
            'distance': f"{self._calc_distance(node.position, query_pos):.1f} meters",
            'direction': self._get_direction(query_pos, node.position)
        }
        if isinstance(node, GeoNode):
            desc['name'] = node.name
            desc['category'] = node.object_type
        elif isinstance(node, ObjectNode):
            desc['class'] = node.obj_class
            desc['confidence'] = node.confidence
        return desc
    
    @staticmethod
    def _calc_distance(p1: Point2D, p2: Point2D):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    
    def _get_direction(self, src: Point2D, target: Point2D):
        dx = target.x - src.x
        dy = target.y - src.y
        angle = math.degrees(math.atan2(dy, dx)) % 360
        return self.DIRECTION_NAMES[round(angle / 45) % 8]
    
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
        return f' not in any landmarks'
    
    # 这里我希望实现输入位置，检索到geo_node的距离和方向，并回复文本答案
    def get_geo_node_info(self, position: Point2D):
        answer = ""
        for node in self.graph.nodes.values():
            if isinstance(node, GeoNode):
                distance = self._calc_distance(node.position, position)
                direction = self._get_direction(position, node.position)
                answer += f"The '{node.name}' at a distance of {distance:.1f} meters towards {direction}.\n"
        return answer
    


def visualize_knowledge_graph(graph, current_pos):
    import matplotlib.pyplot as plt

    # 可视化物体图谱
    fig, ax = plt.subplots()
    ax.set_title("Knowledge Graph Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # 绘制地理节点
    for node in graph.nodes.values():
        if isinstance(node, GeoNode):
            ax.plot(node.position.x, node.position.y, 'bo', label='GeoNode')
            ax.text(node.position.x, node.position.y, node.name, fontsize=9, ha='right')

    # 绘制物体节点
    for node in graph.nodes.values():
        if isinstance(node, ObjectNode):
            ax.plot(node.position.x, node.position.y, 'ro', label='ObjectNode')
            ax.text(node.position.x, node.position.y, node.obj_class, fontsize=9, ha='right')

    # 绘制当前查询位置
    ax.plot(current_pos.x, current_pos.y, 'go', label='Current Position')
    ax.text(current_pos.x, current_pos.y, 'Current Position', fontsize=9, ha='right')

    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()

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
    contain = query_engine.is_within_geo_node(current_pos) + query_engine.get_geo_node_info(current_pos)


    # 打印查询结果
    print("Current Position is", (current_pos.x, current_pos.y))
    if contain:
        print(f"Current Position is in landmark {contain}")
    for item in context:
        print(f"- {item['type'].upper()}: {item.get('name','')}{item.get('class','')}"
                f" {item['distance']} {item['direction']} direction")

    # 可视化知识图谱
    visualize_knowledge_graph(graph, current_pos)