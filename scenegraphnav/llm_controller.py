from openai import OpenAI
from json import dumps as json_dumps  # 添加在文件顶部
from scenegraphnav.parser import parse_args
from gsamllavanav.parser import ExperimentArgs

from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose
from gsamllavanav.observation import cropclient
from scenegraphnav.city_scene_graph import KnowledgeGraph, QueryEngine, visualize_knowledge_graph
from gsamllavanav.dataset.episode import Episode

import os
import json
import numpy as np

class LLMController:
    def __init__(self, args: ExperimentArgs, pose: Pose4D):
        self.args = args
        self.pose = pose
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
        PROMPT = """ You need to extract scene information from RGB image into a structured JSON graph. Output must represent the scene as a graph with "nodes" and "edges". Each node represents an object or scene element, and each edge represents a spatial relationship (i.e., a semantic link) between two nodes.
And the JSON format can be adhered to.
Requirements:
1. Each node must have a unique "id".
2. Each node must have "object": the type or description of the object (e.g., {objects}).
3. Each node must have "bbox": a placeholder for bounding box coordinates in the format [xmin, ymin, xmax, ymax].
4. Each node may include the following attributes:
   - "color": the primary color or color attributes (e.g., "white", "gray", "brown", "multicolored").(if available).
   - "feature": an object containing additional properties (e.g., "roof type", "orientation" such as "front", "side", "rear", "size", etc.).
3. Each edge must include:
   - "source": the id of the source node.
   - "target": the id of the target node.
   - "relationship": a description of the spatial or directional relationship from source node to target node(e.g., "left", "right", "in front", "behind", "across", "facing", "along", "near", "in between", "at bottom left").

You have the flexibility to adapt the JSON structure based on the given instruction. Your output should capture the scene's underlying graph structure by identifying objects and their spatial relationships.
<Question>
Consider the following type of the object: {objects}, and extract the spatial knowledge graph.
<Question>
<Answer>
Must be in JSON format, includes Nodes (with attributes) and Edges (with relationships). like


<Answer>
Now, the answer is
"""
        PROMPT = """
You are given two inputs: 1. A known spatial description graph (prior knowledge) in JSON format:
   {prior_graph}
2. An RGB image representing the current scene.

Following the provided spatial description graph template, your task is to extract scene information from the image into a JSON graph. Specifically:
- Ignore the examples that are built into template graph, and fill the graph with actual nodes and edges.
- For each object in the known spatial description graph, verify its presence in the image and update its attributes (e.g., bounding box, color, features) based on the visual data.
- Identify any additional objects in the RGB image that are not present in the prior graph, and include them as new nodes.
- Determine spatial relationships between objects (both known and new) and add corresponding edges. Use relationships such as "left", "right", "in front", "behind", "near", etc.
- Each node must have a unique "id". Attributes include "object", "bbox" (in the format [xmin, ymin, xmax, ymax]). And if available,"color", "shape" object for other attributes.
- Each edge must include a "source" (node id), a "target" (node id), and a "relationship" describing the spatial connection.

The final output must be a valid JSON object in the following structure:
{{
  "nodes": [ ... ],
  "edges": [ ... ]
}}

Please integrate the prior graph information with the current image extraction and output the updated scene graph in JSON format.

<Answer>
Now, the answer is:
""".format(prior_graph=json.dumps(self.intr_knowledge, ensure_ascii=False))

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
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
        landmark = self.query_engine.get_enhanced_geo_relation(self.pose.xy)
        return landmark

    def build_scene_graph(self, subgraph, gsm):
        #过滤node_data.get("bbox", [])==[]的节点
        subgraph["nodes"] = [node for node in subgraph["nodes"] if node.get("bbox", [])]
        id_mapping = {}
        bboxes = [node_data.get("bbox", []) for node_data in subgraph["nodes"]]
        
        # pos, class, confidence, timestamp
        poses = gsm.bbox_to_global_pos(bboxes)
        for pos, node in zip(poses,subgraph["nodes"]):
            global_id = self.scene_graph.add_object_node(pos, node['object'], confidence=1.0, timestamp=self.timestep)
            if global_id is not None:
                id_mapping[node['id']] = global_id

        for edge in subgraph["edges"]:
            source = id_mapping.get(edge["source"])
            target = id_mapping.get(edge["target"])
            if source is None or target is None: 
                continue
            relationship = edge["relationship"]
            self.scene_graph.add_edge(source, target, relationship)
        self.query_engine = QueryEngine(self.scene_graph)
        
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
        if show == True:
            visualize_knowledge_graph(self.scene_graph, current_pos)
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
