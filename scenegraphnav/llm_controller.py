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
        return rgb, depth

    def understand(self, image_64: str, epi: Episode):
        # 从RGB中提取地标和目标的位置信息，使用自然语言处理来存储json格式
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        detection_prompt = ""
        responses = []

        if epi.description_target:
            detection_prompt += f"""Analyze the image and output the information of {epi.target_type} strictly in JSON format:
            {{
                "detections": [
                    {{
                        "object_type": "{epi.target_type}",
                        "bbox": [xmin, ymin, xmax, ymax],
                        "color": "main_color",
                        "confidence": "from 0 to 1.0"
                    }}
                ]
            }}

            Rules:
            1. Sort position from left to right based on bbox x-center
            2. Use web color names (red, blue, etc.)
            3. For unclear attributes, use "unknown"
            """
            response1 = client.chat.completions.create(
                model="qwen-vl-max-latest",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional urban scene analyzer. Output must be valid JSON."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{image_64}"
                            },
                            {"type": "text", "text": detection_prompt}
                        ]
                    }
                ]
            )
            responses.append(response1)

        if epi.description_surroundings:
            detection_prompt = f"""Analyze the {epi.description_surroundings} in the image and output information strictly in JSON format:
            {{
                "surroundings": [
                    {{
                        "object_type": " ",
                        "bbox": [xmin, ymin, xmax, ymax],
                        "color": "main_color"
                    }}
                ]
            }}

            Rules:
            1. Sort position from left to right based on bbox x-center
            2. Use web color names (red, blue, etc.)
            3. For unclear attributes, use "unknown"
            """
            response2 = client.chat.completions.create(
                model="qwen-vl-max-latest",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional urban scene analyzer. Output must be valid JSON."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{image_64}"
                            },
                            {"type": "text", "text": detection_prompt}
                        ]
                    }
                ]
            )
            responses.append(response2)

        return [json.loads(response.choices[0].message.content) for response in responses]

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

    def build_scene_graph(self, objects, landmarks, show=True, time_window=3):

        for landmark in landmarks:
            self.scene_graph.add_geo_node(landmark)
        
        for object in objects:
            self.scene_graph.add_object_node(object[0], object[1], object[2], self.timestep)

        # 执行查询
        query_engine = QueryEngine(self.scene_graph)
        current_pos = self.pose.xy
        context = query_engine.get_context(self.pose.xy, radius=30.0)
        landmark = query_engine.is_within_geo_node(current_pos) + query_engine.get_geo_node_info(current_pos)

        surrounding = ''
        recent_objects = query_engine.get_recent_objects(current_timestamp=self.timestep, time_window=time_window)
        surrounding += f"Current Position is {(round(current_pos.x,2), round(current_pos.y,2))}"
        for item in context:
            surrounding += f"The {item['type'].upper()} Node {item.get('name','')}{item.get('class','')} is {item['distance']} to the {item['direction']}."
        if show == True:
            visualize_knowledge_graph(self.scene_graph, current_pos)
        
        return surrounding, landmark, recent_objects

    def parse_instruction(self, instruction: str):
        # 解析指令，提取地标和目标
        from scenegraphnav.prompt.instruction import create_prompt, gpt_api_call
        from scenegraphnav.agent import extract_json_from_msg
        prompt = create_prompt(instruction)
        response = gpt_api_call(prompt)
        self.intr_knowledge = extract_json_from_msg(response)
        if self.intr_knowledge is None:
            self.intr_knowledge = extract_json_from_msg(response)
        return self.intr_knowledge
