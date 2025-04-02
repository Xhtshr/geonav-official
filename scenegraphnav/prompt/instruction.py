import os
import json
from openai import OpenAI

# 构造 Prompt
def create_prompt(instruction, landmarks):
    return """You are a graph generator that converts language descriptions into a structured JSON graph. Your output must represent the scene as a graph with "nodes" and "edges". Each node represents an object and each edge represents a semantic link between two nodes.
Requirements:
1. Don't process {} as nodes, because these landmark is not necessary and should not be included in the graph.
2. Each node must have a unique "id".
3. Each node may include the following attributes:
   - "object": the type or description of the object (e.g., "white-roofed house", "tree", "white car", "parking lot").
   - "bbox": a placeholder for bounding box coordinates in the format [xmin, ymin, xmax, ymax]. (place [] if not available)
   - "attributes": an object containing additional properties (e.g., "color", "size", "orientation", "roof type", etc.).
4. Each edge must include:
   - "source": the id of the source node.
   - "target": the id of the target node.
   - "relationship": a description of the spatial or directional relationship (e.g., "in front", "behind", ...).
you have some flexibility in defining node properties and relationship descriptions. Now, based on these patterns, generate a JSON graph for a given instruction: {}.
""".format(landmarks, instruction)
    return """You are a language navigation assistant. Your task is to analyze complex navigation instructions and extract the following structured information in JSON format:
- Target: The main object or location to be navigated to. And list its attributes such as color, shape, etc.
- Landmarks: Any referenced Geosptial names that help identify the target's position. Note that landmarks are usually capitalized names of streets, roads, etc. List the relationships between the target and landmarks, and mention the secondary landmarks if the target near the intersection of two landmarks.
- Surrounding: Any referenced objects, environmental or contextual information not part of the main landmarks but provides additional clues.
- Spatial Relationships with objects: The spatial and positional relationships between the target and landmarks. Provide your answer in JSON format.

**Required Output Format**  
    ```json
{{
  "Target": {{
    "class": "class_name", # object class should be typical, e.g. car, building, etc.
    "attribute":{{
        "attribute 1": "value 1", # optional
        "attribute 2": "value 2",
        ...
    }},
  }},
  "Spatial Relationships": {{
    "intersection": ["Landmark 1", "Landmark 2",...], # return none if no intersection
    "Landmark":{{
        "Landmark 1": "relationship",
        "Landmark 2": "relationship,
       ... 
    }}
  }},
  "Surrounding": {{
    "class": "class_name", # object class should be typical, e.g. car, building, etc.
    "attribute":{{
        "attribute 1": "value 1", # optional
        "attribute 2": "value 2",
        ...
    }}
}}```


Example:
Instruction: "A white car behind a black car, with a black car across from it on the opposite side of Willmore Road facing the edge of the map, in between two identical multi-housing units."
**Required json Format**  
Extracted:
```json
{{
  "Target": {{
    "class": "car",
    "attribute":{{
        "color": "white",
    }}
  }},
  "Relationships with Landmarks": {{
    "intersection": null, # return null if no intersection
    "Landmark":{{
            "Willmore Road": "on the Willmore Road",
        }}
  }},
  "Surrounding": ["car", "multi-housing units"]
}}```

Instruction: "A dark blue car near the corner of Beche Road and Priory Road next to the Cellarer's Chequer building"
**Required json Format**  
```json
{{
  "Target": {{
    "class": "car",
    "attribute":{{
        "color": "dark blue",
    }}
  }},
  "Relationships with Landmarks": {{
    "intersection": ["Beche Road", "Priory Road"], # return none if no intersection
    "Landmark":{{
        "Beche Road": "near the Beche Road",
        "Priory Road": "near the Priory Road",
        "Cellarer's Chequer": "next to the Cellarer's Chequer building"
    }}
  }},
  "Surrounding": ["building"]
}}```
""" + """Now process the following instruction: {}""".format(instruction)

# 调用 OpenAI GPT 模型
def gpt_api_call(prompt):
    # 导入openai key配置
    os.environ["OPENAI_API_KEY"] = "sk-f0de3487904a4a11950ba707623cdbab"
    os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    response = client.chat.completions.create(
        model="qwen-max-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01  # 设置为 0.01 以确保解析的稳定性
    )
    return response.choices[0].message.content

# 解析 GPT 返回的结构化文本
def parse_response(response):
    return json.loads(response)

# 主流程
if __name__ == "__main__":
    # 示例导航指令
    instruction = "The building between the Bin Brook and the St John's College Library that is wider than the other buildings around the courtyard."
    prompt = create_prompt(instruction)
    response = gpt_api_call(prompt)
    structured_data = parse_response(response)

    # 将结构化数据保存为 JSON 文件
    with open("structured_data.json", "w") as json_file:
        json.dump(structured_data, json_file, indent=2)

    # 打印结构化数据
    print(json.dumps(structured_data, indent=2))
