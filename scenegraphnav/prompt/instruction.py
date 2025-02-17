import os
import json
from openai import OpenAI

# 构造 Prompt
def create_prompt(instruction):
    return """You are a language navigation assistant. Your task is to analyze complex navigation instructions and extract the following structured information in JSON format:
- Target: The main object or location to be navigated to. And list its attributes such as color, shape, etc.
- Landmarks: Any referenced Geosptial names that help identify the target's position. Note that landmarks are usually capitalized names of streets, roads, etc. List the relationships between the target and landmarks, and mention the secondary landmarks if the target near the intersection of two landmarks.
- Surrounding: Any referenced objects, environmental or contextual information not part of the main landmarks but provides additional clues.
- Spatial Relationships with objects: The spatial and positional relationships between the target and landmarks.

Provide your answer in JSON format.
{
  "Target": {
    "object": "class_name",
    "attribute":{
        "attribute 1": "value 1",
        "attribute 2": "value 2",
        ...
    }
  },
  "Relationships with Landmarks": {
    "intersection": ["Landmark 1", "Landmark 2",...], # return none if no intersection
    "Landmark":{
        "Landmark 1": "relationship",
        "Landmark 2": "relationship,
       ... 
    }
  },
  "Surrounding": ["object 1", "object 2", ...],
  "Spatial_Relationships with objects": [
    "the relationship between target and objects",
    "the relationship between object 1 and object 2",
    "the relationship between object 2 and object 3",
    ...
  ]
}


Example:
Instruction: "A white car behind a black car, with a black car across from it on the opposite side of Willmore Road facing the edge of the map, in between two identical multi-housing units."
Extracted:
{
  "Target": {
    "object": "car",
    "attribute":{
        "color": "white",
    }
  },
  "Relationships with Landmarks": {
    "intersection": null, # return null if no intersection
    "Landmark":{
            "Willmore Road": "on the Willmore Road",
        }
  },
  "Surrounding": ["black car", "multi-housing units"],
  "Spatial_Relationships with objects": [
    "white car is behind black car",
    "black car is across from white car on the opposite side of Willmore Road",
    "black car is facing the edge of the map",
    "white car is in between two identical multi-housing units"
  ]
}

Instruction: "A dark blue car near the corner of Beche Road and Priory Road next to the Cellarer's Chequer building"
{
  "Target": {
    "object": "car",
    "attribute":{
        "color": "dark blue",
    }
  },
  "Relationships with Landmarks": {
    "intersection": ["Beche Road", "Priory Road"], # return none if no intersection
    "Landmark":{
        "Beche Road": "near the Beche Road",
        "Priory Road": "near the Priory Road",
        "Cellarer's Chequer": "next to the Cellarer's Chequer building"
    }
  },
  "Surrounding": ["buildings", "streets"],
  "Spatial_Relationships with objects": [
    "The dark blue car is next to the Cellarer's Chequer building"
  ]
}
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
        model="qwen-max",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # 设置为 0.0 以确保解析的稳定性
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
