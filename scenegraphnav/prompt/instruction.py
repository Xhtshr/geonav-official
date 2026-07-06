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

# 调用 OpenAI GPT 模型
def gpt_api_call(prompt):
    # 导入openai key配置
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENAI_BASE_URL"] = ""

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    response = client.chat.completions.create(
        model="qwen-max-latest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
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
