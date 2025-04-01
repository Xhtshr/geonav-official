import os
import json
from openai import OpenAI

# extract spatial relation like 'left of', 'right of', ''
SPATIAL_RELATION_PROMPT = ''

#导入openai key配置
os.environ["OPENAI_API_KEY"] = "sk-8xBWP046CnOzBAEaC262872c0f4d40EeAc366eB651B7C020" # 3.5--1美元
# os.environ["OPENAI_API_KEY"] = "sk-dooWu6cCsNTtSsB7Fb5f2f25Cd164b67A94cFd650442EcB2" # 4o--71美元
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
# 设置longchain的基础环境变量
os.environ["OPENAI_API_BASE"] = 'https://xiaoai.plus/v1'
client = OpenAI(
    # 下面两个参数的默认值来自环境变量，可以不加
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

# 构造 Prompt
def create_prompt(instruction):
    return f"""
You are a language navigation assistant. Your task is to analyze complex navigation instructions and extract the following structured information:
1. **Target**: The main object or location to be navigated to.
2. **Landmarks**: Any referenced objects, locations, or features that help identify the target's position.
3. **Surrounding**: Any environmental or contextual information not part of the main landmarks but provides additional clues.
4. **Spatial Relationships**: The spatial and positional relationships between the target and landmarks.

Provide your answer in the following structured format:
- Target: [description]
- Landmarks: [list of landmark descriptions]
- Surrounding: [list of surrounding descriptions or "None"]
- Spatial Relationships:
  - [Target] is [relationship] [Landmark].

Example:
Instruction: "The strip of long brown rectangular roofed building to the left of the Buckingham Room building, right of the 3 columns of parking space. The top one between the two chimneys."
Extracted:
- Target: "strip of long brown rectangular roofed building"
- Landmarks: ["Buckingham Room building", "3 columns of parking space", "two chimneys"]
- Surrounding: None
- Spatial Relationships:
  - "strip of long brown rectangular roofed building" is left of "Buckingham Room building".
  - "strip of long brown rectangular roofed building" is right of "3 columns of parking space".
  - "strip of long brown rectangular roofed building" is between "two chimneys".

Now process the following instruction:
{instruction}
"""

# 调用 OpenAI GPT 模型
def gpt_api_call(prompt):
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.0  # 设置为 0.0 以确保解析的稳定性
  )
  return response.choices[0].message.content

# 解析 GPT 返回的结构化文本
def parse_response(response):
  lines = response.strip().split("\n")
  result = {
      "Target": lines[0].split(": ", 1)[1].strip('"'),
      "Landmarks": json.loads(lines[1].split(": ", 1)[1]),
      "Surrounding": None if lines[2].split(": ", 1)[1] == "None" else json.loads(lines[2].split(": ", 1)[1]),
      "Spatial_Relationships": [line.strip("- ").strip() for line in lines[3:]],
  }
  return result

# 主流程
if __name__ == "__main__":
  # 示例导航指令
  instruction = (
      "The strip of long brown rectangular roofed building to the left of the Buckingham Room building, "
      "right of the 3 columns of parking space. The top one between the two chimneys."
  )
  prompt = create_prompt(instruction)
  response = gpt_api_call(prompt)
  structured_data = parse_response(response)

  # 打印结构化数据
  print(json.dumps(structured_data, indent=2))