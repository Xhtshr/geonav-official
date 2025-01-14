import re
import math
import json


class OneStagePromptManager:
    def __init__(self, task_description, instruction):
        self.task_description = task_description
        self.instruction = instruction
        self.history = []
        self.current_observation = None
        self.action_space = []
        self.current_position = None
        self.target_position = None

    def generate_prompt(self):
        prompt = {
            "Task Description": self.task_description,
            "Instruction": self.instruction,
            "History": self.history,
            "Current Observation": self.current_observation,
            "Action Space": self.action_space,
            "Current Position": self.current_position,
            "Target Position": self.target_position
        }
        return json.dumps(prompt, indent=4)

    def update_history(self, action, observation):
        self.history.append((action, observation))

    def update_current_observation(self, observation):
        self.current_observation = observation

    def update_action_space(self, action_space):
        self.action_space = action_space

    def update_current_position(self, position):
        self.current_position = position

    def update_target_position(self, position):
        self.target_position = position

# 示例用法
task_description = "You are an embodied robot that navigates in a continuous environment."
instruction = "Go to the bedroom with a blue door and stop."

manager = OneStagePromptManager(task_description, instruction)

# 更新当前观察
manager.update_current_observation("You are in a hallway with a blue door.")

# 更新动作空间
manager.update_action_space([
    "A. stop",
    "B. move forward",
    "C. turn left",
    "D. turn right"
])

# 更新当前位置
manager.update_current_position("Hallway")

# 更新目标位置
manager.update_target_position("Bedroom")

# 更新历史信息
manager.update_history("B. move forward", "You are in a hallway with a blue door.")

# 生成提示
prompt = manager.generate_prompt()
print(prompt)