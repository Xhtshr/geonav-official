TASK_DESCRIPTION_PROMPT = """You are an intelligent agent controlling a UAV to navigate in a city environment. Your goal is to find and reach a target object based on the given instruction about its location and visual features.  
While navigating, you can only observe the environment through a 10x10 meter RGB view.  

Instruction: {instruction}  
Important: Use the instruction and observations to plan and adapt your navigation strategy step by step.
"""


OBSERVATION_SUMMARY = """Summarize the current observation in detail, focusing on key features of the environment:  
- Colors and textures (e.g., red roof, concrete road).  
- Shapes and structures (e.g., square building, curved street).  
- Landmarks and unique details (e.g., tall tree, parking lot with blue cars).  

Use this summary to update the navigation history and align it with the target description.  

Observation: {observation}  
Key Features:  
1. Feature_1  
2. Feature_2  
...
"""

HISTORY_PROMPT = """You are an agent navigating above the city.  

At each step, update the navigation history with your actions and observations. Focus on:  
1. Linking the current observation to the previous steps.  
2. Noting any progress toward the target (or deviations).  

History: {history}  
Previous Action: {previous_action}  
Current Observation: {observation}  

Update the navigation history:  
- What has changed?  
- Any new features observed?  
- Progress made toward the target?  

Updated History:"""

PLANNER_PROMPT = """Based on the instruction: {instruction}, generate a detailed and executable action plan.  

Steps should include:  
1. Identifying key visual landmarks or features from the instruction.  
2. Mapping these features to possible discrete actions (e.g., MOVE_FORWARD, TURN_LEFT).  
3. Sequencing actions logically to guide the UAV toward the target.  

Action Plan:  
1. Step 1: Identify {first_feature}, then {corresponding_action}.  
2. Step 2: Use {next_feature} to guide {corresponding_action}.  
3. Step 3: ...  
...
"""

ACTION_PROMPT = """You are an intelligent agent controlling a UAV to navigate in a city environment.  

At each step, you must analyze the observation and history to select the **next discrete action** that will guide the UAV closer to its target.

Input:
- History: {history}
- Observation: {observation}
- Available Actions: STOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, GO_UP, GO_DOWN

Task:
1. Evaluate the history and observation to understand the current context.  
2. Decide on the most appropriate discrete action to take from the available actions.  
3. Optionally, you can generate a sequence of actions if necessary for complex scenarios.  

Output:
- Thought: Explain the reasoning behind your action(s).  
- Final Action: Return the action(s) in the format ['DiscreteAction.ACTION_NAME'] (for single or multiple actions).

----  
History: {history}  
Observation: {observation}  
Thought: Based on the observation, I notice {key_feature}. To move closer to the target or complete the next step, the best action(s) is/are...  
Final Action: ['DiscreteAction.ACTION_NAME']  
----

### Example 1
----
History: The UAV started near an open plaza and moved forward to observe a small red building.  
Observation: The UAV now sees a green park to the left and a curved road ahead.  
Thought: Based on the observation, the curved road aligns with the described path to the target. Moving forward along this road is the best action.  
Final Action: ['DiscreteAction.MOVE_FORWARD']  

### Example 2
----
History: The UAV is navigating along a straight path and observes an intersection ahead.  
Observation: A tall building is visible on the right, which aligns with the target location described in the instruction.  
Thought: The target's description matches the tall building on the right. The best action is to turn toward it.  
Final Action: ['DiscreteAction.TURN_RIGHT']  

### Example 3
----
History: The UAV has moved upward to gain visibility over a dense urban area.  
Observation: The UAV sees a tall tower directly ahead, which matches the instruction's description of the target.  
Thought: The target is now directly visible. Moving forward will bring the UAV closer to it.  
Final Action: ['DiscreteAction.MOVE_FORWARD']  

### Example 4
----
History: The UAV started near ground level and moved toward an open area.  
Observation: The UAV sees a hilltop that aligns with the instruction's description of the target being at a higher elevation.  
Thought: The target is on a higher elevation, and the next steps require gaining altitude and moving forward for better visibility.  
Final Action: ['DiscreteAction.GO_UP', 'DiscreteAction.MOVE_FORWARD']  

"""