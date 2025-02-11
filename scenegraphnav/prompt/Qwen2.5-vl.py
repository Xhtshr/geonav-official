TASK_DESCRIPTION_PROMPT = """You are an intelligent agent controlling a UAV to make zero-shot visual-object navigation in a city environment. Following the text instruction, your goal is to find and reach the described target based on the topdown RGB images.  
First, analyze the instruction to identify key landmarks, objects and features that will guide your navigation.
While navigating, you can observe the environment.
Instruction: {instruction}  
Important: Use the instruction and observations to plan and adapt your navigation strategy step by step.
"""


OBSERVATION_CAPTION = """Image shows an aerial view of the urban area with many objects. detect all the objects in the image, return bounding boxes for all of them using the following format: [{
        "object": "object_name",
        "Color": "object_color",
        "Shape": "object_shape",
        "Descriprtion": "object_description"
     }, ...]
"""

OBJECT_GROUNDING = """detect all the objects in the image, return bounding boxes for all of them using the following format: 
    [{
        
        "object": "object_name",
        "bboxes": [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
    }, ...]
"""


OBSERVATION_SUMMARY = """Image shows an aerial view of the surrounding area with a clear view of the target and nearby surrounding objects. Observe the current image in detail, and find the mentioned things in the Instruction. Focusing on key features of the environment, for example:  
- Colors and textures (e.g., roof, road).  
- Shapes and structures (e.g., building, street).  
- Landmarks and unique details (e.g., trees, parking lot, cars).  
Describe the key findings in the image that can help you navigate toward the target. 
Observation: 
"""

HISTORY_PROMPT = """You are an agent navigating above the city.  

At each step, update the navigation history with your actions and observations. Focus on:  
1. Linking the current observation to the previous steps.  
2. Noting any progress toward the target (or deviations).  
3. Be careful not to get stuck in an endless loop, for example, repeat the same actions with no reason.
Previous History: {history}  
Previous Action: {previous_action}  
Current Observation: {observation}  

Updating and summarize the navigation history, including:  
- What has changed?  
- Any new features observed?  
- Progress made toward the target?  

Now, Output the updated history:"""

PLANNER_PROMPT = """Based on the instruction: {instruction}, generate a detailed and executable action plan.  

Steps should include:  
1. Identifying key visual landmarks or features from the instruction.  
2. Mapping these features to possible discrete actions (e.g., MOVE_FORWARD, TURN_LEFT).  
3. Sequencing actions logically to guide the UAV toward the target.  

Action Plan:  
1. Step 1: Identify {first_feature}, then {corresponding_action}.  
2. Step 2: Use {next_feature} to guide {corresponding_action}.  
3. Step 3: ...  
"""

ACTION_PROMPT = """You are an intelligent agent controlling a UAV to navigate in a city environment.  
At each step, you must analyze the observation and history to select the **next discrete action** that will guide the UAV closer to its target.
Your position is in the center of the image. Since you are at a high altitude, each of your movements has little impact on the overall image.
When performing actions:
1. Ensure that the UAV's flight path stays **within the map's valid boundaries** and does not cross into black regions.
2. Black regions represent **unsafe zones** or areas **outside the designated map** that the UAV should avoid.
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
Thought: Based on the observation, I notice key features of the observation. To move closer to the target or complete the next step, the best action(s) is/are...  
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