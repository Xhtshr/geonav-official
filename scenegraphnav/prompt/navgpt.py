TASK_DESCRIPTION_PROMPT = """You are an intelligent agent controlling a UAV to make zero-shot visual-object navigation in a city environment. Following the text instruction, your goal is to explore and reach the described target based on the topdown RGB images.
Target Instruction: {instruction} Analyze the instruction to identify key landmarks, objects that will guide your navigation.
While navigating, your current relationship with landmark is: You are {geoinstruct}.
Important: Use the instruction and your observations to plan step by step.
"""
TASK_DESCRIPTION_SHORT ="""You are an intelligent agent controlling a UAV to make zero-shot visual-object navigation in a city environment. Following the text instruction, your goal is to Locate the described target based on the topdown RGB images.
Target Instruction: {instruction}"""

OBSERVATION_CAPTION = """Image shows an aerial view of the urban area with many objects. detect all the objects in the image, return bounding boxes for all of them using the following format: [{
        "object": "object_name",
        "Color": "object_color",
        "Shape": "object_shape",
        "Descriprtion": "object_description"
     }, ...]
"""

OBJECT_GROUNDING = """You should detect all the objects in the list: {surroundings} from the image, return position points for all of them using the following format: 
    ``json
    {
        {
        "object": "object_type",
        "object": "id" (from 1 to the total number of surroundings)
        "pos": [x, y]
        }, 
        ...
    }
"""

#利用原生的Visual Grounding 能力
OBSERVATION_SUMMARY = """Image shows an aerial view of your observation may contain a view of the target and nearby surrounding objects. Observe the current image in detail, and find the mentioned target in the Instruction.
Your position is {pos} and you are the center of the view. The length and width of the view are {shape}, translated into the [(x_min,y_min), (x_max,y_max)] of the real world coordination are {area}. 
If u find this target, locate the position points (real world) of the target. Else if your don't find this, you should explore more. You response should consist three types of content: **decision**, **selected_pos**, and **navigation_recommendation**.
Note: *selected_pos* is based on the real world coordination, *navigation_recommendation* choose from "North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest", or "Stay"
JSON does not support annotations! Please output the json in legal format. 
**Required Output Format**  
    ```json
    {{
    "decision": "Explore|Locate",
    "selected_pos": [x,y],
    "navigation_recommendation": "Move [direction]"
    }}```
Now, output:"""

#利用原生的多图记忆能力
MULTI_OBSERVATION_SUMMARY = """These Image shows sliding windows of aerial view. They are continuous shooting with drones and may contain the target and nearby surrounding objects. By thinking the task and understand these images, find the mentioned target.

In the last image, your real world position is {pos} and you are the center of the view. The length and width of any image are {shape}, translated into the [(x_min,y_min), (x_max,y_max)] of the real world coordination are {area}. 
First, if u compare these images
If u find this target, locate the position points (real world) of the target according to the last image. Else if your don't find this, you should reply explore. You response should consist three types of content: **decision**, **selected_pos**, and **recommendation**.
recommendation shortly describes the conclusion of the current observation and the next plan should take.
**Required Output Format**  
    ```json
    {{
    "decision": "Locate|Stop",
    "selected_pos": [x,y], # within {area}
    "recommendation": ...
    }}```"""

TASK_DESCRIPTION_FORMAP = """You are controlling a UAV to make visual-object navigation in a city environment. Target Instruction: {instruction} Following the instruction, your goal is to explore and reach the described target based on the map.
Analyze the instruction to identify key landmarks, objects that will guide your navigation.
Important: Use the instruction and your observations to plan step by step.
"""

#利用Visual-based map的geo reasoning能力
MAP_SUMMARY = """The map shows a simple drawing of the urban district with various objects. It includes three layers:
1. **Landmarks**: Landmarks are outlined in gray, with different shades of gray representing different regions. Their names are annotated at the region centroids.
2. **Objects**: Represented by colored polygons, such as a red squre for a car, Brown for a building, or a green circle for a tree and so on.
3. **Explored Area**: The lavender shading represents the explored area by the field of camera view, and the white area is unexplored.
And **Path**: The path is represented by a line with arrows. The path starts from the UAV's initial position and ends at the current pose.
You response should consist two types of content: **thought** and **answer**.
<question>
The target is not in the current view or not sure about it. You need to use this map to reason where the target is.  tell me where you are and what direction your are facing.
You can choose two decision: Explore or Exploit. If u have not find the target, you should following the instrution and search this area. If you believe the target is under the current view, return Exploit to use tools to locate target. Else if your don't reach this target, Explore more. 
<question>

<thought>
\put your thoughts here
<thought>

<answer>
You answer should consist three types of content: **decision**, **reason**, and **navigation_recommendation**.
**Required Output Format**  
    ```json
    {{
    "decision": "Explore|Exploit",
    "reason": "...",
    "navigation_recommendation": "Move [direction]"
    }}```
<answer>

"""

LANDMARK_DESCRIPTION = """{landmark} """

PLANNER_PROMPTV2 = """ 
You are a planner for a UAV navigation system to locate a target. The target description is {instruction}. You are {geoinstruct}. Answer the following questions:
<question>
The strategy list to achieve the subgoal is ['Navigate', 'Search', 'Locate']. If you are close to the landmark, you should move to search the area according to the described spatial relationship. If you are not close to the landmark, you should navigate near the landmark.
Please decompose task into detailed subgoals and output the result in a structured JSON format.
<question>

<thought>
\put your thoughts here
<thought>

<answer>
Your answer should be in json format ,consist two types of content: **reason**, and **movement**. Subgoals should be precise and clear, and avoid vagueness and lengthy words.
**Required Output Format**
```json
{{
  "plan": "<Overall plan description, explaining the overall task goal and key steps>",
  "sub_goals": [
    {{
      "goal": "<Description of sub-goal 1>",
      "desired_state": "<Expected state after achieving sub-goal 1>",
      "strategy": "<The strategy employed to achieve sub-goal 1>"# choose from "Navigate", "Search", "Locate"
    }},
    {{
      "goal": "<Description of sub-goal 2>",
      "desired_state": "<Expected state after achieving sub-goal 2>",
      "strategy": "<The strategy employed to achieve sub-goal 2>"
    }},
    ... // Additional sub-goals can be added as needed based on the task.
  ]
}}```
<answer>

"""

PLANNER_PROMPT = """ You are a planner for a UAV navigation system to locate a target. Given a task, you can choose two decision: Navigate or Search. First, you should navigate near the landmark. Then, you should search the area according to the described spatial relationship.
To achieve this goal, you can use the map.
The map shows a simple drawing of the urban district, the map will be updated will flying. It includes three layers:
1. Landmarks: Outlined in gray. The names are annotated at their region centroids.
2. Objects: Represented by colored polygons, such as a red squre for a car, Brown for a building, or a green circle for a tree and so on.
3. Explored Area: The lavender shading represents the explored area, and the white area is unexplored.
And the path is represented by a line with arrows.
<question>
The target description is {instruction}. The task have a 20-step limitation, you have used {current_step} steps. 
If u are close to the landmark, you should move to search the area according to the described spatial relationship. If you are not close to the landmark, you should navigate near the landmark.
The reasoned area that need to search should be reasoned from the instruction and the map. 
<question>

response should consist two types of content: **thought** and **answer**.
<thought>
\put your thoughts here
<thought>

<answer>
Your answer should be in json format ,consist three types of content: **decision**, **reason**, and **movement**.
**Required Output Format**  
    ```json
    {{
    "decision": "Navigate|Search",
    "reason": "...",
    "movement": "Move [direction]" # choose from "Stay", "North", "Northeast","South","Southeast", ...
    }}```
<answer>
"""

ACTION_PROMPT = """Analyze the observation and history to select the **next discrete action** that will guide the UAV closer to its target.
Your position is in the center of the image. Since you are at a high altitude, each of your movements has little impact on the overall image.
When performing actions:
1. Ensure that the UAV's flight path stays **within the map's valid boundaries** and does not cross into black regions.
2. Black regions represent **unsafe zones** or areas **outside the designated map** that the UAV should avoid.
Input:
- {history}  
- {observation}  
- Available Actions: STOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, GO_UP, GO_DOWN
Thought:
Explain the reasoning behind your action(s)

Output:
- Final Action: Return the action(s) in the format ['DiscreteAction.ACTION_NAME'] (for single or multiple actions).

----
### Example 1
----
History: The UAV started near an open plaza and moved forward to observe a small red building.  
Observation: The UAV now sees a green park to the left and a curved road ahead.  
Thought: Based on the observation, the curved road aligns with the described path to the target... Moving forward along this road is the best action.  
Final Action: ['DiscreteAction.MOVE_FORWARD']  
### Example 2
----
History: The UAV is navigating along a straight path and observes an intersection ahead.  
Observation: A tall building is visible on the right, which aligns with the target location described in the instruction.  
Thought: The target's description matches the tall building on the right... The best action is to turn toward it.  
Final Action: ['DiscreteAction.TURN_RIGHT']  
----  
"""