TASK_DESCRIPTION_PROMPT = """You are an intelligent agent controlling a UAV to make zero-shot visual-object navigation in a city environment. Following the text instruction, your goal is to explore and reach the described target based on the topdown RGB images.
Target Instruction: {instruction} Analyze the instruction to identify key landmarks, objects that will guide your navigation.
While navigating, your current relationship with landmark is: You are {surroundings}
Important: Use the instruction and your observations to plan step by step.
"""


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

**Required Output Format**  
    ```json
    {{
    "decision": "Explore|Locate",
    "selected_pos": [x,y],
    "navigation_recommendation": "Move [direction]"
    }}```"""


#利用原生的记忆能力
HISTORY_PROMPT = """You are the memory module of a navigation agent with self-reflective capabilities. The mission is ongoing, and your role is to update, analyze, and summarize the navigation history with clear context and temporal structure.
Update the navigation history using the following inputs:
- History (with timestamps/step numbers): {history}
- Previous Action: {previous_action}

Focus on:
1. Linking the current observation with previous events in a structured, temporal manner.
2. Analyzing the causal relationship: How did the previous action influence the current state? What new observations support progress or indicate deviations?
3. Detecting and flagging any repetitive actions that have led to no progress.
4. Suggesting potential strategy adjustments if repeated patterns or loops are detected.

Summarize the updated history by detailing:
- Changes observed (with reference to time or sequence)
- New features or environmental cues detected
- Progress toward the target and any deviations observed
- Recommendations for future actions to avoid ineffective loops

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