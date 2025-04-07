GOAL_DESCRIPTION_NAV = """You are controlling an UAV to navigate in a city environment. Your task is to navigate to a specific location in the city using a top-down map. 
The map provides a simplified representation of the urban district, including a landmark layer and a path represented by a line with the arrow. Landmarks are outlined in gray and their names are annotated at the region centroids.
Important: Use your prior knowledge and the map to plan your steps systematically.
"""

GOAL_DESCRIPTION_SEA = """You are controlling an UAV to navigate in a city environment. Your task is to navigate to a specific location in the city using a top-down map. 
The map provides a simplified representation of the urban district, including:
1. Objects: Represented by colored polygons, such as a red squre for a car, Brown for a building, or a green circle for a tree and so on.
2. Explored Area: The lavender shading represents the explored area, and the white area is unexplored.
3. The path is represented by a line with the arrow.
Important: Use your prior knowledge and the map to plan your steps systematically.
"""

GOAL_DESCRIPTION_LOC = """Your task is to locate the described target based on top-down RGB images. You will be provided two types of information:
1. Target Instruction: A description of the target object that you need to locate.
2. Observation: An aerial view image that may contain the target and nearby surrounding objects.
Use your reasoning and observation skills to identify the target systematically.
"""

LANDMARK_NAVIGATION_PROMPT = """
Answer the following question:
<question>
You are currently {geoinstruct}. Your assigned goal is: {goal}. Your desired state is: {state}.
Based on the top-down map, determine the direction you need to move to achieve the goal.
<question>

<thought>
Provide your reasoning and step-by-step thought process here.
<thought>

<answer>
Do not put thought here, your answer only include two components: **reason** and **movement**.
**Required Output Format**:
    ```json
    {{
        "reason": "Explain your reasoning here.",
        "movement": "Move [northwest|northeast|southwest|southeast|north|south|east|west]"
    }}```
<answer>
"""

OBJECT_SEARCH_PROMPT = """
Answer the following question:
<question>
You are currently near the city landmarks and should search the area according to the described spatial relationship. Your assigned goal is: {goal}. Your desired state is: {state}.
Based on the top-down map, determine the direction you need to move to achieve the goal.
<question>

<thought>
Provide your reasoning and step-by-step thought process here.
<thought>

<answer>
Do not put thought here, your answer only include two components: **reason** and **movement**.
**Required Output Format**:
    ```json
    {{
        "reason": "Explain your reasoning here.",
        "movement": "Move [Move [northwest|northeast|southwest|southeast|north|south|east|west]]"
    }}```
<answer>"""

TARGET_LOCATE_PROMPT = """
Your current position is {pos}, which is at the center of your view. The area of your view are {area}, corresponding to the real-world coordinates [(x_min, y_min), (x_max, y_max)].
<question>
Your assigned goal is: {goal}. You are currently flying above the target and need to locate the specific object based on your observation. 
If you identify the target, determine its position in real-world coordinates. Answer the following question:
<question>

<thought>
Provide your reasoning and a step-by-step explanation of your thought process here.
<thought>

<answer>
Your answer should follow the json format and only includes two components: **reason** and **selected_pos**. [x, y] Coordinates must be within {area}.
**Required Output Format**:
    ```json
    {{
        "reason": "Explain your reasoning here.",
        "selected_pos": [x, y]
    }}```
<answer>"""

QUERY_OPERATION_CHAIN_PROMPT  = """
    Converts navigation commands into a chain of query operations. Available operations:
    - get_geonode_by_name(name_pattern): 根据名称模式查找地理节点。如果不提供名称，则返回所有地理节点。
    - get_child_nodes(parent, relation_type): 获取与父节点具有指定关系的子节点。
    - filter_by_class(obj_class): 按类别过滤物体节点。
    - filter_by_attribute(key, value): 按属性过滤物体节点。

    Example instruction: "Find the red car near the main entrance of the shopping mall"
    return operation chain:
    [
        {{"method": "get_geonode_by_name", "args": ["shopping mall"]}},
        {{"method": "get_child_nodes", "kwargs": {{"relation_type": "near"}}}},
        {{"method": "filter_by_class", "args": ["car"]}},
        {{"method": "filter_by_attribute", "kwargs": {{"color": "red"}}}}
    ]

    Example instruction: "Locate the brown house"
    return operation chain:
    [
        {{"method": "get_geonode_by_name", "args": [""]}},  // 返回所有地理节点
        {{"method": "filter_by_class", "args": ["house"]}},
        {{"method": "filter_by_attribute", "kwargs": {{"color": "brown"}}}}
    ]

    Current instruction: {instruction}
    Please output the chain of operations in JSON format:
    """