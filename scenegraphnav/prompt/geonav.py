GOAL_DESCRIPTION_NAV = """You are controlling an UAV to navigate in a city environment. Your task is to navigate to a specific landmark using a top-down sketch. 
The map provides a simplified representation of the urban district, including a landmark layer and a path represented by a line with the arrow. Landmarks are outlined in gray and their names are annotated at the region centroids.
Important: Use your prior knowledge and the map to plan your steps systematically.
"""

GOAL_DESCRIPTION_SEA = """You are controlling an UAV to navigate in a city environment. Your task is to search a specific object nearby the landmark using a top-down map. 
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
You are currently {geoinstruct}. Your assigned goal is: {goal} Your desired state is: {state}
Based on the top-down map, determine the direction you need to move to achieve the goal. If there are multiple landmarks and their relative directions are inconsistent, the image should prevail.
<question>

Your response should include **answer**.
<answer>
Do not put thought here, your answer only include two components: **reason** and **movement**.
**Required Output Format**:
    ```json
    {{
        "reason": "Explain your reasoning here, no longer than 30 words.",
        "movement": "Move [northwest|northeast|southwest|southeast|north|south|east|west]"
    }}```
<answer>
"""

LANDMARK_NAVIGATION_PROMPT = """
Answer the following question:
<question>
Your assigned goal is: {goal}
Based on the top-down map with containing landmarks, determine the direction you need to move to achieve the goal. If there are multiple landmarks and their relative directions are inconsistent, the image should prevail.
<question>

Your response should include **answer**.
<answer>
Do not put thought here, your answer only include two components: **reason** and **movement**.
**Required Output Format**:
    ```json
    {{
        "reason": "Explain your reasoning here, no longer than 30 words.",
        "movement": "Move [northwest|northeast|southwest|southeast|north|south|east|west]"
    }}```
<answer>
"""

OBJECT_SEARCH_PROMPT = """
Answer the following question:
<question>
You are currently near the city landmarks and should search the area according to the novelty and attractiveness. You should consider the unexplored area and the objects that are expected to be observed.
Assigned goal is: {goal}. Desired state is: {state}.
Based on the top-down map, determine the direction you need to move.
<question>

Your response should include: **answer**.

<answer>
Do not put thought here, your answer only include two components: **reason** and **movement**.
**Required Output Format**:
    ```json
    {{
        "reason": "Explain your reasoning here.",
        "movement": "Move [northwest|northeast|southwest|southeast|north|south|east|west]"
    }}```
<answer>"""


TARGET_LOCATE_PROMPT = """
Your current position is {pos}, which is at the center of your view. The area of your view are {area}, corresponding to the real-world coordinates [(x_min, y_min), (x_max, y_max)].
<question>
Your assigned goal is: {goal}. You are currently flying above the target and need to locate the specific object based on your observation. 
If you identify the target, determine its position in real-world coordinates. Answer the following question:
<question>

Your response should include: **answer**.

<answer>
Your answer should include two components: **reason** and **selected_pos**. [x, y] Coordinates must be within {area}.
**Required Output Format**:
    ```json
    {{
        "reason": "Explain your reasoning here.",
        "selected_pos": [x, y] 
    }}```
<answer>"""

LOCAL_GRAPH_PROMPT = LOCAL_GRAPH_PROMPT = """
Your task is to recognize {objects} and their relationships into a structured JSON graph following these strict rules:

**IMPORTANT FORMATTING RULES**
- Return ONLY valid JSON - no extra text, explanations, or markdown formatting
- Do NOT include any comments (lines starting with //)
- Do NOT include trailing commas after the last element in arrays or objects
- All keys and string values must be enclosed in double quotes
- Ensure all braces and brackets are properly closed

**Node Requirements**
1. Each node must have a unique [id].
2. Mandatory attributes for every node:
- [object_type]: one of ["vehicle", "road", "building", "parking_lot", "green_space", etc]
- [bbox]: bounding box coordinates [xmin, ymin, xmax, ymax]
3. Optional attribute (only if clearly observable):
- `color`: one of ["white", "black", "red", "gray", "blue", "green", "brown", "silver", ...]

**Edge Requirements**
Only use the following relationship labels, with these meanings:
- **Topological**: 
    - "contains" (one object is completely within another)
    - "overlaps" (objects partially cover each other)
    - "separates" (a certain distance between two objects)

- **Directional** (absolute, from aerial perspective):
    - Primary: "north_of", "south_of", "east_of", "west_of"
    - Diagonal: "northeast_of", "northwest_of", "southeast_of", "southwest_of"
For two nodes, both topological and directional relationships can be established simultaneously. For example, the relationship between a car and a building can be either "separates" or "north_of".

**Special Cases & Handling of Ambiguities**
1. **Preset Landmarks**:
- Names like "Leslie Road", "Bridgelands Way", "Livingstone Road", etc., are considered preset. Do not extract these from the image; focus solely on dynamic objects and visible spatial relations.
2. **Ambiguity Reduction**:
- Limit your relationship predicate set to the ones provided. This finite vocabulary helps eliminate ambiguity and ensures consistent mapping from natural language descriptions to spatial relationships.
3. **Hierarchical and Iterative Extraction**:
- First, build an initial graph based on absolute spatial cues (from the north-aligned image).
- Then, refine relationships using explicit ordering and structural cues from the description.

<Example Output>
For "white car parked 1st from bottom in right column":
{{
"nodes": [
    {{
    "id": "White01", 
    "object_type": "vehicle",
    "color": "white",
    "bbox": [320, 580, 360, 620]
    }},
    {{
    "id": "ParkingLot07",
    "object_type": "parking_lot",
    "bbox": [300, 500, 700, 800]
    }}
],
"edges": [
    {{
    "source": "White01",
    "target": "ParkingLot07",
    "relationship": "contains"
    }}
]
}}
</Example>

CRITICAL REMINDERS:
1. Output ONLY valid JSON without any extra formatting or text
2. Check that all brackets and braces are properly closed
3. Remove any trailing commas
4. Do not include comments in the JSON
5. Ensure all keys and string values are enclosed in double quotes

Now analyze: {objects}.
"""

QUERY_OPERATION_CHAIN_PROMPT  = """
    Converts navigation commands into a chain of query operations. 
    Available operations:
    - get_geonode_by_name(name_pattern): finds geonodes based on a name pattern. If no name is provided, you must return all geonodes.
    - get_child_nodes(parent, relation_type): gets child nodes with the specified relation to the parent.
      Available relation types are: "contains", "overlaps", "separates", "north_of", "south_of", "east_of", "west_of", "northeast_of", "northwest_of", "southeast_of", "southwest_of"
      "contains" means one object is completely within another,"overlaps" means objects partially cover each other,"separates" means no contact between the two objects.
      Only distinguish directions such as "north_of", "south_of", "east_of", "west_of", "northeast_of", "northwest_of", "southeast_of", "southwest_of" when there relationship is "separates"
    - filter_by_class(obj_class): filters object nodes by class. One of ["vehicle", "road", "building", "parking_lot", "green_space", etc]
    - filter_by_attribute(key, value): filters object nodes by attribute.

    Notes:
    1. The description "in front of" usually corresponds to the "north_of" or "separates" relation in north-up maps
    2. The description "behind" usually corresponds to the "separates" or "south_of" relationship in north-up maps
    3. For some expressions with ambiguous semantics, multiple relations can be queried to enhance robustness. For example, for "stop on the road", both "contains" and "overlaps" relations can be queried simultaneously
    4. When describing the relative position of multiple objects, you need to find the relationship chain that connects these objects
    5. When multiple chains of operations represent relative relationships, make sure that the chains of operations are coherent
    
    **Strict output format requirements**: - MUST return a JSON array directly, do not wrap the operation chain with additional keys.
    
    Example instruction: "Locate the brown house"
    return operation chain:
    [
        {{"method": "get_geonode_by_name", "args": [""]}},  // if there is no clear description of the landmark, returns all geographic nodes
        {{"method": "get_child_nodes", "kwargs": {{"relation_type": "contains, overlaps, separates"}}}},  // find objects that have topological relationships with geographic nodes
        {{"method": "filter_by_class", "args": ["building"]}},
        {{"method": "filter_by_attribute", "args": ["color", "brown"]}}
    ]

    Example instruction: "Find the red car in the north of the main entrance of the shopping mall"
    return operation chain:
    [
        {{"method": "get_geonode_by_name", "args": ["shopping mall"]}},
        {{"method": "get_child_nodes", "kwargs": {{"relation_type": "north_of"}}}},
        {{"method": "filter_by_class", "args": ["vehicle"]}},
        {{"method": "filter_by_attribute", "args": ["color", "red"]}}
    ]

    Example instruction: "Find the car next to the park"
    return operation chain:
    [
        {{"method": "get_geonode_by_name", "args": ["park"]}},
        {{"method": "get_child_nodes", "kwargs": {{"relation_type": "separates"}}}},  // "next to" can correspond to either "separates"
        {{"method": "filter_by_class", "args": ["car"]}}
    ]
    
    Example instruction: "This is a white car parked on Davey Road. There is a gray car parked in front of it facing the opposite direction."
    return operation chain:
    [
        {{"method": "get_geonode_by_name", "args": ["Davey Road"]}},
        {{"method": "get_child_nodes", "kwargs": {{"relation_type": "contains, overlaps"}}}},  // "parked on" can correspond to either "contains" or "overlaps"
        {{"method": "filter_by_class", "args": ["vehicle"]}},
        {{"method": "filter_by_attribute", "args": ["color", "white"]}}
    ]
    
    Current instruction: {instruction}.
    Please output the chain of operations in JSON format:
    """


QUERY_OPERATION_PROMPT  = """
    Based on the current nodes: {node_text}

    Generate a new operation chain to further refine the search for the target object.
    Available operations:
    - get_child_nodes(parent, relation_type): Gets the child nodes with the specified relationship to the parent node.
    Available relation types are: "contains", "adjacent_to", "near_corner", "north_of", "south_of", "east_of", "west_of", "northeast_of", "northwest_of", "southeast_of", "southwest_of"
    - filter_by_class(obj_class): Filter object nodes by class.
    - filter_by_attribute(key, value): Filter object nodes by attribute.

    Return the operation chain in JSON format.
    """