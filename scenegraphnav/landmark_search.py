import Levenshtein
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.cityreferobject import get_landmarks, remove_duplicate_landmarks_by_area

def landmark_loc(map_name: str, query_names: list[str], default_pose: Pose4D = None) -> Point2D:
    if not query_names:
        return default_pose.xy
    landmarks_cache = remove_duplicate_landmarks_by_area(get_landmarks())
    landmarks = landmarks_cache[map_name].values()
    query_landmarks = [
            min(landmarks, key=lambda lm, q=query: Levenshtein.distance(lm.name, q))
            for query in query_names
        ]
    return Point2D(query_landmarks[0].position.x, query_landmarks[0].position.y) # TODO: decide where to search?