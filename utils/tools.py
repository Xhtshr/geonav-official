import math
from gsamllavanav.space import Point2D

DIRECTION_NAMES = ["East", "Northeast", "North", "Northwest", 
                   "West", "Southwest", "South", "Southeast"]

def get_direction(src: Point2D, target: Point2D):
    dx = target.x - src.x
    dy = target.y - src.y
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return DIRECTION_NAMES[round(angle / 45) % 8]