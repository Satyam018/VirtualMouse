import math

import  numpy as np

def get_angle(point_a,point_b,point_c):
    x1, y1 = point_a
    x2, y2 = point_b
    x3, y3 = point_c


    AB = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    BC = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    AC = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

    if AB==0 or BC==0:
        return None;

    cos_theta = (AB ** 2 + BC ** 2 - AC ** 2) / (2 * AB * BC)
    cos_theta = max(-1, min(1, cos_theta))
    theta = math.acos(cos_theta)
    return math.degrees(theta)



def get_distance(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    return np.interp(distance,[0,1],[0,1000])