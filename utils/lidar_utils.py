import numpy as np
from math import sqrt

def line_intersection(p1, p2, p3, p4):
    denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    if denom == 0:
        return None
    ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denom
    ub = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / denom
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return (p1[0] + ua * (p2[0] - p1[0]), p1[1] + ua * (p2[1] - p1[1]))
    return None

def circle_line_intersection(center, radius, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fx = p1[0] - center[0]
    fy = p1[1] - center[1]
    a = dx * dx + dy * dy
    if a == 0:
        return []
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return []
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b + sqrt_discriminant) / (2 * a)
    t2 = (-b - sqrt_discriminant) / (2 * a)
    intersections = []
    if 0 <= t1 <= 1:
        intersections.append((p1[0] + t1 * dx, p1[1] + t1 * dy))
    if 0 <= t2 <= 1:
        intersections.append((p1[0] + t2 * dx, p1[1] + t2 * dy))
    return intersections

def cast_lidar_rays(lidar, room, num_rays=100, ray_length=200):
    lidar_x, lidar_y = lidar[0]
    front_x, front_y = lidar[1]
    delta_y, delta_x = front_y - lidar_y, front_x - lidar_x
    angle = np.arctan2(delta_y, delta_x)
    angles = np.linspace(angle - np.pi / 4, angle + np.pi / 4, num_rays)
    intersections = []
    for a in angles:
        ray_end_x = lidar_x + ray_length * np.cos(a)
        ray_end_y = lidar_y + ray_length * np.sin(a)
        closest_intersection = None
        closest_distance = float('inf')
        for wall in room['walls']:
            intersection = line_intersection((lidar_x, lidar_y), (ray_end_x, ray_end_y), wall[0], wall[1])
            if intersection:
                distance = sqrt((intersection[0] - lidar_x)**2 + (intersection[1] - lidar_y)**2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_intersection = intersection
        for circle in room['circles']:
            center, radius = circle['center'], circle['radius']
            circle_intersections = circle_line_intersection(center, radius, (lidar_x, lidar_y), (ray_end_x, ray_end_y))
            for intersection in circle_intersections:
                distance = sqrt((intersection[0] - lidar_x)**2 + (intersection[1] - lidar_y)**2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_intersection = intersection
        intersections.append(closest_intersection)
    return intersections