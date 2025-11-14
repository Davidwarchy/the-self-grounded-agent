import numpy as np
from math import sqrt

# These functions are no longer needed for bitmap-based collision
# Keeping file for compatibility if other code imports from here

def line_intersection(p1, p2, p3, p4):
    """Legacy function - not used in bitmap system"""
    return None

def circle_line_intersection(center, radius, p1, p2):
    """Legacy function - not used in bitmap system"""
    return []

def cast_lidar_rays(lidar, room, num_rays=100, ray_length=200):
    """Legacy function - not used in bitmap system"""
    return []