import numpy as np

def get_map_objects(size=80):
    height = size
    width = size * 2
    radius = height / 10
    objects = {
        'walls': [
            # Outer walls
            [(0, 0), (width, 0)],                      # Bottom wall
            [(0, height), (width, height)],            # Top wall
            [(0, 0), (0, height)],                     # Left wall
            [(width, 0), (width, height)],             # Right wall
            
            # Interior room divisions
            [(0.25*width, 0), (0.25*width, 0.5*height)],              # Living room left wall
            [(0.25*width, 0.625*height), (0.25*width, height)],       # Living room left upper wall
            [(0.75*width, 0.75*height), (0.75*width, height)],        # Kitchen left upper wall
            [(1.25*width, 0.75*height), (1.25*width, height)],        # Bedroom left upper wall
            [(1.75*width, 0.25*height), (1.75*width, 0.5*height)],    # Bathroom left lower wall
            [(1.75*width, 0.625*height), (1.75*width, height)],       # Bathroom left upper wall
            
            # Hallway walls
            [(0.25*width, 0.5*height), (0.75*width, 0.5*height)],     # Hallway top left wall
            [(0.75*width, 0.625*height), (1.25*width, 0.625*height)], # Hallway top middle wall
            [(1.25*width, 0.5*height), (1.75*width, 0.5*height)],     # Hallway top right wall
        ],
        'circles': [
            {'center': (0.125*width, 0.125*height), 'radius': radius},    # Bottom left
            {'center': (0.625*width, 0.125*height), 'radius': radius},    # Bottom center
            {'center': (1.875*width, 0.125*height), 'radius': radius},    # Bottom right
            {'center': (0.125*width, 0.875*height), 'radius': radius},    # Top left
            {'center': (0.625*width, 0.875*height), 'radius': radius},    # Top center
            {'center': (1.875*width, 0.875*height), 'radius': radius},    # Top right
        ]
    }
    return objects

def create_robot(x, y, orientation, size=1):
    angle = orientation * (3.14159 / 180)
    octagon = [
        (x + size * 0.707 * np.cos(angle + i * 3.14159 / 4), y + size * 0.707 * np.sin(angle + i * 3.14159 / 4))
        for i in range(8)
    ]
    octagon.append(octagon[0])
    front_x = x + size * .626 * np.cos(angle)
    front_y = y + size * .626 * np.sin(angle)
    lidar = [(x, y), (front_x, front_y)]
    return octagon, lidar