import numpy as np

def get_map_objects(size=10):
    height = size
    width = size * 2
    radius = height / 10
    objects = {
        'walls': [
            [(0, 0), (width, 0)],
            [(0, height), (width, height)],
            [(0, 0), (0, height)],
            [(width, 0), (width, height)],
            [(.6*height, 0), (height, height / 2)],
            [(1.1*height, 0), (height, height / 2)],
            [(1.8*height, height), (width, .8*height)],
            [(.8*height, height), (.8*height, .9*height)],
            [(1.2*height, height), (1.2*height, .9*height)],
            [(.8*height, .9*height), (1.2*height, .9*height)],
            [(.1*height, 0), (.1*height, .05*height)],
            [(.1*height, .05*height), (.05*height, .1*height)],
            [(0, .1*height), (.05*height, .1*height)],
        ],
        'circles': [
            {'center': (radius * 2, height - radius * 2), 'radius': radius*1.3},
            {'center': (width - radius * 2, radius * 2), 'radius': radius}
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