import numpy as np

KERNELS = {
    "edge_horizontal": np.array([[-1,-1,-1],
                                 [ 0, 0, 0],
                                 [ 1, 1, 1]]),
    "edge_vertical": np.array([[-1,0,1],
                               [-1,0,1],
                               [-1,0,1]]),
    "sharpen": np.array([[ 0,-1, 0],
                         [-1, 5,-1],
                         [ 0,-1, 0]]),
    "blur": (1/9)*np.ones((3,3))
}
