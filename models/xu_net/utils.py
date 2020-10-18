import numpy as np
from scipy import ndimage


def hpf(img):
    kernel_3x3 = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ])
    k3 = ndimage.convolve(img, kernel_3x3)
    return k3