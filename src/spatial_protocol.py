"""One shared, aspect-preserving spatial transform for paired data and cameras."""
import numpy as np


def center_square_box(width, height):
    side = min(width, height)
    left, top = (width-side)//2, (height-side)//2
    return left, top, side


def crop_camera(matrix, left, top):
    return np.array([[1., 0., -left], [0., 1., -top], [0., 0., 1.]]) @ matrix


def center_square(array):
    left, top, side = center_square_box(array.shape[1], array.shape[0])
    return array[top:top+side, left:left+side], (left, top, side)
