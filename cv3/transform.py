import cv2
import numpy as np
from functools import partial

from ._utils import type_decorator, _relative_check
from .utils import rel2abs

__all__ = [
    'vflip',
    'hflip',
    'dflip',
    'transform',
    'rotate',
    'scale',
    'translate',
    'resize',
    'shift',
    'rotate90',
    'rotate180',
    'rotate270',
    'xtranslate',
    'xshift',
    'ytranslate',
    'yshift'
]

@type_decorator
def vflip(img):
    return cv2.flip(img, 0)


@type_decorator
def hflip(img):
    return cv2.flip(img, 1)


# diagonal flip
@type_decorator
def dflip(img):
    return cv2.flip(img, -1)


# TODO flags
@type_decorator
def transform(img, angle, scale):
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, scale)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


@type_decorator
def rotate(img, angle):
    return transform(img, angle, 1)

@type_decorator
def scale(img, factor):
    return transform(img, 0, factor)

@type_decorator
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)

@type_decorator
def xtranslate(img, x):
    return translate(img, x, 0)

@type_decorator
def ytranslate(img, y):
    return translate(img, 0, y)


# TODO interpolation
@type_decorator
def resize(img, width, height, relative=None):
    if _relative_check(width, height, relative=relative):
        h, w = img.shape[:2]
        width, height = rel2abs(width, height, width=w, height=h)
    else:
        width, height = map(int, (width, height))
    if not relative and (width == 0 or height == 0):
        raise ValueError('Width or height have zero size. Try set `relative` to True')
    return cv2.resize(img, (width, height))


shift = translate
xshift = xtranslate
yshift = ytranslate
rotate90 = partial(rotate, angle=90)
rotate180 = partial(rotate, angle=180)
rotate270 = partial(rotate, angle=270)
