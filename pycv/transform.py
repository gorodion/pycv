import cv2
import numpy as np
from functools import partial

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

def vflip(img):
    return cv2.flip(img, 0)


def hflip(img):
    return cv2.flip(img, 1)

# diagonal flip
def dflip(img):
    return cv2.flip(img, -1)


# TODO flags
def transform(img, angle, scale):
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, scale)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate(img, angle):
    return transform(img, angle, 1)


def scale(img, factor):
    return transform(img, 0, factor)


def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)


def xtranslate(img, x):
    return translate(img, x, 0)


def ytranslate(img, y):
    return translate(img, 0, y)


# TODO interpolation
def resize(img, width, height):
    h, w = img.shape[:2]
    if 0 <= width <= 1:
        width *= w
    if 0 <= height <= 1:
        height *= h
    width, height = int(width), int(height)
    return cv2.resize(img, (width, height))


shift = translate
xshift = xtranslate
yshift = ytranslate
rotate90 = partial(rotate, angle=90)
rotate180 = partial(rotate, angle=180)
rotate270 = partial(rotate, angle=270)
