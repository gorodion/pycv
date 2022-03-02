import cv2
import numpy as np

__all__ = [
    'vflip',
    'hflip',
    'dflip',
    'transform',
    'rotate',
    'scale',
    'translate',
    'resize',
    'shift'
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


# TODO interpolation
def resize(img, width, height):
    return cv2.resize(img, (width, height))


shift = translate
