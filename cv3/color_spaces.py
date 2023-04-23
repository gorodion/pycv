from functools import partial
import cv2
import numpy as np
from ._utils import type_decorator

__all__ = [
    'rgb',
    'bgr',
    'rgba',
    'bgra',
    'rgb2bgr',
    'bgr2rgb',
    'rgba2bgra',
    'bgra2rgba',
    'rgb2gray',
    'bgr2gray',
    'rgb2gray',
    'gray2rgb',
    'gray2bgr',
    'bgr2hsv',
    'rgb2hsv',
    'hsv2bgr',
    'hsv2rgb',
]


@type_decorator
def _cvtColor(img, code):
    if code == cv2.COLOR_GRAY2RGB:
        if img.ndim != 2:
            raise ValueError('Image should be grayscale (2 dims)')
    return cv2.cvtColor(img, code=code)


rgb2bgr = bgr2rgb = bgr = rgb = partial(_cvtColor, code=cv2.COLOR_RGB2BGR)
rgba2bgra = bgra2rgba = rgba = bgra = partial(_cvtColor, code=cv2.COLOR_RGBA2BGRA)
gray2rgb = gray2bgr = partial(_cvtColor, code=cv2.COLOR_GRAY2RGB)
bgr2gray = partial(_cvtColor, code=cv2.COLOR_BGR2GRAY)
rgb2gray = partial(_cvtColor, code=cv2.COLOR_RGB2GRAY)
bgr2hsv = partial(_cvtColor, code=cv2.COLOR_BGR2HSV)
rgb2hsv = partial(_cvtColor, code=cv2.COLOR_RGB2HSV)
hsv2bgr = partial(_cvtColor, code=cv2.COLOR_HSV2BGR)
hsv2rgb = partial(_cvtColor, code=cv2.COLOR_HSV2RGB)
