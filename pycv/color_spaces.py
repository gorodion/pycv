from functools import partial
import cv2
import numpy as np

__all__ = [
    'rgb',
    'bgr',
    'rgb2bgr',
    'bgr2rgb',
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
def rgb(img: np.ndarray):
    if img.ndim != 3:  # only if 3-color image
        return img
    return cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)


rgb2bgr = bgr2rgb = bgr = rgb
bgr2gray = partial(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY)
rgb2gray = partial(cv2.cvtColor, code=cv2.COLOR_RGB2GRAY)
gray2rgb = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2RGB)
gray2bgr = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2BGR)
bgr2hsv = partial(cv2.cvtColor, code=cv2.COLOR_BGR2HSV)
rgb2hsv = partial(cv2.cvtColor, code=cv2.COLOR_RGB2HSV)
hsv2bgr = partial(cv2.cvtColor, code=cv2.COLOR_HSV2BGR)
hsv2rgb = partial(cv2.cvtColor, code=cv2.COLOR_HSV2RGB)