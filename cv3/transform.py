import cv2
import numpy as np
from functools import partial
import warnings

from ._utils import type_decorator, _relative_handle, _process_color


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
    'yshift',
    'crop',
    'copyMakeBorder',
    'pad'
]

_INTER_DICT = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'lanczos4': cv2.INTER_LANCZOS4
}


def _inter_flag_match(flag):
    assert flag in _INTER_DICT, f'no such flag: "{flag}". Available: {", ".join(_INTER_DICT.keys())}'
    return _INTER_DICT[flag]


_BORDER_DICT = {
    'constant': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT,
    'wrap': cv2.BORDER_WRAP,
    'default': cv2.BORDER_DEFAULT,
    # 'transparent': cv2.BORDER_TRANSPARENT
}


def _border_flag_match(flag):
    assert flag in _BORDER_DICT, f'no such flag: "{flag}". Available: {", ".join(_BORDER_DICT.keys())}'
    return _BORDER_DICT[flag]


def _border_value_check(border, value):
    if isinstance(border, str):
        border = _border_flag_match(border)
    if value is not None:
        value = _process_color(value)
        if border != cv2.BORDER_CONSTANT:
            warnings.warn('`value` parameter is not used when `border` is not cv2.BORDER_CONSTANT')
    return border, value


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


@type_decorator
def transform(img, angle, scale, interpolation=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    if isinstance(interpolation, str):
        interpolation = _inter_flag_match(interpolation)
    border, value = _border_value_check(border, value)
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, scale)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=interpolation, borderMode=border, borderValue=value)
    return result


def rotate(img, angle, interpolation=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    return transform(img, angle, 1, interpolation=interpolation, border=border, value=value)


def scale(img, factor, interpolation=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    return transform(img, 0, factor, interpolation=interpolation, border=border, value=value)

@type_decorator
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)


def xtranslate(img, x):
    return translate(img, x, 0)


def ytranslate(img, y):
    return translate(img, 0, y)


@type_decorator
def resize(img, width, height, interpolation=cv2.INTER_LINEAR, relative=None):
    if isinstance(interpolation, str):
        interpolation = _inter_flag_match(interpolation)
    width, height = _relative_handle(img, width, height, relative=relative)
    if not relative and (width == 0 or height == 0):
        raise ValueError('Width or height have zero size. Try set `relative` to True')
    return cv2.resize(img, (width, height), interpolation=interpolation)


@type_decorator
def crop(img, y0, y1, x0, x1, relative=None):
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, relative=relative)
    return img[y0:y1, x0:x1]


@type_decorator
def copyMakeBorder(img, y0, y1, x0, x1, border=cv2.BORDER_CONSTANT, value=None, relative=None):
    border, value = _border_value_check(border, value)
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, relative=relative)
    return cv2.copyMakeBorder(img, y0, y1, x0, x1, borderType=border, dst=None, value=value)


shift = translate
xshift = xtranslate
yshift = ytranslate
rotate90 = partial(rotate, angle=90)
rotate180 = partial(rotate, angle=180)
rotate270 = partial(rotate, angle=270)
pad = copyMakeBorder
