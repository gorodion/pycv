import cv2
import numpy as np
from functools import partial
import warnings

from ._utils import type_decorator, _relative_check, _relative_handle, _process_color, _handle_rect_coords
from .utils import xywh2xyxy, ccwh2xyxy, yyxx2xyxy

__all__ = [
    'vflip', 'hflip', 'dflip',
    'transform',
    'rotate', 'rotate90', 'rotate180', 'rotate270',
    'scale',
    'shift', 'translate',
    'xshift', 'xtranslate',
    'yshift', 'ytranslate',
    'resize',
    'crop',
    'pad', 'copyMakeBorder',
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
def transform(img, angle, scale, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    if isinstance(inter, str):
        inter = _inter_flag_match(inter)
    border, value = _border_value_check(border, value)
    rot_mat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, scale)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=inter, borderMode=border, borderValue=value)
    return result


def rotate(img, angle, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    return transform(img, angle, 1, inter=inter, border=border, value=value)


def scale(img, factor, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    return transform(img, 0, factor, inter=inter, border=border, value=value)

@type_decorator
def shift(img, x, y, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    x, y = _relative_handle(img, x, y, rel=rel)
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    border, value = _border_value_check(border, value)
    return cv2.warpAffine(img, transMat, dimensions, borderMode=border, borderValue=value)


def xshift(img, x, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    h, w = img.shape[:2]
    x = round(x * w if _relative_check(x, rel=rel) else x)
    return translate(img, x, 0, border=border, value=value)


def yshift(img, y, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    h, w = img.shape[:2]
    y = round(y * h if _relative_check(y, rel=rel) else y)
    return translate(img, 0, y, border=border, value=value)


@type_decorator
def resize(img, width, height, inter=cv2.INTER_LINEAR, rel=None):
    if isinstance(inter, str):
        inter = _inter_flag_match(inter)
    width, height = _relative_handle(img, width, height, rel=rel)
    if width == 0 or height == 0:
        if not rel:
            warnings.warn('Try to set `rel` to True')
        raise ValueError('Width or height have zero size')
    return cv2.resize(img, (width, height), interpolation=inter)


@type_decorator
def crop(img, x0, y0, x1, y1, mode='xyxy', rel=None):
    """
    Returns copied crop of the image
    """
    x0, y0, x1, y1 = _handle_rect_coords(img, x0, y0, x1, y1, mode=mode, rel=rel)

    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    x0 = max(x0, 0)
    y0 = max(y0, 0)

    if y1 == y0 or x1 == x0:
        if not rel:
            warnings.warn('zero-size array. Try to set `rel` to True')
    return img[y0:y1, x0:x1].copy()


@type_decorator
def pad(img, y0, y1, x0, x1, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    border, value = _border_value_check(border, value)
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, rel=rel)
    return cv2.copyMakeBorder(img, y0, y1, x0, x1, borderType=border, value=value)


translate = shift
xtranslate = xshift
ytranslate = yshift
rotate90 = partial(rotate, angle=90)
rotate180 = partial(rotate, angle=180)
rotate270 = partial(rotate, angle=270)
copyMakeBorder = pad
