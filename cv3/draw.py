import cv2
import warnings
import numpy as np
from typing import List

from . import opt
from .utils import xywh2xyxy, ccwh2xyxy, rel2abs
from ._utils import type_decorator, _relative_check, _relative_handle, _process_color

__all__ = [
    'rectangle',
    'polylines',
    'circle',
    'point',
    'points',
    'line',
    'hline',
    'vline',
    'putText',
    'text',
    'rectangles',
]


def _draw_decorator(func):
    def is_number(obj):
        return isinstance(obj, (float, np.floating))

    def not_relative_number(obj):
         return is_number(obj) and obj >= 1

    @type_decorator
    def wrapper(img, *args, color=None, copy=False, **kwargs):
        if copy:
            img = img.copy()

        color = _process_color(color)

        if kwargs.get('t') is None:
            kwargs['t'] = opt.THICKNESS

        # other kw arguments
        for k, v in kwargs.items():
            if is_number(v):
                kwargs[k] = int(v)

        args = (int(arg) if not_relative_number(arg) else arg for arg in args) # TODO for relative > 1

        return func(img, *args, color=color, **kwargs)

    return wrapper


# TODO filled=False
@_draw_decorator
def rectangle(img, x0, y0, x1, y1, mode='xyxy', relative=None, **kwargs):
    assert mode in ('xyxy', 'xywh', 'ccwh')

    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, relative=relative)

    if mode == 'xywh':
        x0, y0, x1, y1 = xywh2xyxy(x0, y0, x1, y1)
    elif mode == 'ccwh':
        x0, y0, x1, y1 = map(int, ccwh2xyxy(x0, y0, x1, y1))

    cv2.rectangle(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def polylines(img, pts, is_closed=True, relative=None, **kwargs):
    """
    :param img:
    :param pts: np.array or List[List] ot Tuple[Tuple]
    :param is_closed: bool
    :return:
    """
    pts = np.array(pts).reshape(-1)
    pts = _relative_handle(img, *pts, relative=relative)
    pts = np.int32(pts).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], is_closed, kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def circle(img, x0, y0, r, relative=None, **kwargs):
    x0, y0 = _relative_handle(img, x0, y0, relative=relative)
    cv2.circle(img, (x0, y0), r, kwargs['color'], kwargs['t'])
    return img


def point(img, x0, y0, r=1, relative=None, **kwargs):
    if 't' in kwargs:
        kwargs.pop('t')
        warnings.warn('Parameter `t` is not used')
    return circle(img, x0, y0, r, t=-1, relative=relative, **kwargs)
    # h, w = img.shape[:2]
    # if all(0 <= x <= 1 for x in (x0, y0)):
    #     x0, y0 = rel2abs(x0, y0, width=w, height=h)
    # cv2.circle(img, (x0, y0), r, kwargs['color'], -1)
    # return img


@_draw_decorator
def line(img, x0, y0, x1, y1, relative=None, **kwargs):
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, relative=relative)
    cv2.line(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def hline(img, y, relative=None, **kwargs):
    h, w = img.shape[:2]
    y = int(y * h if _relative_check(y, relative=relative) else y)
    cv2.line(img, (0, y), (w, y), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def vline(img, x, relative=None, **kwargs):
    h, w = img.shape[:2]
    x = int(x * w if _relative_check(x, relative=relative) else x)
    cv2.line(img, (x, 0), (x, h), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def putText(img, text, x=0.5, y=0.5, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=None, t=None, line_type=cv2.LINE_AA, flip=False, relative=None):
    x, y = _relative_handle(img, x, y, relative=relative)
    cv2.putText(
        img,
        str(text),
        (x, y),
        font,
        scale,
        color,
        t,
        line_type,
        flip
    )
    return img


@type_decorator
def rectangles(img: np.array, rects: List[List], **kwargs) -> np.array:
    for rect in rects:
        img = rectangle(img, *rect, **kwargs)
    return img


@type_decorator
def points(img: np.array, pts: List[List], **kwargs) -> np.array:
    for pt in pts:
        img = point(img, *pt, **kwargs)
    return img

text = putText
