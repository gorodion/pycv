import cv2
# import warnings
import numpy as np

from . import options
from .utils import xywh2xyxy, ccwh2xyxy, rel2abs
from ._utils import type_decorator

__all__ = [
    'rectangle',
    'circle',
    'point',
    'line',
    'hline',
    'vline',
    'putText',
    'text'
]


def _draw_decorator(func):
    def is_number(obj):
        return isinstance(obj, (float, np.floating))

    def not_relative_number(obj):
         return is_number(obj) and obj >= 1

    # TODO if 0 < color < 1
    def process_color(color):
        if color is None:
            color = options.COLOR
        if isinstance(color, np.ndarray):
            color = color.tolist()
        if isinstance(color, (list, tuple)):
            color = tuple(map(round, color))
        else:
            color = round(color), 0, 0
        if options.RGB:
            color = color[::-1]
        return color

    @type_decorator
    def wrapper(img, *args, color=None, copy=False, **kwargs):
        if copy:
            img = img.copy()

        color = process_color(color)

        if kwargs.get('t') is None:
            kwargs['t'] = options.THICKNESS

        # other kw arguments
        for k, v in kwargs.items():
            if is_number(v):
                kwargs[k] = round(v)

        args = (round(arg) if not_relative_number(arg) else arg for arg in args)

        return func(img, *args, color=color, **kwargs)

    return wrapper


# TODO filled=False
@_draw_decorator
def rectangle(img, x0, y0, x1, y1, mode='xyxy', **kwargs):
    assert mode in ('xyxy', 'xywh', 'ccwh')
    if mode == 'xywh':
        x0, y0, x1, y1 = xywh2xyxy(x0, y0, x1, y1)
    elif mode == 'ccwh':
        x0, y0, x1, y1 = ccwh2xyxy(x0, y0, x1, y1)

    h, w = img.shape[:2] # TODO imshape
    x0, y0, x1, y1 = rel2abs(x0, y0, x1, y1, width=w, height=h)
    x0, y0, x1, y1 = map(round, (x0, y0, x1, y1))
    cv2.rectangle(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def circle(img, x0, y0, r, **kwargs):
    h, w = img.shape[:2] # TODO imshape
    x0, y0 = rel2abs(x0, y0, width=w, height=h)
    x0, y0, r = map(round, (x0, y0, r))
    cv2.circle(img, (x0, y0), r, kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def point(img, x0, y0, r=0, **kwargs):
    # if 't' in kwargs:
    #     warnings.warn('Parameter `t` is not used')
    h, w = img.shape[:2] # TODO imshape
    x0, y0 = rel2abs(x0, y0, width=w, height=h)
    x0, y0, r = map(round, (x0, y0, r))
    cv2.circle(img, (x0, y0), r, kwargs['color'], -1)
    return img


@_draw_decorator
def line(img, x0, y0, x1, y1, **kwargs):
    h, w = img.shape[:2] # TODO imshape
    x0, y0, x1, y1 = rel2abs(x0, y0, x1, y1, width=w, height=h)
    x0, y0, x1, y1 = map(round, (x0, y0, x1, y1))
    cv2.line(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def hline(img, y, **kwargs):
    h, w = img.shape[:2]
    y = round(y * h if 0 <= y <= 1 else y)
    cv2.line(img, (0, y), (w, y), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def vline(img, x, **kwargs):
    h, w = img.shape[:2]
    x = round(x * w if 0 <= x <= 1 else x)
    cv2.line(img, (x, 0), (x, h), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def putText(img, text, x=0, y=None, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=None, t=None, line_type=cv2.LINE_AA, flip=False):
    h, w = img.shape[:2]
    if y is None:
        y = h // 2
    x, y = rel2abs(x, y, width=w, height=h)
    x, y = map(round, (x, y))
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        t,
        line_type,
        flip
    )
    return img


text = putText
