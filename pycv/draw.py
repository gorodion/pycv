import cv2
# import warnings

from ._utils import _draw_decorator
from . import options

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

# TODO filled=False
# TODO type: `xyxy` and `xywh` and `ccwh`
@_draw_decorator
def rectangle(img, x0, y0, x1, y1, **kwargs):
    cv2.rectangle(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def circle(img, x0, y0, r, **kwargs):
    cv2.circle(img, (x0, y0), r, kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def point(img, x0, y0, r=0, **kwargs):
    # if 't' in kwargs:
    #     warnings.warn('Parameter `t` is not used')
    cv2.circle(img, (x0, y0), r, kwargs['color'], -1)
    return img


@_draw_decorator
def line(img, x0, y0, x1, y1, **kwargs):
    cv2.line(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def hline(img, y, **kwargs):
    w = img.shape[1]
    cv2.line(img, (0, y), (w, y), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def vline(img, x, **kwargs):
    h = img.shape[0]
    cv2.line(img, (x, 0), (x, h), kwargs['color'], kwargs['t'])
    return img


@_draw_decorator
def putText(img, text, x=0, y=None, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=None, t=None, line_type=cv2.LINE_AA, flip=False):
    if y is None:
        h = img.shape[0]
        y = h // 2
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
