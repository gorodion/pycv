import cv2

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
# TODO type: `xyxy` and `xywh` and `ccwh`
@_draw_decorator
def rectangle(img, x0, y0, x1, y1, color=options.COLOR, t=options.THICKNESS):
    cv2.rectangle(img, (x0, y0), (x1, y1), color, t)
    return img


@_draw_decorator
def circle(img, x0, y0, r, color=options.COLOR, t=options.THICKNESS):
    cv2.circle(img, (x0, y0), r, color, t)
    return img


@_draw_decorator
def point(img, x0, y0, r=0, color=options.COLOR):
    cv2.circle(img, (x0, y0), r, color, -1)
    return img


@_draw_decorator
def line(img, x0, y0, x1, y1, color=options.COLOR, t=options.THICKNESS):
    cv2.line(img, (x0, y0), (x1, y1), color, t)
    return img


@_draw_decorator
def hline(img, y, color=options.COLOR, t=options.THICKNESS):
    w = img.shape[1]
    cv2.line(img, (0, y), (w, y), color, t)
    return img


@_draw_decorator
def vline(img, x, color=options.COLOR, t=options.THICKNESS):
    h = img.shape[0]
    cv2.line(img, (x, 0), (x, h), color, t)
    return img


@_draw_decorator
def putText(img, text, x=0, y=-1, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=options.COLOR, t=options.THICKNESS, line_type=cv2.LINE_AA, flip=False):
    h = img.shape[0]
    if y == -1:
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
