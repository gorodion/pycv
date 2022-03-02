import cv2

from ._utils import _draw_decorator

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
def rectangle(img, x0, y0, x1, y1, color=255, t=3):
    cv2.rectangle(img, (x0, y0), (x1, y1), color, t)
    return img


@_draw_decorator
def circle(img, x0, y0, r, color=255, t=3):
    cv2.circle(img, (x0, y0), r, color, t)
    return img


@_draw_decorator
def point(img, x0, y0, color=255):
    cv2.circle(img, (x0, y0), 0, color, -1)
    return img


@_draw_decorator
def line(img, x0, y0, x1, y1, color=255, t=3):
    cv2.line(img, (x0, y0), (x1, y1), color, t)
    return img


@_draw_decorator
def hline(img, y, color=255, t=3):
    w = img.shape[1]
    cv2.line(img, (0, y), (w, y), color, t)
    return img


@_draw_decorator
def vline(img, x, color=255, t=3):
    h = img.shape[0]
    cv2.line(img, (x, 0), (x, h), color, t)
    return img


@_draw_decorator
def putText(img, text, x=0, y=-1, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=255, t=3, line_type=cv2.LINE_AA, flip=False):
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
