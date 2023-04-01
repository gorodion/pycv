import warnings
import numpy as np

from . import opt
from .utils import rel2abs, xywh2xyxy, ccwh2xyxy, yyxx2xyxy

warnings.simplefilter('always', UserWarning)


def typeit(img):
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        return img
    warnings.warn('The image was copied because it needs to be cast to the correct type. To avoid copying, please cast the image to np.uint8')
    return np.uint8(img)
    # if isinstance(img, list):
    #     img = np.array(img, 'uint8')
    # if not isinstance(img, np.ndarray):
    #     raise TypeError(f"Unsupported type: {type(img)}")
    # assert img.ndim == 3 and img.shape[-1] == 3 or img.ndim == 2, f'Incorrect image shape: {img.shape}'
    # if img.dtype == np.uint8:
    #     return img
    # if issubclass(img.dtype.type, np.floating):
        # TODO if img.max() <= 1:
        # return img.round().astype('uint8')
    # if issubclass(img.dtype.type, np.integer):
    # if issubclass(img.dtype.type, np.number):
    #     return img.astype('uint8')
    # if img.dtype == np.bool:
    #     return 255 * img.astype('uint8')
    # raise TypeError(f"Unsupported dtype: {img.dtype}")


def type_decorator(func):
    def wrapper(img, *args, **kwargs):
        img = typeit(img)
        return func(img, *args, **kwargs)
    return wrapper


def _process_color(color):
    if color is None:
        color = opt.COLOR
    if isinstance(color, (int, np.unsignedinteger)):
        return int(color)
    if isinstance(color, (float, np.floating)):
        assert 0 <= color <= 255, 'if `color` passed as number it should be in range [0, 255]'
        color = (color, 0., 0.)
    if isinstance(color, np.ndarray):
        color = color.ravel().tolist()
    if isinstance(color, (list, tuple)):
        if all(0 <= x <= 1 and isinstance(x, (float, np.floating)) for x in color):
            color = tuple(round(c*255) for c in color)
        else:
            assert all(0 <= c <= 255 for c in color), '`color` must be in range [0, 255]'
            color = tuple(map(round, color))
    else:
        raise ValueError('Unexpected type of `color` arg (int, float, np.array, list, tuple supported)')
    return color


def _relative_check(*args, rel):
    is_relative_coords = all(0 <= abs(x) <= 1 and isinstance(x, (float, np.floating)) for x in args)
    if is_relative_coords and rel is False:
        warnings.warn('`rel` param set to False but relative args passed')
    if rel is None:
        rel = is_relative_coords
    return rel


def _relative_handle(img, *args, rel):
    if _relative_check(*args, rel=rel):
        h, w = img.shape[:2]
        return tuple(rel2abs(*args, width=w, height=h))
    return tuple(map(int, args))


def _handle_rect_mode(mode, x0, y0, x1, y1):
    assert mode in ('xyxy', 'xywh', 'ccwh', 'yyxx')
    if mode == 'xyxy':
        return x0, y0, x1, y1
    if mode == 'xywh':
        return xywh2xyxy(x0, y0, x1, y1)
    if mode == 'ccwh':
        return ccwh2xyxy(x0, y0, x1, y1)
    if mode == 'yyxx':
        return yyxx2xyxy(x0, y0, x1, y1)