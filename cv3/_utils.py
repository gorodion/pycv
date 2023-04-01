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


# TODO if 0 < color < 1
def _process_color(color):
    if color is None:
        color = opt.COLOR
    if isinstance(color, np.ndarray):
        color = color.tolist()
    if isinstance(color, (list, tuple)):
        color = tuple(map(int, color))
    else:
        return int(color)
    # if opt.RGB:
    #     color = color[::-1]
    return color


def is_relative(*args):
    return all(0 < x < 1 for x in args)


def _relative_check(*args, relative):
    is_relative_coords = all(0 < x < 1 for x in args)
    if is_relative_coords and relative is False:
        warnings.warn('`relative` param set to False but relative args passed')
    if relative is None:
        relative = is_relative_coords
    return relative


def _relative_handle(img, *args, relative):
    if _relative_check(*args, relative=relative):
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