from itertools import cycle
from pathlib import Path

import cv2
import numpy as np

from .color_spaces import rgb
from . import options

__all__ = [
    'imread',
    'imwrite',
    'imshow'
]


def _imread_flag_match(flag):
    assert flag in ('color', 'gray', 'alpha')
    if flag == 'color':
        flag = cv2.IMREAD_COLOR
    elif flag == 'gray':
        flag = cv2.IMREAD_GRAYSCALE
    elif flag == 'alpha':
        flag = cv2.IMREAD_UNCHANGED
    return flag


def imread(imgp, flag=cv2.IMREAD_COLOR):
    if not Path(imgp).is_file():
        raise FileNotFoundError(str(imgp))
    if isinstance(imgp, Path):
        imgp = str(imgp)
    if isinstance(flag, str):
        flag = _imread_flag_match(flag)
    img = cv2.imread(imgp, flag)
    assert img is not None, f'File was not read: {imgp}'
    if options.RGB:
        img = rgb(img)
    return img


def imwrite(imgp, img, **kwargs):
    Path(imgp).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(imgp, Path):
        imgp = str(imgp)
    if options.RGB:
        img = rgb(img)
    assert cv2.imwrite(imgp, img), 'Something went wrong'


# TODO window_name increment
# TODO list of sources
def imshow(to_show, window_name=''):
    if isinstance(to_show, np.ndarray):
        to_show = cycle((to_show,))
    assert hasattr(to_show, '__next__') # isinstance(to_show, types.GeneratorType)
    for img in to_show:
        if options.RGB:
            img = rgb(img)
        cv2.imshow(window_name, img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
