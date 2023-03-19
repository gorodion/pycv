from itertools import cycle
from pathlib import Path

import cv2
import numpy as np

from .color_spaces import rgb
from . import opt
from ._utils import typeit, type_decorator

__all__ = [
    'imread',
    'imwrite',
    'imshow',
    'Window'
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
    if opt.RGB:
        img = rgb(img)
    return img


def imwrite(imgp, img, **kwargs):
    Path(imgp).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(imgp, Path):
        imgp = str(imgp)
    if opt.RGB:
        img = rgb(img)  # includes typeit
    else:
        img = typeit(img)
    assert cv2.imwrite(imgp, img), 'Something went wrong'


def imshow(window_name, img):
    if opt.RGB:
        img = rgb(img)
    else:
        img = typeit(img)
    cv2.imshow(window_name, img)


class Window:
    __window_count = 0

    def __init__(self, window_name=None, pos=None, flag=cv2.WINDOW_AUTOSIZE):
        """

        :param window_name:
        :param pos: tuple. Starting position of the window (x, y)
        :param flag:
        """
        if window_name is None:
            window_name = f'window{Window.__window_count}'
            Window.__window_count += 1

        self.window_name = window_name
        cv2.namedWindow(window_name, flag)

        if pos is not None:
            cv2.moveWindow(window_name, *pos)

    def imshow(self, img):
        if opt.RGB:
            img = rgb(img)
        else:
            img = typeit(img)
        cv2.imshow(self.window_name, img)

    def close(self):
        cv2.destroyWindow(self.window_name)

    def wait_key(self, t):
        return cv2.waitKey(t) & 0xFF

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    waitKey = wait_key