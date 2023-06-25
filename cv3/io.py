from itertools import cycle
from pathlib import Path
import warnings
import cv2
import numpy as np

from .color_spaces import rgb, rgba
from . import opt
from ._utils import typeit, type_decorator

__all__ = [
    'imread',
    'imdecode',
    'imwrite',
    'imshow',
    'Window',
    'Windows',
    'wait_key', 'waitKey',
    'destroy_windows', 'destroyAllWindows',
    'destroy_window', 'destroyWindow'
]

def _imread_flag_match(flag):
    assert flag in ('color', 'gray', 'alpha', 'unchanged')
    if flag == 'color':
        flag = cv2.IMREAD_COLOR
    elif flag == 'gray':
        flag = cv2.IMREAD_GRAYSCALE
    elif flag == 'unchanged':
        flag = cv2.IMREAD_UNCHANGED
    elif flag == 'alpha':
        warnings.warn('Flag name "alpha" deprecated. Please use "unchanged"')
        flag = cv2.IMREAD_UNCHANGED
    return flag



def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def imdecode(buf, flag):
    if isinstance(flag, str):
        flag = _imread_flag_match(flag)
    img = cv2.imdecode(buf, flag)
    return img

def imread(img_path, flag=cv2.IMREAD_COLOR):
    if Path(img_path).is_dir():
        raise IsADirectoryError(str(img_path))
    if not Path(img_path).is_file():
        raise FileNotFoundError(str(img_path))
    if isinstance(img_path, Path):
        img_path = str(img_path)
    if isinstance(flag, str):
        flag = _imread_flag_match(flag)
    img = cv2.imread(img_path, flag)
    if img is None:
        if not is_ascii(img_path):
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), flag)
        if img is None:
            raise OSError(f'File was not read: {img_path}')
    if img.ndim == 2:
        return img
    if opt.RGB:
        if img.shape[-1] == 4:
            img = rgba(img)
        else:
            img = rgb(img)
    return img


def imwrite(img_path, img, mkdir=False, ascii=True):
    if mkdir:
        Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img_path, Path):
        img_path = str(img_path)
    if opt.RGB:
        img = rgb(img)  # includes typeit
    else:
        img = typeit(img)

    if not ascii:
        # if is_ascii(img_path):
        #     warnings.warn('Passed ascii filename but `ascii`=True')
        ext = Path(img_path).suffix
        ret2, buf = cv2.imencode(ext=ext, img=img)
        if not ret2:
            raise OSError('Something went wrong when writing image (non-ascii filename)')
        buf.tofile(img_path)
        return

    ret = cv2.imwrite(img_path, img)
    if not ret:
        raise OSError('Something went wrong when writing image')


def imshow(window_name, img):
    if opt.RGB:
        img = rgb(img)
    else:
        img = typeit(img)
    cv2.imshow(window_name, img)


def wait_key(t):
    return cv2.waitKey(t) & 0xFF

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

        window_name = str(window_name)
        cv2.namedWindow(window_name, flag)

        if pos is not None:
            cv2.moveWindow(window_name, *pos)

        self.window_name = window_name
        Window.__window_count += 1

    def imshow(self, img):
        if opt.RGB:
            img = rgb(img)
        else:
            img = typeit(img)
        cv2.imshow(self.window_name, img)

    def move(self, x, y):
        cv2.moveWindow(self.window_name, x, y)

    def close(self):
        cv2.destroyWindow(self.window_name)

    @staticmethod
    def wait_key(t):
        return wait_key(t)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Windows:
    def __init__(self, window_names, poses=None):
        if poses is None:
            poses = (None,) * len(window_names)

        self.windows = {}
        for window_name, pos in zip(window_names, poses):
            self.windows[window_name] = Window(window_name, pos=pos)

    def __getitem__(self, name):
        return self.windows[name]

    def close(self):
        for window_name in self.windows:
            self.windows[window_name].close()

    @staticmethod
    def wait_key(t):
        return wait_key(t)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

waitKey = wait_key

destroy_windows = destroyAllWindows = cv2.destroyAllWindows
destroy_window = destroyWindow = cv2.destroyWindow

