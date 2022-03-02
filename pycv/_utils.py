import cv2
import numpy as np


def _type(img):
    if isinstance(img, list):
        img = np.array(img, 'uint8')
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported type: {type(img)}")
    assert img.ndim == 3 and img.shape[-1] == 3 or img.ndim == 2, f'Incorrect image shape: {img.shape}'  # TODO сравнить условия из scikit-image/matplotlib

    if img.dtype == np.uint8:
        return img
    # if issubclass(img.dtype.type, np.floating):
        # TODO if img.max() <= 1:
        # return img.round().astype('uint8')
    # if issubclass(img.dtype.type, np.integer):
    if issubclass(img.dtype.type, np.number):
        return img.astype('uint8')
    if img.dtype == np.bool:
        return 255 * img.astype('uint8')
    raise TypeError(f"Unsupported dtype: {img.dtype}")

# def _type_decorator


def _imread_flag_match(flag):
    assert flag in ('color', 'gray', 'alpha')
    if flag == 'color':
        flag = cv2.IMREAD_COLOR
    elif flag == 'gray':
        flag = cv2.IMREAD_GRAYSCALE
    elif flag == 'alpha':
        flag = cv2.IMREAD_UNCHANGED
    return flag


# TODO args to integer
# TODO take args and parse numbers/tuples/lists
# TODO color as int/list of int/tuple of int
def _draw_decorator(func):
    def wrapper(img, *args, color=255, copy=False, **kwargs):
        img = _type(img)
        if copy:
            img = img.copy()

        # TODO if 0 < color < 1
        # color
        if isinstance(color, np.ndarray):
            color = color.tolist()
        if isinstance(color, (list, tuple)):
            color = tuple(map(int, color))
        else:
            color = int(color)

        # other kw arguments
        for k, v in kwargs.items():
            kwargs[k] = int(v)

        return func(img, *args, color=color, **kwargs)
    return wrapper