import cv2
import numpy as np

from ._utils import type_decorator
__all__ = [
    'threshold'
]


# TODO flags
@type_decorator
def threshold(img: np.ndarray, thr=127, max=255):
    assert img.ndim == 2, '`img` must be gray image'
    # TODO if img.max() < 1
    _, thresh = cv2.threshold(img, thr, max, cv2.THRESH_BINARY)
    return thresh
