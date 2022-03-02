import cv2
import numpy as np

__all__ = [
    'threshold'
]


# TODO flags
def threshold(img: np.ndarray, thr=127, max=255):
    assert img.ndim == 2, '`img` should be gray image'
    # TODO if img.max() < 1
    _, thresh = cv2.threshold(img, thr, max, cv2.THRESH_BINARY)
    return thresh
