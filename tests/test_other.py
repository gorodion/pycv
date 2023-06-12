import numpy as np
import cv2
import cv3
import pytest

TEST_IMG = 'img.jpeg'
img_bgr = cv2.imread(TEST_IMG)
img_gray = cv2.imread(TEST_IMG, 0)
img = cv2.cvtColor(img_bgr, code=cv2.COLOR_RGB2BGR)


def test_gray2rgb():
    # GRAY to RGB
    rgb = cv3.gray2rgb(img_gray)
    assert rgb.shape == (*img_gray.shape, 3)

    # image with shape (height, width, 1)
    img_1 = img_gray[..., None]
    cv3.gray2rgb(img_1)
    assert rgb.shape == (*img_gray.shape, 3)

    # to RGBA
    rgba = cv3.gray2rgba(img_1)
    assert rgba.shape == (*img_gray.shape, 4)

    # image with shape (height, width, 3)
    with pytest.raises(ValueError):
        cv3.gray2rgba(img)
