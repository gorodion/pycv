import numpy as np
import cv2
import cv3
from PIL import Image
from pathlib import Path
import os
import shutil
import pytest
import time

# NOTE: RGB==True by default

TEST_IMG = 'img.jpeg'
TEST_ALPHA_IMG = 'parrot.webp'
NON_EXIST_IMG = 'spamimage.jpg'
OUT_PATH_IMG = 'img_out.png'
INVALID_EXT_PATH = 'spam.pngxxx'
TMP_DIR = 'temp/'
TESTS_DIR = 'tests'

test_img_bgr = cv2.imread(TEST_IMG)
assert test_img_bgr is not None
test_img = cv2.cvtColor(test_img_bgr, code=cv2.COLOR_RGB2BGR)
assert cv2.imread(NON_EXIST_IMG) is None


def test_imread_str():
    "Testing imread"
    img = cv3.imread(TEST_IMG)


def test_imread_path():
    "Testing imread with pathlib.Path"
    path = Path(TEST_IMG)
    img = cv3.imread(path)


def test_imread():
    "Check if cv3.imread gives same array as cv2"
    img = cv3.imread(TEST_IMG)
    assert np.array_equal(test_img, img)


def test_imread_not_found():
    "Check if FileNotFound exception raises"
    with pytest.raises(FileNotFoundError):
        cv3.imread(NON_EXIST_IMG)

@pytest.fixture()
def invalid_ext_fixture():
    Path(INVALID_EXT_PATH).touch()
    yield
    Path(INVALID_EXT_PATH).unlink()


@pytest.mark.usefixtures('invalid_ext_fixture')
def test_imread_invalid_extension():
    with pytest.raises(OSError):
        cv3.imread(INVALID_EXT_PATH)


def test_imread_dir():
    with pytest.raises(IsADirectoryError):
        cv3.imread(TESTS_DIR)


def test_imread_rgb():
    "Testing cv3.opt.RGB flag for imread"
    # bgr
    try:
        cv3.opt.set_bgr()
        img = cv3.imread(TEST_IMG)
        assert np.array_equal(test_img_bgr, img)
    finally:
        cv3.opt.set_rgb()

    # rgb
    img = cv3.imread(TEST_IMG)
    assert np.array_equal(test_img, img)


def test_imread_gray():
    img0 = cv2.imread(TEST_IMG, 0)
    img1 = cv3.imread(TEST_IMG, 'gray')
    assert np.array_equal(img0, img1)

def test_imread_unchanged():
    "Testing different cv3.imread flags"
    img0 = cv2.imread(TEST_ALPHA_IMG, cv2.IMREAD_UNCHANGED)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2RGBA)
    img1 = cv3.imread(TEST_ALPHA_IMG, 'alpha')

    assert img1.ndim == 3 and img1.shape[-1] == 4
    assert np.array_equal(img0, img1)

    # bgr
    try:
        cv3.opt.set_bgr()
        img0 = cv2.imread(TEST_ALPHA_IMG, cv2.IMREAD_UNCHANGED)
        img1 = cv3.imread(TEST_ALPHA_IMG, 'alpha')
        assert np.array_equal(img0, img1)
    finally:
        cv3.opt.set_rgb()


def test_imwrite_str():
    "Testing imwrite with string path"
    Path(OUT_PATH_IMG).unlink(missing_ok=True)
    cv3.imwrite(OUT_PATH_IMG, test_img_bgr)

    assert os.path.isfile(OUT_PATH_IMG)
    os.unlink(OUT_PATH_IMG)


def test_imwrite_path():
    "Testing imwrite with pathlib.Path"
    out_path = Path(OUT_PATH_IMG)
    out_path.unlink(missing_ok=True)

    cv3.imwrite(out_path, test_img_bgr)
    assert out_path.is_file()
    out_path.unlink()


def test_imwrite():
    "Check if cv3.imwrite saving files as expected"
    cv3.imwrite(OUT_PATH_IMG, test_img)
    out_img_bgr = cv2.imread(OUT_PATH_IMG)
    try:
        assert np.array_equal(test_img_bgr, out_img_bgr)
    finally:
        os.unlink(OUT_PATH_IMG)


def test_imwrite_rgb():
    "Testing cv3.opt.RGB for imwrite"
    try:
        cv3.opt.set_bgr()
        cv3.imwrite(OUT_PATH_IMG, test_img_bgr)

        out_img_bgr = cv2.imread(OUT_PATH_IMG)

        assert np.array_equal(test_img_bgr, out_img_bgr)
    finally:
        cv3.opt.set_rgb()
        os.unlink(OUT_PATH_IMG)


def test_imwrite_mkdir():
    "Testing auto mkdir if dir not exists"
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    out_path = os.path.join(TMP_DIR, OUT_PATH_IMG)

    cv3.imwrite(out_path, test_img, mkdir=True)
    assert os.path.isfile(out_path)
    shutil.rmtree(TMP_DIR)


def test_imwrite_nomkdir():
    "Testing behavior if dir not exists and mkdir=False"
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    out_path = os.path.join(TMP_DIR, OUT_PATH_IMG)

    with pytest.raises(OSError):
        cv3.imwrite(out_path, test_img)


def test_imwrite_invalid_extension():
    "Testing writing with unknown extensions"
    with pytest.raises(cv2.error):
        cv3.imwrite(INVALID_EXT_PATH, test_img)


def test_imread_imwrite_rgb():
    "Check if reading and writing are using cv3.opt.RGB flag"
    # bgr
    try:
        cv3.opt.set_bgr()
        cv3.imwrite(OUT_PATH_IMG, test_img)
        img = cv3.imread(OUT_PATH_IMG)
        assert np.array_equal(img, test_img)
    finally:
        cv3.opt.set_rgb()
        os.unlink(OUT_PATH_IMG)

    # rgb
    cv3.imwrite(OUT_PATH_IMG, test_img)
    img = cv3.imread(OUT_PATH_IMG)
    assert np.array_equal(img, test_img)
    os.unlink(OUT_PATH_IMG)
