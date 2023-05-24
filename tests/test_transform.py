import numpy as np
import cv2
import cv3
from functools import partial
import pytest

COLOR = cv3.opt.COLOR
TEST_IMG = 'img.jpeg'
img_bgr = cv2.imread(TEST_IMG)
img = cv2.cvtColor(img_bgr, code=cv2.COLOR_RGB2BGR)

def test_flips():
    assert np.array_equal(
        cv3.vflip(img),
        cv2.flip(img, 0)
    )
    assert np.array_equal(
        cv3.hflip(img),
        cv2.flip(img, 1)
    )
    assert np.array_equal(
        cv3.dflip(img),
        cv2.flip(img, -1)
    )


class BaseTestTransform:
    cv3_foo = None
    cv2_foo = None

    def test_basic(self):
        assert np.array_equal(
            self.cv3_foo(),
            self.cv2_foo()
        )

class BaseTestInter(BaseTestTransform):
    inter_key = 'flags'
    def test_inter(self):
        # as integer
        assert np.array_equal(
            self.cv3_foo(inter=cv2.INTER_NEAREST),
            self.cv2_foo(**{self.inter_key: cv2.INTER_NEAREST})
        )
        # as string
        assert np.array_equal(
            self.cv3_foo(inter='nearest'),
            self.cv2_foo(**{self.inter_key: cv2.INTER_NEAREST})
        )

class BaseTestBorder(BaseTestTransform):
    border_mode_key = 'borderMode'
    border_value_key = 'borderValue'
    def test_border(self):
        # as integer
        assert np.array_equal(
            self.cv3_foo(border=cv2.BORDER_DEFAULT),
            self.cv2_foo(**{self.border_mode_key: cv2.BORDER_DEFAULT})
        )
        # as flag
        assert np.array_equal(
            self.cv3_foo(border='default'),
            self.cv2_foo(**{self.border_mode_key: cv2.BORDER_DEFAULT})
        )

    def test_border_color(self):
        assert np.array_equal(
            self.cv3_foo(value=(250, 0, 0)),
            self.cv2_foo(**{self.border_value_key: (250, 0, 0)})
        )

class TestTransform(BaseTestInter, BaseTestBorder):
    cv3_foo = partial(cv3.transform, img, angle=25.3, scale=0.6)
    cv2_foo = partial(cv2.warpAffine, img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 25.3, 0.6), img.shape[1::-1])

class TestRotate(BaseTestInter, BaseTestBorder):
    cv3_foo = partial(cv3.rotate, img, 25.3)
    cv2_foo = partial(cv2.warpAffine, img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 25.3, 1), img.shape[1::-1])

class TestScale(BaseTestInter, BaseTestBorder):
    cv3_foo = partial(cv3.scale, img, 0.6)
    cv2_foo = partial(cv2.warpAffine, img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 0, 0.6), img.shape[1::-1])

class TestShift(BaseTestBorder):
    x = 30
    y = -50
    cv3_foo = partial(cv3.shift, img, x, y)
    cv2_foo = partial(cv2.warpAffine, img, np.float32([[1, 0, x], [0, 1, y]]), img.shape[1::-1])

    def test_rel(self):
        h, w = img.shape[:2]
        assert np.array_equal(
            cv3.shift(img, self.x/w, self.y/h),
            self.cv2_foo()
        )

class TestXShift(TestShift):
    y = 0
    cv3_foo = partial(cv3.xshift, img, TestShift.x)
    cv2_foo = partial(cv2.warpAffine, img, np.float32([[1, 0, TestShift.x], [0, 1, y]]), img.shape[1::-1])

class TestYShift(TestShift):
    x = 0
    cv3_foo = partial(cv3.yshift, img, TestShift.y)
    cv2_foo = partial(cv2.warpAffine, img, np.float32([[1, 0, x], [0, 1, TestShift.y]]), img.shape[1::-1])


class TestResize(BaseTestInter):
    inter_key = 'interpolation'
    cv3_foo = partial(cv3.resize, img, 100.3, 200)
    cv2_foo = partial(cv2.resize, img, (100, 200))

    def test_rel(self):
        h, w = img.shape[:2]
        assert np.array_equal(
            cv3.resize(img, 0.2, 1.2, rel=True),
            cv2.resize(img, None, fx=0.2, fy=1.2)
        )

    def test_width_zero(self):
        with pytest.raises(ValueError):
            cv3.resize(img, 0.1, 200)


class TestCrop:
    def test_basic(self):
        assert np.array_equal(
            cv3.crop(img, 20, 30.1, 70, 59.9),
            img[30:60, 20:70]
        )
        # rel
        h, w = img.shape[:2]
        assert np.array_equal(
            cv3.crop(img, 0.2, 0.3, 0.7, 0.6),
            img[round(0.3*h):round(0.6*h), round(0.2*w):round(0.7*w)]
        )

    def test_overflow(self):
        h, w = img.shape[:2]
        for pt1, pt2 in [
            [(-w-10, 10), (w-20, h-20)],
            [(10, -h-10), (w-20, h-20)],
            [(10, 10), (w+10, h-20)],
            [(10, 10), (w-20, h+10)]
        ]:
            crop_cv2 = img[max(pt1[1],0):pt2[1], max(pt1[0],0):pt2[0]]
            crop_cv3 = cv3.crop(img, *pt1, *pt2)

            box = (*pt1, *pt2)
            box = box[0] / w, box[1] / h, box[2] / w, box[3] / h
            crop_cv3_rel = cv3.crop(img, *box)
            crop_cv3_rel_true = cv3.crop(img, *box, rel=True)
            crop_cv3_rel_false = cv3.crop(img, *box, rel=False)

            assert np.array_equal(crop_cv2, crop_cv3)
            assert not np.array_equal(crop_cv2, crop_cv3_rel)
            assert np.array_equal(crop_cv2, crop_cv3_rel_true)
            assert not np.array_equal(crop_cv2, crop_cv3_rel_false)


    def test_x0y0x1y1(self):
        assert np.array_equal(
            cv3.crop(img, 200, 25, 250, 100),
            img[25:100, 200:250]
        )
        assert np.array_equal(
            cv3.crop(img, 200, 100, 250, 25),
            img[25:100, 200:250]
        )
        assert np.array_equal(
            cv3.crop(img, 250, 100, 200, 25),
            img[25:100, 200:250]
        )
        assert np.array_equal(
            cv3.crop(img, 250, 25, 200, 100),
            img[25:100, 200:250]
        )

    def test_zero_size(self):
        cv3.crop(img, 20, 30, 20, 60)


class TestPad(BaseTestBorder):
    border_mode_key = 'borderType'
    border_value_key = 'value'
    cv3_foo = partial(cv3.pad, img, 10, 20., 30.1, 40)
    cv2_foo = partial(cv2.copyMakeBorder, img, 10, 20, 30, 40, borderType=cv2.BORDER_CONSTANT, value=None)

    def test_rel(self):
        h, w = img.shape[:2]
        assert np.array_equal(
            cv3.pad(img, 10/h, 20/h, 30/w, 40/w),
            self.cv2_foo()
        )
