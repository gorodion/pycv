import numpy as np
import cv2
import cv3
from pathlib import Path
import os
import shutil
import pytest



class TestRectangle():
    def test_rect_draw(self):
        zeros = cv3.zeros(100, 100, 3)
        cv3.rectangle(zeros, 25, 25, 75, 75)
        assert zeros.any()

    def test_rect_draw_check(self):
        paint_cv2 = cv3.zeros(100, 100)
        cv2.rectangle(paint_cv2, (25, 30), (70, 75), cv3.opt.COLOR, cv3.opt.THICKNESS)

        paint_cv3 = cv3.zeros(100, 100)
        paint_cv3 = cv3.rectangle(paint_cv3, 25, 30, 70, 75)

        assert np.array_equal(paint_cv2, paint_cv3)

    def test_rect_mode_xywh(self):
        paint_cv2 = cv3.zeros(100, 100)
        cv2.rectangle(paint_cv2, (50, 60), (70, 70), cv3.opt.COLOR, cv3.opt.THICKNESS)

        paint_cv3 = cv3.zeros(100, 100)
        paint_cv3 = cv3.rectangle(paint_cv3, 50, 60, 20, 10, mode='xywh')

        assert np.array_equal(paint_cv2, paint_cv3)

    def test_rect_mode_ccwh(self):
        paint_cv2 = cv3.zeros(100, 100)
        cv2.rectangle(paint_cv2, (40, 55), (60, 65), cv3.opt.COLOR, cv3.opt.THICKNESS)

        paint_cv3 = cv3.zeros(100, 100)
        paint_cv3 = cv3.rectangle(paint_cv3, 50, 60, 20, 10, mode='ccwh')

        assert np.array_equal(paint_cv2, paint_cv3)

    def test_rect_overflow(self):
        for pt1, pt2 in [
            [(-10, 10), (80, 80)],
            [(10, -10), (80, 80)],
            [(10, 10), (110, 80)],
            [(10, 10), (80, 110)]
        ]:

            paint_cv2 = cv3.zeros(100, 100)
            cv2.rectangle(paint_cv2, pt1, pt2, cv3.opt.COLOR, cv3.opt.THICKNESS)

            paint_cv3 = cv3.zeros(100, 100)
            paint_cv3 = cv3.rectangle(paint_cv3, *pt1, *pt2)

            assert np.array_equal(paint_cv2, paint_cv3)

    def test_rect_relative(self):
        paint_cv2 = cv3.zeros(100, 100)
        cv2.rectangle(paint_cv2, (20, 30), (80, 60), cv3.opt.COLOR, cv3.opt.THICKNESS)

        # xyxy
        paint_xyxy = cv3.rectangle(cv3.zeros(100, 100), 0.2, 0.3, 0.8, 0.6)
        assert np.array_equal(paint_xyxy, paint_cv2)

        # xywh
        paint_xywh = cv3.rectangle(cv3.zeros(100, 100), 0.2, 0.3, 0.6, 0.3, mode='xywh')
        assert np.array_equal(paint_xywh, paint_cv2)

        # ccwh
        paint_ccwh = cv3.rectangle(cv3.zeros(100, 100), 0.5, 0.45, 0.6, 0.3, mode='ccwh')
        assert np.array_equal(paint_ccwh, paint_cv2)


class TestDrawUtils:

    def test_util(self):
        pass

# test frame
# test relative overflow
# test corner relative
# test relative true/false
