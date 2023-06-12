import numpy as np
import cv2
import cv3
import pytest

COLOR = cv3.opt.COLOR
THICKNESS = cv3.opt.THICKNESS

@pytest.fixture()
def set_bgr_fixture():
    cv3.opt.set_bgr()
    yield
    cv3.opt.set_rgb()



class TestDrawDecorator:
    def test_no_copy(self):
        canvas = np.zeros((100, 100, 3), np.uint8)
        cv3.rectangle(canvas, 25, 30, 65, 70)
        assert canvas.any()

    def test_copy(self):
        canvas = np.zeros((100, 100, 3), np.uint8)
        draw = cv3.rectangle(canvas, 25, 30, 65, 70, copy=True)
        assert draw.any()
        assert not canvas.any()

    def test_thickness_float(self):
        assert np.array_equal(
            cv3.rectangle(cv3.zeros(100, 100), 25, 30, 65, 70, t=1.9),
            cv3.rectangle(cv3.zeros(100, 100), 25, 30, 65, 70, t=2)
        )

    def test_color(self):
        draw_red = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=(255, 0, 0))
        draw_red_cv2 = cv2.rectangle(np.zeros((100, 100, 3), np.uint8), (25, 30), (65, 70), (255, 0, 0), THICKNESS)
        assert np.array_equal(draw_red, draw_red_cv2)

    @pytest.mark.usefixtures('set_bgr_fixture')
    def test_color_bgr(self):
        self.test_color()

    def test_color_text(self):
        draw_blue = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color="blue")
        draw_blue_cv2 = cv2.rectangle(np.zeros((100, 100, 3), np.uint8), (25, 30), (65, 70), (0, 0, 255), THICKNESS)
        assert np.array_equal(draw_blue, draw_blue_cv2)

    def test_color_text_notfound(self):
        with pytest.raises(AssertionError):
            cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color="blue123")

    @pytest.mark.usefixtures('set_bgr_fixture')
    def test_color_text_bgr(self):
        draw_blue = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color="red")
        draw_blue_cv2 = cv2.rectangle(np.zeros((100, 100, 3), np.uint8), (25, 30), (65, 70), (0, 0, 255), THICKNESS)
        assert np.array_equal(draw_blue, draw_blue_cv2)

    def test_color_number(self):
        draw_red_int = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=200)
        draw_red_uint_np = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=np.uint32([200, 0, 0])[0])
        draw_red_float = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=200.)
        draw_red_float_rel = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=200/255)
        draw_red_float_np = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=np.float32([200., 0., 0.])[0])
        draw_red_float_np_rel = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=np.float32([200/255, 0., 0.])[0])
        draw_red_cv2 = cv2.rectangle(np.zeros((100, 100, 3), np.uint8), (25, 30), (65, 70), (200, 0, 0), THICKNESS)

        assert np.array_equal(draw_red_int, draw_red_cv2)
        assert np.array_equal(draw_red_uint_np, draw_red_cv2)
        assert np.array_equal(draw_red_float, draw_red_cv2)
        assert np.array_equal(draw_red_float_rel, draw_red_cv2)
        assert np.array_equal(draw_red_float_np, draw_red_cv2)
        assert np.array_equal(draw_red_float_np_rel, draw_red_cv2)

    def test_color_asarray(self):
        draw_red_tuple = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=(200, 0, 0))
        draw_red_tuple_rel = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=(200/255, 0., 0.))
        draw_red_list = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=[200, 0, 0])
        draw_red_ndarray = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=np.array([200, 0, 0]))
        draw_red_ndarray2 = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=np.array([[200, 0, 0]]))
        draw_red_ndarray_float = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=np.array([200., 0., 0.]))
        draw_red_cv2 = cv2.rectangle(np.zeros((100, 100, 3), np.uint8), (25, 30), (65, 70), (200, 0, 0), THICKNESS)

        assert np.array_equal(draw_red_tuple, draw_red_cv2)
        assert np.array_equal(draw_red_tuple_rel, draw_red_cv2)
        assert np.array_equal(draw_red_list, draw_red_cv2)
        assert np.array_equal(draw_red_ndarray, draw_red_cv2)
        assert np.array_equal(draw_red_ndarray2, draw_red_cv2)
        assert np.array_equal(draw_red_ndarray_float, draw_red_cv2)

    def test_color_out_of_bounds(self):
        with pytest.raises(AssertionError):
            cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=-5)

        with pytest.raises(AssertionError):
            cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color=260)

    def test_color_invalid(self):
        with pytest.raises(ValueError):
            cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, color={1: 2})

    def test_line_type(self):
        linetype_flag = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, line_type=cv2.LINE_8)
        linetype_str_flag = cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 65, 70, line_type='line_8')
        linetype_cv2 = cv2.rectangle(np.zeros((100, 100, 3), np.uint8), (25, 30), (65, 70), COLOR, THICKNESS, lineType=cv2.LINE_8)

        assert np.array_equal(linetype_flag, linetype_cv2)
        assert np.array_equal(linetype_str_flag, linetype_cv2)


class Shape:
    def get_sample_cv2(self, color, t, line_type):
        pass

    def get_sample_cv3(self, color, t, line_type):
        pass

    def test_color_thickness_linetype(self, color=(100, 0, 250), t=3, line_type=cv2.LINE_AA):
        assert np.array_equal(
            self.get_sample_cv2(color, t, line_type),
            self.get_sample_cv3(color, t, line_type)
        )


class TestRectangle(Shape):
    def get_sample_cv2(self, color, t, line_type):
        return cv2.rectangle(np.zeros((100, 100, 3), np.uint8), (25, 30), (70, 75), color, t, lineType=line_type)
    def get_sample_cv3(self, color, t, line_type):
        return cv3.rectangle(cv3.zeros(100, 100, 3), 25, 30, 70, 75, color=color, t=t, line_type=line_type)

    def test_rect_draw_check(self):
        paint_cv2 = cv2.rectangle(np.zeros((100, 100), np.uint8), (25, 30), (70, 75), COLOR, THICKNESS)
        paint_cv3 = cv3.rectangle(cv3.zeros(100, 100), 25, 30, 70, 75)
        paint_cv3_float = cv3.rectangle(cv3.zeros(100, 100), 25.2, 29.9, 70.1, 75.12)

        assert np.array_equal(paint_cv2, paint_cv3)
        assert np.array_equal(paint_cv2, paint_cv3_float)


    def test_rect_mode_xyxy(self):
        paint_cv2 = np.zeros((100, 100), np.uint8)
        cv2.rectangle(paint_cv2, (70, 75), (25, 30), COLOR, THICKNESS)

        paint_cv3 = cv3.rectangle(cv3.zeros(100, 100), 70, 30, 25, 75)
        paint_cv3_rel = cv3.rectangle(cv3.zeros(100, 100), 0.7, 0.3, 0.25, 0.75)

        assert np.array_equal(paint_cv2, paint_cv3)
        assert np.array_equal(paint_cv2, paint_cv3_rel)

    def test_rect_mode_xywh(self):
        paint_cv2 = np.zeros((100, 100), np.uint8)
        cv2.rectangle(paint_cv2, (50, 60), (70, 70), COLOR, THICKNESS)

        paint_cv3 = cv3.rectangle(cv3.zeros(100, 100), 50, 60, 20, 10, mode='xywh')
        paint_cv3_rel = cv3.rectangle(cv3.zeros(100, 100), 0.5, 0.6, 0.2, 0.1, mode='xywh')

        assert np.array_equal(paint_cv2, paint_cv3)
        assert np.array_equal(paint_cv2, paint_cv3_rel)

    def test_rect_mode_ccwh(self):
        paint_cv2 = np.zeros((100, 100), np.uint8)
        cv2.rectangle(paint_cv2, (40, 55), (60, 65), COLOR, THICKNESS)

        paint_cv3 = cv3.rectangle(cv3.zeros(100, 100), 50, 60, 20, 10, mode='ccwh')
        paint_cv3_rel = cv3.rectangle(cv3.zeros(100, 100), 0.5, 0.6, 0.2, 0.1, mode='ccwh')

        assert np.array_equal(paint_cv2, paint_cv3)
        assert np.array_equal(paint_cv2, paint_cv3_rel)

    def test_rect_overflow(self):
        for pt1, pt2 in [
            [(-110, 10), (80, 80)],
            [(10, -110), (80, 80)],
            [(10, 10), (110, 80)],
            [(10, 10), (80, 110)]
        ]:

            paint_cv2 = np.zeros((100, 100), np.uint8)
            cv2.rectangle(paint_cv2, pt1, pt2, COLOR, THICKNESS)

            paint_cv3 = cv3.rectangle(cv3.zeros(100, 100), *pt1, *pt2)

            paint_cv3_rel = cv3.rectangle(cv3.zeros(100, 100), pt1[0]/100, pt1[1]/100, pt2[0]/100, pt2[1]/100)
            paint_cv3_rel_true = cv3.rectangle(cv3.zeros(100, 100), pt1[0]/100, pt1[1]/100, pt2[0]/100, pt2[1]/100, rel=True)
            paint_cv3_rel_false = cv3.rectangle(cv3.zeros(100, 100), pt1[0]/100, pt1[1]/100, pt2[0]/100, pt2[1]/100, rel=False)

            assert np.array_equal(paint_cv2, paint_cv3)
            assert not np.array_equal(paint_cv2, paint_cv3_rel)
            assert np.array_equal(paint_cv2, paint_cv3_rel_true)
            assert not np.array_equal(paint_cv2, paint_cv3_rel_false)


    def test_rect_relative(self):
        paint_cv2 = np.zeros((100, 100), np.uint8)
        cv2.rectangle(paint_cv2, (20, 30), (80, 60), COLOR, THICKNESS)

        # xyxy
        paint_xyxy = cv3.rectangle(cv3.zeros(100, 100), 0.2, 0.3, 0.8, 0.6)
        assert np.array_equal(paint_xyxy, paint_cv2)

        # xywh
        paint_xywh = cv3.rectangle(cv3.zeros(100, 100), 0.2, 0.3, 0.6, 0.3, mode='xywh')
        assert np.array_equal(paint_xywh, paint_cv2)

        # xywh hard
        assert np.array_equal(
            cv2.rectangle(np.zeros((100, 100), np.uint8), (100, 100), (130, 140), COLOR, 10),
            cv3.rectangle(cv3.zeros(100, 100), 1.0, 1.0, 0.3, 0.4, t=10, mode='xywh')
        )

        # ccwh
        paint_ccwh = cv3.rectangle(cv3.zeros(100, 100), 0.5, 0.45, 0.6, 0.3, mode='ccwh')
        assert np.array_equal(paint_ccwh, paint_cv2)

    def test_rect_rel(self):
        # (0., 0., 1., 1.) rel=None
        assert np.array_equal(
            cv2.rectangle(np.zeros((100, 100), np.uint8), (0, 0), (100, 100), COLOR, 10),
            cv3.rectangle(cv3.zeros(100, 100), 0., 0., 1., 1., t=10)
        )
        # (0, 0, 1, 1) rel=None
        assert np.array_equal(
            cv2.rectangle(np.zeros((100, 100), np.uint8), (0, 0), (1, 1), COLOR, 10),
            cv3.rectangle(cv3.zeros(100, 100), 0, 0, 1, 1, t=10)
        )
        # (0., 0., 1., 1.) rel=True
        assert np.array_equal(
            cv2.rectangle(np.zeros((100, 100), np.uint8), (0, 0), (100, 100), COLOR, 10),
            cv3.rectangle(cv3.zeros(100, 100), 0., 0., 1., 1., rel=True, t=10)
        )
        # (0, 0, 1, 1) rel=True
        assert np.array_equal(
            cv2.rectangle(np.zeros((100, 100), np.uint8), (0, 0), (100, 100), COLOR, 10),
            cv3.rectangle(cv3.zeros(100, 100), 0, 0, 1, 1, rel=True, t=10)
        )
        # (0., 0., 1., 1.) rel=False
        assert np.array_equal(
            cv2.rectangle(np.zeros((100, 100), np.uint8), (0, 0), (1, 1), COLOR, 10),
            cv3.rectangle(cv3.zeros(100, 100), 0., 0., 1., 1., rel=False, t=10)
        )
        # (0, 0, 1, 1) rel=False
        assert np.array_equal(
            cv2.rectangle(np.zeros((100, 100), np.uint8), (0, 0), (1, 1), COLOR, 10),
            cv3.rectangle(cv3.zeros(100, 100), 0, 0, 1, 1, rel=False, t=10)
        )


class TestPolylines(Shape):
    pts = np.int32([
        [[10, 10]],
        [[70, 25]],
        [[60, 85]],
        [[20, 90]]
    ])
    def get_sample_cv2(self, color, t, line_type):
        return cv2.polylines(np.zeros((100, 100, 3), np.uint8), [self.pts], False, color, t, lineType=line_type)
    def get_sample_cv3(self, color, t, line_type):
        return cv3.polylines(cv3.zeros(100, 100, 3), self.pts, color=color, t=t, line_type=line_type)

    def test_basic(self):
        paint_cv2 = cv2.polylines(np.zeros((100, 100), np.uint8), [self.pts], False, COLOR, THICKNESS)
        paint_cv3 = cv3.polylines(cv3.zeros(100, 100), self.pts)
        assert np.array_equal(paint_cv2, paint_cv3)

    def test_poly_closed(self):
        paint_cv2 = cv2.polylines(np.zeros((100, 100), np.uint8), [self.pts], True, COLOR, THICKNESS)
        paint_cv3 = cv3.polylines(cv3.zeros(100, 100), self.pts, is_closed=True)
        assert np.array_equal(paint_cv2, paint_cv3)

    def test_poly_pts_formats(self):
        paint_cv2 = cv2.polylines(np.zeros((100, 100), np.uint8), [self.pts], False, COLOR, THICKNESS)

        paint_cv3_uint8 = cv3.polylines(cv3.zeros(100, 100), np.uint8(self.pts))
        paint_cv3_float = cv3.polylines(cv3.zeros(100, 100), np.float32(self.pts))
        paint_cv3_arr_of_pair = cv3.polylines(cv3.zeros(100, 100), self.pts.reshape(-1, 2))
        paint_cv3_ravel = cv3.polylines(cv3.zeros(100, 100), self.pts.ravel())
        paint_cv3_list = cv3.polylines(cv3.zeros(100, 100), self.pts.tolist())

        assert np.array_equal(paint_cv2, paint_cv3_uint8)
        assert np.array_equal(paint_cv2, paint_cv3_float)
        assert np.array_equal(paint_cv2, paint_cv3_arr_of_pair)
        assert np.array_equal(paint_cv2, paint_cv3_ravel)
        assert np.array_equal(paint_cv2, paint_cv3_list)

    def test_poly_rel(self):
        paint_cv2 = cv2.polylines(np.zeros((100, 100), np.uint8), [self.pts], False, COLOR, THICKNESS)
        paint_cv3_rel = cv3.polylines(cv3.zeros(100, 100), self.pts / 100)
        assert np.array_equal(paint_cv2, paint_cv3_rel)


class TestFillPoly:
    pts = np.int32([
        [[10, 10]],
        [[70, 25]],
        [[60, 85]],
        [[20, 90]]
    ])
    def test_basic(self):
        paint_cv2 = cv2.fillPoly(np.zeros((100, 100), np.uint8), [self.pts], COLOR)
        paint_cv3 = cv3.fill_poly(cv3.zeros(100, 100), self.pts)
        assert np.array_equal(paint_cv2, paint_cv3)

    def test_poly_pts_formats(self):
        paint_cv2 = cv2.fillPoly(np.zeros((100, 100), np.uint8), [self.pts], COLOR)

        paint_cv3_uint8 = cv3.fill_poly(cv3.zeros(100, 100), np.uint8(self.pts))
        paint_cv3_float = cv3.fill_poly(cv3.zeros(100, 100), np.float32(self.pts))
        paint_cv3_arr_of_pair = cv3.fill_poly(cv3.zeros(100, 100), self.pts.reshape(-1, 2))
        paint_cv3_ravel = cv3.fill_poly(cv3.zeros(100, 100), self.pts.ravel())
        paint_cv3_list = cv3.fill_poly(cv3.zeros(100, 100), self.pts.tolist())

        assert np.array_equal(paint_cv2, paint_cv3_uint8)
        assert np.array_equal(paint_cv2, paint_cv3_float)
        assert np.array_equal(paint_cv2, paint_cv3_arr_of_pair)
        assert np.array_equal(paint_cv2, paint_cv3_ravel)
        assert np.array_equal(paint_cv2, paint_cv3_list)

    def test_poly_rel(self):
        paint_cv2 = cv2.fillPoly(np.zeros((100, 100), np.uint8), [self.pts], COLOR)
        paint_cv3_rel = cv3.fill_poly(cv3.zeros(100, 100), self.pts / 100)
        assert np.array_equal(paint_cv2, paint_cv3_rel)

    def test_color(self):
        paint_cv2 = cv2.fillPoly(np.zeros((100, 100, 3), np.uint8), [self.pts], (250, 0, 190))
        paint_cv3 = cv3.fill_poly(cv3.zeros(100, 100, 3), self.pts, color=(250, 0, 190))
        assert np.array_equal(paint_cv2, paint_cv3)


class TestCircle(Shape):
    def get_sample_cv2(self, color, t, line_type):
        return cv2.circle(np.zeros((100, 100, 3), np.uint8), (50, 60), 30, color, t, lineType=line_type)
    def get_sample_cv3(self, color, t, line_type):
        return cv3.circle(cv3.zeros(100, 100, 3), 50, 60, 30, color=color, t=t, line_type=line_type)

    def test_basic(self):
        paint_cv2 = cv2.circle(np.zeros((100, 100), np.uint8), (50, 60), 30, COLOR, THICKNESS)
        paint_cv3 = cv3.circle(cv3.zeros(100, 100), 50, 60, 30)
        assert np.array_equal(paint_cv2, paint_cv3)

    def test_rel(self):
        paint_cv2 = cv2.circle(np.zeros((100, 100), np.uint8), (50, 60), 30, COLOR, THICKNESS)
        paint_cv3 = cv3.circle(cv3.zeros(100, 100), 0.5, 0.6, 30)
        assert np.array_equal(paint_cv2, paint_cv3)

    def test_r_float(self):
        paint_cv2 = cv2.circle(np.zeros((100, 100), np.uint8), (50, 60), 30, COLOR, THICKNESS)
        paint_cv3 = cv3.circle(cv3.zeros(100, 100), 50, 60, 30.2)
        assert np.array_equal(paint_cv2, paint_cv3)

class TestPoint:
    def test_basic(self):
        paint_cv2 = cv2.circle(np.zeros((100, 100), np.uint8), (50, 60), cv3.opt.PT_RADIUS, COLOR, -1)
        paint_cv3 = cv3.point(cv3.zeros(100, 100), 50, 60)
        assert np.array_equal(paint_cv2, paint_cv3)

    def test_r_zero(self):
        paint_cv2 = cv2.circle(np.zeros((100, 100), np.uint8), (50, 60), 0, COLOR, -1)
        paint_cv3 = cv3.point(cv3.zeros(100, 100), 50, 60, 0)
        assert np.array_equal(paint_cv2, paint_cv3)


class TestLine(Shape):
    def get_sample_cv2(self, color, t, line_type):
        return cv2.line(np.zeros((100, 100, 3), np.uint8), (25, 30), (70, 80), color, t, lineType=line_type)
    def get_sample_cv3(self, color, t, line_type):
        return cv3.line(cv3.zeros(100, 100, 3), 25, 30, 70, 80, color=color, t=t, line_type=line_type)

    def test_basic(self):
        paint_cv2 = cv2.line(np.zeros((100, 100), np.uint8), (25, 30), (70, 80), COLOR, THICKNESS)
        paint_cv3 = cv3.line(cv3.zeros(100, 100), 25, 30, 70, 80)
        paint_cv3_rel = cv3.line(cv3.zeros(100, 100), 0.25, 0.30, 0.70, 0.80)
        assert np.array_equal(paint_cv2, paint_cv3)
        assert np.array_equal(paint_cv2, paint_cv3_rel)

    def test_hline(self):
        paint_cv2 = cv2.line(np.zeros((100, 100), np.uint8), (0, 30), (100, 30), COLOR, THICKNESS)
        paint_cv3 = cv3.hline(cv3.zeros(100, 100), 30)
        paint_cv3_rel = cv3.hline(cv3.zeros(100, 100), 0.3)
        assert np.array_equal(paint_cv2, paint_cv3)
        assert np.array_equal(paint_cv2, paint_cv3_rel)

    def test_vline(self):
        paint_cv2 = cv2.line(np.zeros((100, 100), np.uint8), (60, 0), (60, 100), COLOR, THICKNESS)
        paint_cv3 = cv3.vline(cv3.zeros(100, 100), 60)
        paint_cv3_rel = cv3.vline(cv3.zeros(100, 100), 0.6)
        assert np.array_equal(paint_cv2, paint_cv3)
        assert np.array_equal(paint_cv2, paint_cv3_rel)


class TestText(Shape):
    def get_sample_cv2(self, color, t, line_type):
        return cv2.putText(np.zeros((100, 100, 3), np.uint8), 'abc', (30, 40), fontFace=cv3.opt.FONT, fontScale=cv3.opt.SCALE, color=color, thickness=t, lineType=line_type)
    def get_sample_cv3(self, color, t, line_type):
        return cv3.text(cv3.zeros(100, 100, 3), 'abc', 30, 40, color=color, t=t, line_type=line_type)

    def test_basic(self):
        paint_cv2 = cv2.putText(np.zeros((100, 100), np.uint8), 'abc', (30, 40), fontFace=cv3.opt.FONT, fontScale=cv3.opt.SCALE, color=COLOR, thickness=THICKNESS)
        paint_cv3 = cv3.text(cv3.zeros(100, 100), 'abc', 30, 40)
        paint_cv3_rel = cv3.text(cv3.zeros(100, 100), 'abc', 0.3, 0.4)
        assert np.array_equal(paint_cv2, paint_cv3)
        assert np.array_equal(paint_cv2, paint_cv3_rel)

    def test_font(self):
        paint_cv2 = cv2.putText(np.zeros((100, 100), np.uint8), 'abc', (30, 40), fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=cv3.opt.SCALE, color=COLOR, thickness=THICKNESS)
        font_flag = cv3.text(cv3.zeros(100, 100), 'abc', 30, 40, font=cv2.FONT_HERSHEY_PLAIN)
        font_str_flag = cv3.text(cv3.zeros(100, 100), 'abc', 30, 40, font='plain')
        assert np.array_equal(paint_cv2, font_flag)
        assert np.array_equal(paint_cv2, font_str_flag)

    def test_scale(self):
        paint_cv2 = cv2.putText(np.zeros((100, 100), np.uint8), 'abc', (30, 40), fontFace=cv3.opt.FONT, fontScale=0.9, color=COLOR, thickness=THICKNESS)
        paint_cv3 = cv3.text(cv3.zeros(100, 100), 'abc', 30, 40, scale=0.9)
        assert np.array_equal(paint_cv2, paint_cv3)

    def test_flip(self):
        paint_cv2 = cv2.putText(np.zeros((100, 100), np.uint8), 'abc', (30, 40), fontFace=cv3.opt.FONT,
                                fontScale=cv3.opt.SCALE, color=COLOR, thickness=THICKNESS, bottomLeftOrigin=True)
        paint_cv3 = cv3.text(cv3.zeros(100, 100), 'abc', 30, 40, flip=True)
        assert np.array_equal(paint_cv2, paint_cv3)

