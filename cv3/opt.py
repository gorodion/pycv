import cv2
import numpy as np

RGB = True
FPS = 30
FOURCC = 'mp4v'
THICKNESS = 1
COLOR = 255
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 1
PT_RADIUS = 1

def set_rgb():
    global RGB
    RGB = True


def set_bgr():
    global RGB
    RGB = False


def video(fps=None, fourcc=None):
    global FPS, FOURCC
    if fps is not None:
        assert fps > 0, 'default fps must be more 0'
        FPS = fps
    if fourcc is not None:
        # TODO asserts flags
        FOURCC = cv2.VideoWriter_fourcc(*fourcc) if isinstance(fourcc, str) else fourcc


def draw(thickness=None, color=None, font=None, pt_radius=None, scale=None, line_type=None):
    global THICKNESS, COLOR, FONT, PT_RADIUS, SCALE, LINE_TYPE
    if thickness is not None:
        assert isinstance(thickness, (int, np.unsignedinteger)), 'default thickness must be positive integer'
        THICKNESS = thickness
    if color is not None:
        assert isinstance(color, (str, int, float, np.unsignedinteger, np.floating, np.ndarray, list, tuple))
        COLOR = color
    if font is not None:
        # TODO asserts flags
        FONT = font
    if pt_radius is not None:
        # TODO asserts
        PT_RADIUS = pt_radius
    if scale is not None:
        # TODO asserts
        SCALE = scale
    if line_type is not None:
        # TODO asserts flags
        LINE_TYPE = line_type
