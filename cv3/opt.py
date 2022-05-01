import cv2

RGB = True
FPS = 30
FOURCC = cv2.VideoWriter_fourcc(*'MP4V')
THICKNESS = 3
COLOR = 255


def set_rgb():
    global RGB
    RGB = True


def set_bgr():
    global RGB
    RGB = False


def video(fps=None, fourcc=None):
    global FPS, FOURCC
    if fps:
        FPS = fps
    if fourcc:
        FOURCC = cv2.VideoWriter_fourcc(*fourcc) if isinstance(fourcc, str) else fourcc
