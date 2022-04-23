from pathlib import Path
import numpy as np
import cv2
from cv2 import VideoCapture as BaseVideoCapture, VideoWriter as BaseVideoWriter

from . import opt
from .color_spaces import rgb
from ._utils import typeit

__all__ = [
    'VideoCapture',
    'VideoWriter',
    'VideoReader',
    'Video'
]

class VideoCapture(BaseVideoCapture):
    def __init__(self, src):
        if isinstance(src, str) and src.isdecimal():
            src = int(src)
        elif isinstance(src, (str, Path)):
            if not Path(src).is_file():
                raise FileNotFoundError(str(src))
            src = str(src)
        super().__init__(src)
        assert self.isOpened(), f"Video {src} didn't open"
        self.frame_cnt = round(self.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = round(self.get(cv2.CAP_PROP_FPS))
        self.width = round(self.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = round(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.shape = self.width, self.height

    @property
    def now(self):  # current frame
        return round(self.get(cv2.CAP_PROP_POS_FRAMES))

    def read(self):
        assert self.isOpened(), f"Video is closed"
        _, frame = super().read()
        if frame is None:
            raise StopIteration('Video has finished')
        if opt.RGB:
            frame = rgb(frame)
        return frame

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.read()
        return frame

    def rewind(self, nframe):
        assert isinstance(nframe, int) or (isinstance(nframe, float) and nframe.is_integer())
        assert nframe in range(0, len(self))
        # if 0 <= nframe <= 1:

        self.set(cv2.CAP_PROP_POS_FRAMES, nframe)
        return self

    def __len__(self):
        return self.frame_cnt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    __getitem__ = rewind
    close = BaseVideoCapture.release


class VideoWriter(BaseVideoWriter):
    def __init__(self, save_path, fps=opt.FPS, fourcc=cv2.VideoWriter_fourcc(*opt.FOURCC)):  # TODO Nones
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_path = save_path
        self.started = False
        self.width = None
        self.height = None
        self.fps = fps
        if isinstance(fourcc, str):
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.fourcc = fourcc

    @property
    def shape(self):
        return self.width, self.height

    def write(self, frame: np.ndarray):
        if not self.started:
            self.started = True
            self.height, self.width = frame.shape[:2]
            super().__init__(self.save_path, self.fourcc, self.fps, (self.width, self.height))
        assert (self.height, self.width) == frame.shape[:2], f'Shape mismatch. Required: {self.shape}'
        if opt.RGB:
            frame = rgb(frame)
        frame = typeit(frame)
        super().write(frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    close = BaseVideoWriter.release


def Video(path, mode='r', **kwds):
    assert mode in 'rw'
    if mode == 'r':
        base_class = VideoCapture
    elif mode == 'w':
        base_class = VideoWriter
    return base_class(path, **kwds)


VideoReader = VideoCapture
