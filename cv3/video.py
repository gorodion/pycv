import warnings
from pathlib import Path
import numpy as np
import cv2
from cv2 import VideoCapture as BaseVideoCapture, VideoWriter as BaseVideoWriter
from typing import Union

from . import opt
from .color_spaces import rgb
from ._utils import typeit

__all__ = [
    'VideoCapture',
    'VideoWriter',
    'VideoReader',
    'Video'
]


class VideoInterface:
    stream = None
    width = None
    height = None

    def isOpened(self):
        return self.stream.isOpened()

    @property
    def is_opened(self):
        return self.isOpened()

    def release(self):
        self.stream.release()

    @property
    def shape(self):
        return self.width, self.height

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    close = release


class VideoCapture(VideoInterface):
    def __init__(self, src: Union[Path, str, int]):
        if isinstance(src, str) and src.isdecimal():
            src = int(src)
        elif isinstance(src, (str, Path)) and Path(src).is_dir():
            raise IsADirectoryError(str(src))
        elif isinstance(src, Path):
            if not src.is_file():
                raise FileNotFoundError(str(src))
            src = str(src)
        self.stream = BaseVideoCapture(src)
        if not self.is_opened:
            raise OSError(f"Video from source {src} didn't open")
        self.frame_cnt = round(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = round(self.stream.get(cv2.CAP_PROP_FPS))
        self.width = round(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = round(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def now(self):  # current frame
        if not self.is_opened:
            raise OSError('Video is closed')
        return round(self.stream.get(cv2.CAP_PROP_POS_FRAMES))

    def read(self):
        if not self.is_opened:
            raise OSError(f"Video is closed")
        _, frame = self.stream.read()
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
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, nframe)
        return self

    def __len__(self):
        if self.frame_cnt < 0:
            return 0
        return self.frame_cnt

    def __getitem__(self, idx):
        self.rewind(idx)
        frame = self.read()
        return frame

    imread = read


class VideoWriter(VideoInterface):
    def __init__(self, save_path, fps=None, fourcc=None, mkdir=False):
        if mkdir:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(save_path, Path):
            save_path = str(save_path)
        self.save_path = save_path
        self.width = None
        self.height = None
        self.fps = fps or opt.FPS
        fourcc = fourcc or opt.FOURCC
        if isinstance(fourcc, str):
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.fourcc = fourcc
        self.stream = None

    def isOpened(self):
        if self.stream is None:
            return False
        return super().isOpened()

    def release(self):
        if self.stream is None:
            warnings.warn("Stream not started")
            return
        super().release()

    def write(self, frame: np.ndarray):
        frame = typeit(frame)
        if self.stream is None:
            self.height, self.width = frame.shape[:2]
            self.stream = BaseVideoWriter(self.save_path, self.fourcc, self.fps, (self.width, self.height))
        if not self.is_opened:
            raise OSError(f"Stream is closed")
        assert (self.height, self.width) == frame.shape[:2], f'Shape mismatch. Required: {self.shape}'
        if opt.RGB:
            frame = rgb(frame)
        self.stream.write(frame)

    imwrite = write


def Video(path, mode='r', **kwds):
    assert mode in 'rw'
    if mode == 'r':
        if kwds:
            raise TypeError(
                "VideoCapture doesn't accept keyword args. If you need VideoWriter then pass mode='w'"
            )
        return VideoCapture(path)
    elif mode == 'w':
        return VideoWriter(path, **kwds)


VideoReader = VideoCapture
