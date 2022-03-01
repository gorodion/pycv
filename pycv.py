import numpy as np
from pathlib import Path
import cv2
from functools import partial
from cv2 import VideoCapture as BaseVideoCapture, VideoWriter as BaseVideoWriter
# from cv2 import *

def _type(img):
    if isinstance(img, list):
        img = np.array(img, 'uint8')
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported type: {type(img)}")
    assert img.ndim == 3 and img.shape[-1] == 3 or img.ndim == 2, f'Incorrect image shape: {img.shape}'  # TODO сравнить условия из scikit-image/matplotlib

    if img.dtype == np.uint8:
        return img
    # if issubclass(img.dtype.type, np.floating):
        # TODO if img.max() <= 1:
        # return img.round().astype('uint8')
    # if issubclass(img.dtype.type, np.integer):
    if issubclass(img.dtype.type, np.number):
        return img.astype('uint8')
    if img.dtype == np.bool:
        return 255 * img.astype('uint8')
    raise TypeError(f"Unsupported dtype: {img.dtype}")

# def _type_decorator

RGB = False

# Color Spaces

def rgb(img: np.ndarray):
    if img.ndim != 3:  # only if 3-color image
        return img
    return cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)

bgr = rgb

bgr2gray = partial(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY)
rgb2gray = partial(cv2.cvtColor, code=cv2.COLOR_RGB2GRAY)
gray2rgb = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2RGB)
gray2bgr = partial(cv2.cvtColor, code=cv2.COLOR_GRAY2BGR)

bgr2hsv = partial(cv2.cvtColor, code=cv2.COLOR_BGR2HSV)
rgb2hsv = partial(cv2.cvtColor, code=cv2.COLOR_RGB2HSV)
hsv2bgr = partial(cv2.cvtColor, code=cv2.COLOR_HSV2BGR)
hsv2rgb = partial(cv2.cvtColor, code=cv2.COLOR_HSV2RGB)


# Reading/Writing
def _imread_flag_match(flag):
    assert flag in ('color', 'gray', 'alpha')
    if flag == 'color':
        flag = cv2.IMREAD_COLOR
    elif flag == 'gray':
        flag = cv2.IMREAD_GRAYSCALE
    elif flag == 'alpha':
        flag = cv2.IMREAD_UNCHANGED
    return flag

# TODO проверять кириллицу
def imread(imgp, flag=cv2.IMREAD_COLOR):
    if not Path(imgp).is_file():
        raise FileNotFoundError(str(imgp))
    if isinstance(imgp, Path):
        imgp = str(imgp)
    if isinstance(flag, str):
        flag = _imread_flag_match(flag)
    img = cv2.imread(imgp, flag)
    assert img is not None, f'File was not read: {imgp}'
    if RGB:
        img = rgb(img)
    return img


def imwrite(imgp, img, **kwargs):
    Path(imgp).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(imgp, Path):
        imgp = str(imgp)
    if RGB:
        img = rgb(img)
    assert cv2.imwrite(imgp, img), 'Something went wrong'
    

# TODO добавить waitkey и &&    
# def imshow(window_name, image):

    
# Drawing
def _draw_decorator(func):
    def wrapper(img, *args, color=255, copy=False, **kwargs):
        img = _type(img)
        if copy:
            img = img.copy()

        # TODO if 0 < color < 1
        # color
        if isinstance(color, np.ndarray):
            color = color.tolist()
        if isinstance(color, (list, tuple)):
            color = tuple(map(int, color))
        else:
            color = int(color)

        # other kw arguments
        for k, v in kwargs.items():
            kwargs[k] = int(v)

        return func(img, *args, color=color, **kwargs)
    return wrapper


# TODO type: `xyxy` and `xywh` and `ccwh`
@_draw_decorator
def rectangle(img, x0, y0, x1, y1, color=255, t=3):
    cv2.rectangle(img, (x0, y0), (x1, y1), color, t)
    return img

@_draw_decorator
def circle(img, x0, y0, r, color=255, t=3):
    cv2.circle(img, (x0, y0), r, color, t)
    return img

@_draw_decorator
def point(img, x0, y0, color=255):
    cv2.circle(img, (x0, y0), 0, color, -1)
    return img

@_draw_decorator
def line(img, x0, y0, x1, y1, color=255, t=3):
    cv2.line(img, (x0, y0), (x1, y1), color, t)
    return img

@_draw_decorator
def hline(img, y, color=255, t=3):
    w = img.shape[1]
    cv2.line(img, (0, y), (w, y), color, t)
    return img

@_draw_decorator
def vline(img, x, color=255, t=3):
    h = img.shape[0]
    cv2.line(img, (x, 0), (x, h), color, t)
    return img
    

@_draw_decorator
def putText(img, text, x=0, y=-1, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=255, t=3, line_type=cv2.LINE_AA, flip=False):
    h = img.shape[0]
    if y == -1:
        y = h // 2
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        t,
        line_type,
        flip
    )
    return img

text = putText

# Transformations

def vflip(img):
    return cv2.flip(img, 0)


def hflip(img):
    return cv2.flip(img, 1)

# diagonal flip
def dflip(img):
    return cv2.flip(img, -1)

# TODO flags
def transform(img, angle, scale):
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, scale)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def rotate(img, angle):
    return transform(img, angle, 1)

def scale(img, factor):
    return transform(img, 0, factor)

def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)

shift = translate

# TODO interpolation
def resize(img, width, height):
    return cv2.resize(img, (width, height))

# TODO flags
def threshold(img: np.ndarray, thr=127, max=255):
    assert img.ndim == 2, '`img` should be gray image'
    # TODO if img.max() < 1
    _, thresh = cv2.threshold(img, thr, max, cv2.THRESH_BINARY)
    return thresh

# Video
# TODO __enter__ and __exit__
class VideoCapture(BaseVideoCapture):
  def __init__(self, src):
    if isinstance(src, Path):
        src = str(src)
    if src == '0':
        src = 0
    super().__init__(src)
    #assert self.isOpened(), f"Video {src} didn't open"
    self.frame_cnt = self.get(cv2.CAP_PROP_FRAME_COUNT)
    self.fps = self.get(cv2.CAP_PROP_FPS)
    self.width  = self.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.height = self.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.i = 0 # Current frame

# TODO Raise an exception if closed?
  def read(self):
      _, frame = super().read()
      if RGB:
          frame = rgb(frame)
      return frame
    
  def __iter__(self):
    return self

  def __next__(self):
      frame = self.read()
      if frame is None:
        raise StopIteration
      self.i += 1
      return frame
      
  def rewind(self, nframe):
    assert isinstance(nframe, int) or (isinstance(nframe, float) and nframe.is_integer())
    assert nframe in range(0, len(self))
    self.set(cv2.CAP_PROP_POS_FRAMES, nframe) # TODO what if float
    self.i = nframe
    
  def __len__(self):
      return self.frame_cnt

  def close(self):
      self.release()
      
VideoReader = VideoCapture
      
class VideoWriter(BaseVideoWriter):
    def __init__(self, save_path, fps=30, fourcc=cv2.VideoWriter_fourcc(*'MP4V')):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_path = save_path
        self.started = False
        self.width = None
        self.height = None
        self.fps = fps
        self.fourcc = fourcc
        
    def write(self, frame: np.ndarray):
        if not self.started:
            self.started = True
            self.height, self.width = frame.shape[:2]
            super().__init__(self.save_path, self.fourcc, self.fps, (self.width, self.height))
        assert self.height, self.width == frame.shape[:2]
        if RGB:
            frame = rgb(frame)
        super().write(frame)

    def close(self):
        self.release()


class Video:
    def __init__(self, path, mode='r', **kwds):
        assert mode in 'rw'
        if mode == 'r':
            base_class = VideoReader
        elif mode == 'w':
            base_class = VideoWriter
            
        [setattr(self, name, func) for name, func in base_class.__dict__.items() if not name.startswith('__')]
        base_class.__init__(path, **kwds)

if __name__ == '__main__':
    img = np.zeros((200, 200), 'uint8')
    text(img, 'aaaa')
    imwrite('cba.png', img)