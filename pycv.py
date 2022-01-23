import numpy as np
from pathlib import Path
import cv2
from functools import partial
from cv2 import VideoCapture as BaseVideoCapture, VideoWriter as BaseVideoWriter
# from cv2 import *

def _type(img):
    if isinstance(img, list):
        img = np.array(img, 'uint8')
    assert img.ndim == 3 and img.shape[-1] == 3 or img.ndim == 2, f'Incorrect image shape: {img.shape}' # TODO сравнить условия из scikit-image/matplotlib
    
    if isinstance(img, np.ndarray):
        if img.dtype == np.uint8
            return img
        if issubclass(img.dtype.type, np.floating):
            return img.round().astype('uint8')
#        if img.max() <= 1:
        if issubclass(img.dtype.type, np.integer) or img.dtype == np.bool:
            return img.astype('uint8')
#        return img.astype('uint8')
        raise TypeError(f"Unsupported dtype: {img.dtype}")
    raise TypeError(f"Unsupported type: {type(img)}")
        

# Color Spaces

def rgb(img: np.ndarray):
    return img[...,::-1]
    
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
# TODO imread with type `rgb`, `bgr`, `gray`


# TODO проверять кириллицу
def imread(imgp, flag=cv2.IMREAD_COLOR):
    if not Path(imgp).is_file():
        raise FileNotFoundError(str(imgp))
    if isinstance(imgp, Path):
        imgp = str(imgp)
    img = cv2.imread(imgp, flag)
    assert img is not None, f'File was not read: {imgp}'
    return img
    
def imwrite(imgp, img, **kwargs)
    Path(imgp).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(imgp, Path):
        imgp = str(imgp)
        
    assert cv2.imwrite(imgp, img), 'Something went wrong'
    

# TODO добавить waitkey и &&    
# def imshow(window_name, image):

    
# Drawing
def _draw_decorator(func):
    def wrapper(img, *args, color=255, t=3, copy=False):
        img = _type(img)
        if copy:
            img = img.copy()
        if isinstance(color, np.ndarray): 
            color = color.tolist()
        return func(img, *args, color, t, copy)
    return wrapper


# TODO type: `xyxy` and `xywh` and `xywh`
def rectangle(img, x0, y0, x1, y1, color=255, t=3, copy=False):
    cv2.rectangle(img, (x0, y0), (x1, y1), color, t)
    return img
    
def circle(img, x0, y0, r, color=255, t=3, copy=False):
    cv2.circle(img, (x0, y0), r, color, t)
    return img
    
def line(img, x0, y0, x1, y1, color=255, t=3, copy=False):
    cv2.line(img, (x0, y0), (x1, y1), color, t)
    return img
    
def hline(img, y, color=255, t=3, copy=False):
    w = img.shape[1]
    return line(img, 0, y, w, y, color=color, t=t, copy=copy) # TODO kwargs
    
def vline(img, x, color=255, t=3, copy=False):
    h = img.shape[0]
    return line(img, x, 0, x, h, color=color, t=t, copy=copy) # TODO kwargs
    
def vflip(img):
    return cv2.flip(img, 0)
    
def hflip(img):
    return cv2.flip(img, 1)
    
# diagonal flip
def dflip(img):
    return cv2.flip(img, -1)
    
def text(): # TODO implement
    pass
    
# Transformations
    
def rotate(img, angle):
  img_center = tuple(np.array(img.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
  result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
  
#def shift():
#    pass


# def threshold

# Video

class VideoCapture(BaseVideoCapture):
  def __init__(self, src):
    if isinstance(src, Path):
        src = str(src)
    super().__init__(src) # TODO if '0'
    #assert self.isOpened(), f"Video {src} didn't open"
    self.frame_cnt = self.get(cv2.CAP_PROP_FRAME_COUNT)
    self.fps = self.get(cv2.CAP_PROP_FPS)
    self.width  = self.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.height = self.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.i = 0 # Current frame

# TODO Raise an exception if closed?
  def read(self):
      _, frame = super().read()
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
      
VideoReader = VideoCapture
      
class VideoWriter(BaseVideoWriter):
    def __init__(self, save_path, fps=30, fourcc=cv2.VideoWriter_fourcc(*'MP4V')):
        self.save_path = save_path
        self.out = None
        self.width = None
        self.height = None
        self.fps = fps
        self.fourcc = fourcc
        
    def write(self, frame: np.ndarray):
        if self.out is None:
            self.height, self.width = frame.shape[:2]
            self.out = super().__init__(self.save_path, self.fourcc, self.fps, (self.width, self.height))
        assert self.height, self.width == frame.shape[:2]
        self.out.write(frame)

class Video:
    def __init__(self, path, mode='r', **kwds):
        assert mode in 'rw'
        if mode == 'r':
            base_class = VideoReader
        elif mode == 'w':
            base_class = VideoWriter
            
        [setattr(self, name, func) for name, func in base_class.__dict__.items() if not name.startswith('__')]
        base_class.__init__(path, **kwds)