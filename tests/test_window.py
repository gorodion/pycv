import cv2
import cv3
from PIL import Image
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

def test_imshow():
    window_name = 'parrot_rgb'
    cv3.imshow(window_name, test_img)
    cv3.wait_key(2000)
    cv2.destroyWindow(window_name)


def test_imshow_bgr():
    window_name = 'parrot_bgr'
    try:
        cv3.opt.set_bgr()
        cv3.imshow(window_name, test_img)
        cv3.wait_key(2000)
    finally:
        cv3.opt.set_rgb()
        cv2.destroyWindow(window_name)


def test_imshow_window():
    window_name = 'parrot_rgb Window'
    w = cv3.Window(window_name)
    w.imshow(test_img)
    w.wait_key(2000)
    w.close()

    # using context manager
    with cv3.Window('spam'):
        pass


def test_imshow_pil():
    pil_img = Image.open(TEST_IMG)
    with cv3.Window('PIL Image') as w:
        w.imshow(pil_img)
        w.wait_key(2000)


def test_imshow_window_pos():
    with (
        cv3.Window('pos 0,0', pos=(0,0)) as w1,
        cv3.Window('pos 960,540', pos=(480,270)) as w2,
        cv3.Window('pos 0,700', pos=(0,540)) as w3,
    ):
        for w in w1, w2, w3:
            w.imshow(test_img)
        cv3.wait_key(2000)


def test_window_move():
    secs = 2
    with cv3.Window(f'Move in {secs} secs', pos=(0,0)) as w:
        w.imshow(test_img)
        w.wait_key(secs * 1000)
        w.move(100, 100)
        w.wait_key(secs * 1000)


def test_imshow_window_noname():
    w1 = cv3.Window(pos=(0,0))
    w2 = cv3.Window(pos=(100,100))
    w3 = cv3.Window('another window without name', pos=(200,200))
    for w in w1, w2, w3:
        w.imshow(test_img)
    cv3.wait_key(2000)
    cv2.destroyAllWindows()


def test_imshow_window_bgr():
    try:
        cv3.opt.set_bgr()
        with cv3.Window('parrot_bgr (Window)') as w:
            w.imshow(test_img)
            w.wait_key(2000)
    finally:
        cv3.opt.set_rgb()


def test_specific_wait_key():
    start = time.time()
    with cv3.Window('Exit on `q` only') as w:
        w.imshow(test_img)
        while True:
            if w.wait_key(1) == ord('q'):
                break
            # break if nothing happens for 3 seconds
            if time.time() - start > 3:
                break
