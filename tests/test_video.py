import numpy as np
import cv2
import cv3
from pathlib import Path
import os
import shutil
import pytest


TEST_VID = 'vid.mp4'
NON_EXIST_VID = 'spamvid.mp4'
OUT_PATH_VID = 'output.mp4'
OUT_PATH_VID_AVI = 'output.avi'
INVALID_EXT_PATH = 'vid.mp4xxx'
TMP_DIR = 'temp/'
IMGSEQ_DIR = 'imgseq/'
TESTS_DIR = 'tests'
frame = cv3.imread('img.jpeg')


def test_capture_open_read():
    # VideoCapture
    cap = cv3.VideoCapture(TEST_VID)
    cap.read()
    cap.close()

    # Video
    cap = cv3.Video(TEST_VID)
    cap.read()
    cap.close()

    # using context manager
    with cv3.Video(TEST_VID) as cap:
        cap.read()



def test_capture_read():
    cap_cv2 = cv2.VideoCapture(TEST_VID)
    cap = cv3.Video(TEST_VID)

    for i in range(5):
        _, frame1 = cap_cv2.read()
        frame2 = cap.read()
        assert np.array_equal(frame1, cv3.rgb2bgr(frame2))


def test_capture_open_path():
    path = Path(TEST_VID)
    with cv3.Video(path):
        pass

def test_capture_open_path_nonexist():
    with pytest.raises(OSError):
        cv3.Video(NON_EXIST_VID)

    # using Path
    with pytest.raises(FileNotFoundError):
        cv3.Video(Path(NON_EXIST_VID))


def test_capture_open_webcam():
    with cv3.Video(0):
        pass

    with cv3.Video('0'):
        pass


@pytest.fixture()
def imgseq_fixture():
    imgseq_dir = Path(IMGSEQ_DIR)
    imgseq_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(cv3.Video(TEST_VID)):
        cv3.imwrite(imgseq_dir / f'img{i//30:02d}.png', frame)
    yield
    shutil.rmtree(IMGSEQ_DIR, ignore_errors=True)


@pytest.mark.usefixtures('imgseq_fixture')
def test_capture_open_imgseq():
    imgseq = os.path.join(IMGSEQ_DIR, 'img%02d.png')

    cap = cv3.Video(imgseq)
    cap.read()
    cap.close()


@pytest.fixture()
def invalid_ext_fixture():
    Path(INVALID_EXT_PATH).touch()
    yield
    Path(INVALID_EXT_PATH).unlink()


@pytest.mark.usefixtures('invalid_ext_fixture')
def test_capture_invalid_ext():
    with pytest.raises(OSError):
        cv3.Video(INVALID_EXT_PATH)


def test_capture_open_dir():
    with pytest.raises(IsADirectoryError):
        cv3.Video(TESTS_DIR)


def test_capture_closed():
    cap = cv3.Video(TEST_VID)
    cap.close()
    assert not cap.stream.isOpened()
    with pytest.raises(OSError):
        cap.read()


def test_capture_num_frame():
    with cv3.Video(TEST_VID) as cap:
        print(cap.now)
        for i in range(5):
            cap.read()

        assert cap.now == 5


def test_capture_for():
    cap_cv2 = cv2.VideoCapture(TEST_VID)
    cap = cv3.Video(TEST_VID)

    for i, frame in enumerate(cap):
        _, frame_cv2 = cap_cv2.read()
        assert np.array_equal(frame_cv2, cv3.rgb2bgr(frame))
        if i == 6:
            break


def test_capture_finished():
    cap = cv3.Video(TEST_VID)
    for _ in cap:
        pass
    with pytest.raises(StopIteration):
        cap.read()

    assert len(cap) == cap.now


def test_capture_rewind():
    cap_cv2 = cv2.VideoCapture(TEST_VID)
    cap = cv3.Video(TEST_VID)

    for i in range(5):
        _, frame1 = cap_cv2.read()

    cap.rewind(4)
    frame2 = cap.read()
    assert np.array_equal(frame1, cv3.rgb2bgr(frame2))


def test_video_extra_kw():
    with pytest.raises(TypeError):
        cv3.Video(TEST_VID, fps=30)


@pytest.fixture()
def out_path_fixture():
    Path(OUT_PATH_VID).unlink(missing_ok=True)
    Path(OUT_PATH_VID_AVI).unlink(missing_ok=True)
    yield
    Path(OUT_PATH_VID).unlink(missing_ok=True)
    Path(OUT_PATH_VID_AVI).unlink(missing_ok=True)


class TestWriterOpenWrite:
    @pytest.mark.usefixtures('out_path_fixture')
    def test_using_videowriter(self):
        out = cv3.VideoWriter(OUT_PATH_VID)
        out.write(frame)
        out.close()
        assert os.path.isfile(OUT_PATH_VID)

    @pytest.mark.usefixtures('out_path_fixture')
    def test_using_video(self):
        out = cv3.Video(OUT_PATH_VID, 'w')
        out.write(frame)
        out.close()
        assert os.path.isfile(OUT_PATH_VID)

    @pytest.mark.usefixtures('out_path_fixture')
    def test_using_with(self):
        with cv3.Video(OUT_PATH_VID, 'w') as out:
            out.write(frame)
        assert os.path.isfile(OUT_PATH_VID)


@pytest.mark.usefixtures('out_path_fixture')
def test_writer_open_path():
    path = Path(OUT_PATH_VID)
    with cv3.Video(path, 'w') as out:
        out.write(frame)
    assert path.is_file()


@pytest.mark.usefixtures('out_path_fixture')
def test_writer_closed():
    with cv3.Video(OUT_PATH_VID, 'w') as out:
        out.write(frame)
    assert os.path.isfile(OUT_PATH_VID)


@pytest.mark.usefixtures('out_path_fixture')
def test_writer_write_correct():
    with cv3.Video(TEST_VID) as cap:
        with cv3.Video(OUT_PATH_VID_AVI, 'w', fourcc='RGBA') as out:
            for i, frame in enumerate(cap):
                if i == 5:
                    break
                out.write(frame)

        assert os.path.isfile(OUT_PATH_VID_AVI)
        cap.rewind(0)

        with cv3.Video(OUT_PATH_VID_AVI) as new_cap:
            assert len(new_cap) == 5
            for frame1, frame2 in zip(cap, new_cap):
                assert np.array_equal(frame1, frame2)


@pytest.mark.usefixtures('out_path_fixture')
def test_writer_fourcc():
    # using str
    with cv3.Video(OUT_PATH_VID, 'w', fourcc='MP4V') as out:
        out.write(frame)

    # using VideoWriter_fourcc
    with cv3.Video(OUT_PATH_VID, 'w', fourcc=cv2.VideoWriter_fourcc(*'MP4V')) as out:
        out.write(frame)


@pytest.mark.usefixtures('out_path_fixture')
def test_writer_fps():
    FPS = 15
    with cv3.Video(OUT_PATH_VID, 'w', fps=FPS) as out:
        out.write(frame)

    new_cap = cv3.Video(OUT_PATH_VID)
    assert new_cap.fps == FPS


def test_writer_write2closed():
    # writing not started
    with cv3.Video(OUT_PATH_VID, 'w') as out:
        pass

    # writing started once
    with cv3.Video(OUT_PATH_VID, 'w') as out:
        out.write(frame)
    with pytest.raises(OSError):
        out.write(frame)


@pytest.fixture()
def tmpdir_fixture():
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    yield
    shutil.rmtree(TMP_DIR, ignore_errors=True)


@pytest.mark.usefixtures('tmpdir_fixture')
def test_writer_mkdir():
    out_path = os.path.join(TMP_DIR, OUT_PATH_VID)
    with cv3.Video(out_path, 'w', mkdir=True) as out:
        out.write(frame)
    assert os.path.isfile(out_path)


@pytest.mark.usefixtures('tmpdir_fixture')
def test_writer_nomkdir():
    out_path = os.path.join(TMP_DIR, OUT_PATH_VID)
    with cv3.Video(out_path, 'w') as out:
        with pytest.raises(OSError):
            out.write(frame)
