import cv3

def test_capture_open_webcam():
    with cv3.Video(0):
        pass

    with cv3.Video('0'):
        pass