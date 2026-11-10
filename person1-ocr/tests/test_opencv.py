import cv2


def test_opencv_is_working():
    print("OpenCV version:", cv2.__version__)
    assert isinstance(cv2.__version__, str) and cv2.__version__
