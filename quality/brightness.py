import cv2
from config import MIN_MEAN_BRIGHTNESS


def check_brightness(frame, bbox):

    x1,y1,x2,y2 = map(int,bbox)

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return False,0

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    m = gray.mean()

    return m >= MIN_MEAN_BRIGHTNESS, m