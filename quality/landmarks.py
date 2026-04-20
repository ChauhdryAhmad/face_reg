from config import REQUIRED_LANDMARKS


def check_landmarks(lm):

    if not lm:
        return False,0

    c = len(lm)

    return c >= REQUIRED_LANDMARKS, c