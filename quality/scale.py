from config import MIN_FACE_AREA_RATIO


def check_scale(bbox, shape):

    fa = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    ra = fa / (shape[0]*shape[1])

    return ra >= MIN_FACE_AREA_RATIO, ra