def iou(a,b):

    xA = max(a[0],b[0])
    yA = max(a[1],b[1])
    xB = min(a[2],b[2])
    yB = min(a[3],b[3])

    inter = max(0,xB-xA)*max(0,yB-yA)

    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])

    union = areaA + areaB - inter

    if union==0:
        return 0

    return inter/union


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def center_in_bbox(bbox, container):
    cx, cy = bbox_center(bbox)
    x1, y1, x2, y2 = container
    return x1 <= cx <= x2 and y1 <= cy <= y2


def expand_face_to_person_bbox(face_bbox, frame_shape):
    x1, y1, x2, y2 = map(int, face_bbox)
    h, w = frame_shape[:2]
    face_h = max(y2 - y1, 1)
    face_w = max(x2 - x1, 1)

    px1 = max(0, x1 - int(face_w * 0.5))
    py1 = max(0, y1 - int(face_h * 0.3))
    px2 = min(w, x2 + int(face_w * 0.5))
    py2 = min(h, y2 + int(face_h * 4.5))

    return [px1, py1, px2, py2]


def match_face_to_person(face_bbox, person_detections):
    best = None
    best_area = None

    for person in person_detections:
        pbbox = person["bbox"]
        if not center_in_bbox(face_bbox, pbbox):
            continue

        area = (pbbox[2] - pbbox[0]) * (pbbox[3] - pbbox[1])
        if best is None or area < best_area:
            best = person
            best_area = area

    return best