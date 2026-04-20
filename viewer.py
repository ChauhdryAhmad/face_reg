import argparse
import cv2
import pandas as pd
import numpy as np
import json


# -------------------------
# HELPERS
# -------------------------

def parse_bbox(bbox_raw):
    if pd.isna(bbox_raw):
        return None
    try:
        return list(map(int, json.loads(bbox_raw)))
    except:
        return None


def group_by_frame(df):
    grouped = {}
    for _, row in df.iterrows():
        f = int(row["frame"])
        if f not in grouped:
            grouped[f] = []
        grouped[f].append(row)
    return grouped


# -------------------------
# CONDITIONS (MATCH YOUR CSV)
# -------------------------

def has_face(rows):
    return any(r["face_detected"] == 1 for r in rows if "face_detected" in r)


def is_failure(rows):
    return any(
        (str(r.get("status", "")) == "NOT_ACCEPTABLE") or
        (pd.notna(r.get("failure")) and str(r.get("failure")) != "")
        for r in rows
    )


def is_new_identity(rows):
    return any(
        str(r.get("recognition_status", "")).upper() == "NEW_IDENTITY"
        for r in rows
    )


def is_recognition_failure(rows):
    return any(
        str(r.get("recognition_status", "")).upper() in ["UNKNOWN", "FAIL", "NO_MATCH"]
        for r in rows
    )


# -------------------------
# SEARCH
# -------------------------

def find_next(frames, keys, idx, cond):
    for i in range(idx + 1, len(keys)):
        if cond(frames[keys[i]]):
            return i
    return None


def find_prev(frames, keys, idx, cond):
    for i in range(idx - 1, -1, -1):
        if cond(frames[keys[i]]):
            return i
    return None


# -------------------------
# DRAW
# -------------------------

def draw(frame, rows):
    out = frame.copy()

    y_offset = 20

    for r in rows:
        bbox = parse_bbox(r.get("bbox"))

        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if r.get("status") == "ACCEPTABLE" else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text = f"ID:{r.get('tracker_id')} | IDen:{r.get('identity_id')} | {r.get('recognition_status')} | {r.get('failure')}"
        cv2.putText(out, text, (520, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,250), 3)
        y_offset += 18

    return out


# -------------------------
# MAIN
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--video", required=True)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.log)

    frames = group_by_frame(df)
    keys = sorted(frames.keys())

    cap = cv2.VideoCapture(args.video)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    idx = 0
    playing = False

    def show(i):
        frame_no = keys[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        out = draw(frame, frames[frame_no])
        cv2.imshow("viewer", out)

        if writer:
            writer.write(out)

    show(idx)

    while True:
        k = cv2.waitKey(30 if playing else 0) & 0xFF

        if k == 27:
            break

        # ← →
        elif k in [81, ord('a')]:
            idx = max(0, idx - 1)

        elif k in [83, ord('d')]:
            idx = min(len(keys) - 1, idx + 1)

        # failure
        elif k == ord('f'):
            res = find_next(frames, keys, idx, is_failure)
            if res is not None: idx = res

        elif k == ord('F'):
            res = find_prev(frames, keys, idx, is_failure)
            if res is not None: idx = res

        # new identity
        elif k == ord('i'):
            res = find_next(frames, keys, idx, is_new_identity)
            if res is not None: idx = res

        elif k == ord('I'):
            res = find_prev(frames, keys, idx, is_new_identity)
            if res is not None: idx = res

        # recognition failure
        elif k == ord('r'):
            res = find_next(frames, keys, idx, is_recognition_failure)
            if res is not None: idx = res

        elif k == ord('R'):
            res = find_prev(frames, keys, idx, is_recognition_failure)
            if res is not None: idx = res

        # no face
        elif k == ord('n'):
            res = find_next(frames, keys, idx, lambda r: not has_face(r))
            if res is not None: idx = res

        elif k == ord('N'):
            res = find_prev(frames, keys, idx, lambda r: not has_face(r))
            if res is not None: idx = res

        # detected face
        elif k == ord('x'):  # changed from d to avoid conflict
            res = find_next(frames, keys, idx, has_face)
            if res is not None: idx = res

        elif k == ord('X'):
            res = find_prev(frames, keys, idx, has_face)
            if res is not None: idx = res

        # play/pause
        elif k == ord(' '):
            playing = not playing

        if playing:
            idx = min(len(keys) - 1, idx + 1)

        show(idx)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()