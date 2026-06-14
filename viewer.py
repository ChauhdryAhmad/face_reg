import argparse
import os
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
        data = json.loads(bbox_raw)
        if isinstance(data, list) and len(data) == 4:
            return [int(v) for v in data]
    except Exception:
        pass
    return None


def group_by_frame(df):
    grouped = {}
    for _, row in df.iterrows():
        f = int(row["frame"])
        grouped.setdefault(f, []).append(row)
    return grouped


def clean_value(value):
    if pd.isna(value):
        return "?"

    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]

    return text if text else "?"


def row_label(row, prefix=""):
    tracker_id = clean_value(row.get("tracker_id"))
    identity_id = clean_value(row.get("identity_id"))
    status = str(row.get("recognition_status", "") or "").strip()

    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(f"T{tracker_id}")
    parts.append(f"ID{identity_id}")

    if status in {"PERSON_ONLY", "PERSON_ONLY_TRACK", "TRACKER_PREDICTED"}:
        parts.append(status)

    return " ".join(parts)


def draw_label(frame, text, anchor, color):
    x, y = anchor
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(int(x), w - tw - 6))
    y = max(th + baseline + 6, min(int(y), h - 4))

    top_left = (x, y - th - baseline - 6)
    bottom_right = (x + tw + 6, y + baseline)
    cv2.rectangle(frame, top_left, bottom_right, color, -1)
    cv2.putText(
        frame,
        text,
        (x + 3, y - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# -------------------------
# CONDITIONS
# -------------------------

def has_face(rows):
    return any(int(r.get("face_detected", 0)) == 1 for r in rows)


def is_failure(rows):
    return any(
        (str(r.get("status", "")) == "NOT_ACCEPTABLE") or
        (pd.notna(r.get("failure")) and str(r.get("failure")).strip() != "")
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
# FRAME SAFETY
# -------------------------

def ensure_valid_frame(frame, fallback_h=720, fallback_w=1280):
    if frame is None:
        frame = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)

    if not isinstance(frame, np.ndarray):
        frame = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)

    if frame.size == 0:
        frame = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        frame = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)

    # Make it contiguous and writable for OpenCV
    frame = np.ascontiguousarray(frame.copy(), dtype=np.uint8)
    return frame


# -------------------------
# DRAW
# -------------------------

def draw(frame, rows):
    out = ensure_valid_frame(frame)
    h, w = out.shape[:2]

    y_offset = 30
    line_height = 28

    parsed_rows = []
    for r in rows:
        bbox = parse_bbox(r.get("bbox"))
        person_bbox = parse_bbox(r.get("person_bbox"))
        parsed_rows.append((r, bbox, person_bbox))

    for r, bbox, person_bbox in parsed_rows:
        if person_bbox:
            px1, py1, px2, py2 = person_bbox
            px1 = max(0, min(int(px1), w - 1))
            py1 = max(0, min(int(py1), h - 1))
            px2 = max(0, min(int(px2), w - 1))
            py2 = max(0, min(int(py2), h - 1))

            person_color = (180, 120, 0)
            overlay = out.copy()
            cv2.rectangle(overlay, (px1, py1), (px2, py2), person_color, 2)
            out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
            draw_label(out, row_label(r, prefix="P"), (px1, py2 + 22), person_color)

    for r, bbox, person_bbox in parsed_rows:
        if bbox:
            x1, y1, x2, y2 = bbox

            # Clamp bbox to frame size
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))

            color = (0, 255, 0) if str(r.get("status", "")) == "ACCEPTABLE" else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            draw_label(out, row_label(r), (x1, y1), color)

    for r, _, _ in parsed_rows:
        text = (
            f"Tracker:{r.get('tracker_id')} | "
            f"Identity:{r.get('identity_id')} | "
            f"{r.get('recognition_status', '')} | "
            f"Face:{r.get('face_score', '')} "
            f"Person:{r.get('person_score', '')} "
            f"Fused:{r.get('fused_score', '')} "
            f"Cum:{r.get('cumulative_identity_score', '')} | "
            f"{r.get('failure', '')}"
        )

        # Put text on left side so it stays visible on most videos
        cv2.putText(
            out,
            text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 250),
            2,
            cv2.LINE_AA,
        )
        y_offset += line_height

        # Avoid writing beyond frame height
        if y_offset > h - 20:
            break

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

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"CSV file not found: {args.log}")

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")

    df = pd.read_csv(args.log)

    if "frame" not in df.columns:
        raise ValueError("CSV must contain a 'frame' column")

    frames = group_by_frame(df)
    keys = sorted(frames.keys())

    if not keys:
        raise ValueError("No frame data found in CSV")

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if w <= 0 or h <= 0:
        w, h = 1280, 720

    if not fps or fps <= 0:
        fps = 25

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True) if os.path.dirname(args.save) else None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video for writing: {args.save}")

    idx = 0
    playing = False

    def show(i):
        frame_no = keys[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret or frame is None:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Missing frame {frame_no}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        out = draw(frame, frames[frame_no])

        title = f"viewer | frame={frame_no} | idx={i+1}/{len(keys)}"
        cv2.imshow(title, out)

        if writer:
            writer.write(out)

    window_name = "viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def render(i):
        frame_no = keys[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret or frame is None:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Missing frame {frame_no}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        out = draw(frame, frames[frame_no])
        cv2.imshow(window_name, out)

        if writer:
            writer.write(out)

    render(idx)

    while True:
        k = cv2.waitKey(30 if playing else 0) & 0xFF

        if k == 27:
            break

        elif k in [81, ord('a')]:
            idx = max(0, idx - 1)

        elif k in [83, ord('d')]:
            idx = min(len(keys) - 1, idx + 1)

        elif k == ord('f'):
            res = find_next(frames, keys, idx, is_failure)
            if res is not None:
                idx = res

        elif k == ord('F'):
            res = find_prev(frames, keys, idx, is_failure)
            if res is not None:
                idx = res

        elif k == ord('i'):
            res = find_next(frames, keys, idx, is_new_identity)
            if res is not None:
                idx = res

        elif k == ord('I'):
            res = find_prev(frames, keys, idx, is_new_identity)
            if res is not None:
                idx = res

        elif k == ord('r'):
            res = find_next(frames, keys, idx, is_recognition_failure)
            if res is not None:
                idx = res

        elif k == ord('R'):
            res = find_prev(frames, keys, idx, is_recognition_failure)
            if res is not None:
                idx = res

        elif k == ord('n'):
            res = find_next(frames, keys, idx, lambda r: not has_face(r))
            if res is not None:
                idx = res

        elif k == ord('N'):
            res = find_prev(frames, keys, idx, lambda r: not has_face(r))
            if res is not None:
                idx = res

        elif k == ord('x'):
            res = find_next(frames, keys, idx, has_face)
            if res is not None:
                idx = res

        elif k == ord('X'):
            res = find_prev(frames, keys, idx, has_face)
            if res is not None:
                idx = res

        elif k == ord(' '):
            playing = not playing

        if playing:
            idx = min(len(keys) - 1, idx + 1)

        render(idx)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
