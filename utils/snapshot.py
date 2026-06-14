import cv2
import os
import time


def save_snapshot(frame,bbox,identity,failiure,reason):


    if bbox is None:
        face = frame
    else:
        x1,y1,x2,y2 = map(int,bbox)
        face = frame[y1:y2, x1:x2]

    if failiure:
        folder = f"snapshots/failures/{reason}"
    else:
        folder = f"snapshots/person_{identity}"

    os.makedirs(folder,exist_ok=True)

    path = f"{folder}/{identity}_{int(time.time()*1000)}.jpg"

    cv2.imwrite(path,face)

    return path

def save_track_snapshot(prev_frame, frame, prev_bbox, bbox, identity):
    
    if prev_frame is None or frame is None or prev_bbox is None or bbox is None:
        return

    folder = f"snapshots/tracks/{identity}"
    os.makedirs(folder, exist_ok=True)
    
    prev_frame = cv2.rectangle(prev_frame.copy(), (prev_bbox[0], prev_bbox[1]), (prev_bbox[2], prev_bbox[3]), (255, 0, 0), 2)
    frame = cv2.rectangle(frame.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    prev_path = f"{folder}/prev_{identity}_{int(time.time()*1000)}.jpg"
    curr_path = f"{folder}/curr_{identity}_{int(time.time()*1000)}.jpg"

    cv2.imwrite(prev_path, prev_frame)
    cv2.imwrite(curr_path, frame)

    return prev_path, curr_path
