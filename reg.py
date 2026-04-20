import cv2
import itertools
import json

from detectors.detector import RetinaFaceDetector
from tracking.tracker import FaceTracker
from recognization.recognizer import FaceRecognizer

from quality.scale import check_scale
from quality.brightness import check_brightness
from quality.landmarks import check_landmarks

from utils.geometry import iou
from utils.snapshot import save_snapshot, save_track_snapshot
from utils.logger import CSVLogger

from config import IOU_MATCH_THRESHOLD


video = cv2.VideoCapture("./inputs/clip2.mp4")

detector = RetinaFaceDetector()
recognizer = FaceRecognizer()

logger = CSVLogger("log.csv")

trackers = {}
tracker_gen = itertools.count()

frame_id = 0

prev_frame = None
prev_bbox = None

c=0

while True:

    ret, frame = video.read()

    if not ret:
        break

    detections = detector.detect(frame)

    matched = set()
    
    if detections is None:
        snapshot = save_snapshot(frame,None,None,True,"NO_FACE_DETECTED")
        logger.log({

            "frame":frame_id,
            "tracker_id":None,

            "identity_id":None,
            "recognized_identity":None,
            "recognition_score":None,
            "recognition_status":None,

            "recognition_correct_by_tracker":None,

            "snapshot_path":snapshot,

            "face_detected":None,
            "tracker_active":None,
            "bbox":None,

            "area_ratio":None,
            "brightness":None,
            "landmark_conf":None,

            "scale_pass":None,
            "brightness_pass":None,
            "landmark_pass":None,

            "status":None,
            "failure":"NO_FACE_DETECTED",
            "tracker_invalid":None
            })

        
    else:

        for d in detections:

            cs = []
            best_id=None
            best_iou=0

            for tid,trk in trackers.items():

                ov=iou(d["bbox"],trk.last_bbox)

                if ov>best_iou:
                    best_id, best_iou = tid,ov

            if best_iou>IOU_MATCH_THRESHOLD:

                trackers[best_id].update(d["bbox"])
                tid=best_id

            else:
                
                for tid,trk in trackers.items():

                    ov=iou(d["bbox"],trk.last_bbox)

                    if ov>best_iou:
                        best_id, best_iou = tid,ov
                    else:
                        # print(f"Frame {frame_id}, Tracker {tid}, IOU: {ov}, Last BBOX: {trk.last_bbox}, BBOX: {d['bbox']}")
                        save_track_snapshot(prev_frame, frame, trk.last_bbox, d["bbox"], c)
                        cs.append(c)
                        c+=1

                print(f"Frame {frame_id}, No matching tracker found, Best IOU: {best_iou}")

                tid=next(tracker_gen)
                trackers[tid]=FaceTracker(tid,d["bbox"])

            trk=trackers[tid]

            matched.add(tid)

            s_ok,ar=check_scale(d["bbox"],frame.shape)
            # print(s_ok, ar)
            b_ok,br=check_brightness(frame,d["bbox"])
            l_ok,lc=check_landmarks(d["landmarks"])

            fails=[]

            if not s_ok: fails.append("SCALE")
            if not b_ok: fails.append("BRIGHTNESS")
            if not l_ok: fails.append("LANDMARK")

            status="ACCEPTABLE" if not fails else "NOT_ACCEPTABLE"
            
            recognition_correct = None

            identity=None
            rec_score=None
            rec_status="NONE"
            snapshot=None
            if status == "ACCEPTABLE":

                x1,y1,x2,y2 = map(int,d["bbox"])
                face = frame[y1:y2,x1:x2]

                emb = recognizer.embedding(face)
                # print(f"Frame {frame_id}, Tracker {tid}, Embedding: {emb}")

                if emb is not None:

                    identity, rec_score = recognizer.recognize(emb)
                    # print(f"Frame {frame_id}, Tracker {tid}, Recognized: {identity}, Score: {rec_score}")

                    

                    if identity is None:

                        identity = recognizer.register(emb)
                        rec_status = "NEW_IDENTITY"
                        recognition_correct = True

                    elif trk.identity_id is None:

                        rec_status = "INITIAL_RECOGNITION"
                        recognition_correct = True
                        
                    elif identity == trk.identity_id:

                        rec_status = "CONFIRMED"
                        recognition_correct = False

                    else:

                        rec_status = "IDENTITY_CONFLICT"
                        recognition_correct = False

                    trk.update_identity(identity)

                    snapshot = save_snapshot(frame,d["bbox"],identity,False,"")

                else:
                    print(f"Frame {frame_id}, Tracker {tid}, Failed to compute embedding")
                    
            else:
                reason = "Scale" if not s_ok else ("Brightness" if not b_ok else "Landmark")
                snapshot = save_snapshot(frame,d["bbox"],identity,True,reason)

            logger.log({

                "frame":frame_id,
                "tracker_id":tid,

                "identity_id":trk.identity_id,
                "recognized_identity":identity,
                "recognition_score":rec_score,
                "recognition_status":rec_status,

                "recognition_correct_by_tracker":recognition_correct,

                "snapshot_path":snapshot,

                "face_detected":1,
                "tracker_active":1,
                "bbox":json.dumps(list(map(int,d["bbox"]))),

                "area_ratio":ar,
                "brightness":br,
                "landmark_conf":lc,

                "scale_pass":s_ok,
                "brightness_pass":b_ok,
                "landmark_pass":l_ok,

                "status":status,
                "failure":",".join(fails),
                "tracker_invalid":",".join(str(c) for c in cs)
                })

            prev_frame, prev_bbox = frame.copy(), d["bbox"]

    frame_id+=1

























# import cv2, itertools
# import json
# from detectors.retinaface_detector import RetinaFaceDetector
# from tracking.kalman_tracker import FaceTracker
# from quality.scale import check_scale
# from quality.brightness import check_brightness
# from quality.landmarks import check_landmarks
# from utils.logger import CSVLogger
# from utils.geometry import iou

# video = cv2.VideoCapture("../inputs/clip2.mp4")
# detector = RetinaFaceDetector()
# logger = CSVLogger("log.csv")

# trackers = {}
# face_id_gen = itertools.count()

# frame_id = 0

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

#     detections = detector.detect(frame)
#     matched = set()

#     for d in detections:
#         best_id, best_iou = None, 0
#         for fid, trk in trackers.items():
#             ov = iou(d["bbox"], trk.last_bbox)
#             if ov > best_iou:
#                 best_id, best_iou = fid, ov

#         if best_iou > 0.3:
#             trackers[best_id].update(d["bbox"])
#             fid = best_id
#         else:
#             fid = next(face_id_gen)
#             trackers[fid] = FaceTracker(fid, d["bbox"])

#         matched.add(fid)

#         s_ok, ar = check_scale(d["bbox"], frame.shape)
#         b_ok, br = check_brightness(frame, d["bbox"])
#         l_ok, lc = check_landmarks(d["landmarks"])

#         fails = []
#         if not s_ok: fails.append("SCALE")
#         if not b_ok: fails.append("BRIGHTNESS")
#         if not l_ok: fails.append("LANDMARK")

#         logger.log({
#             "frame": frame_id,
#             "face_id": fid,
#             "face_detected": 1,
#             "tracker_active": 1,
#             "bbox": json.dumps(list(map(int, d["bbox"]))),
#             "area_ratio": ar,
#             "brightness": br,
#             "landmark_conf": lc,
#             "scale_pass": s_ok,
#             "brightness_pass": b_ok,
#             "landmark_pass": l_ok,
#             "status": "ACCEPTABLE" if not fails else "NOT_ACCEPTABLE",
#             "failure": ",".join(fails),
#             "tracker_gap": 0
#         })

#     for fid, trk in list(trackers.items()):
#         if fid not in matched:
#             if trk.alive():
#                 pb = trk.predict()
#                 logger.log({
#                     "frame": frame_id,
#                     "face_id": fid,
#                     "face_detected": 0,
#                     "tracker_active": 1,
#                     "bbox": json.dumps(list(map(int, pb))),
#                     "area_ratio": None,
#                     "brightness": None,
#                     "landmark_conf": None,
#                     "scale_pass": None,
#                     "brightness_pass": None,
#                     "landmark_pass": None,
#                     "status": "TRACKER_ONLY",
#                     "failure": "",
#                     "tracker_gap": trk.missing
#                 })
#             else:
#                 del trackers[fid]

#     frame_id += 1