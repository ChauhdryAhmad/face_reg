import argparse
import cv2
import itertools
import json

from detectors.detector import RetinaFaceDetector
from detectors.person_detector import PersonDetector
from tracking.tracker import FaceTracker
from recognization.recognizer import FaceRecognizer
from recognization.person_recognizer import PersonRecognizer
from recognization.identity_manager import IdentityManager

from quality.scale import check_scale
from quality.brightness import check_brightness
from quality.landmarks import check_landmarks

from utils.geometry import iou, match_face_to_person, expand_face_to_person_bbox, bbox_center, center_in_bbox
from utils.snapshot import save_snapshot, save_track_snapshot
from utils.logger import CSVLogger

from config import (
    FACE_ONLY_RECOGNITION_THRESHOLD,
    IOU_MATCH_THRESHOLD,
    OCCLUSION_IOU_THRESHOLD,
    PERSON_DUPLICATE_IOU_THRESHOLD,
    PERSON_IOU_MATCH_THRESHOLD,
    PERSON_RECOGNITION_THRESHOLD,
)


def crop_bbox(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return frame[y1:y2, x1:x2]


def detection_is_occluded(face_bbox, person_bbox, detections, person_detections, matched_person):
    for other in detections:
        if other["bbox"] is face_bbox:
            continue
        if iou(face_bbox, other["bbox"]) >= OCCLUSION_IOU_THRESHOLD:
            return True

    if person_bbox is None:
        return False

    for other in person_detections:
        if other is matched_person:
            continue
        if iou(person_bbox, other["bbox"]) >= OCCLUSION_IOU_THRESHOLD:
            return True

    return False


def person_is_occluded(person_bbox, person_detections, matched_idx):
    if person_bbox is None:
        return False

    for idx, other in enumerate(person_detections):
        if idx == matched_idx:
            continue
        if iou(person_bbox, other["bbox"]) >= OCCLUSION_IOU_THRESHOLD:
            return True

    return False


def matched_person_index(face_bbox, person_detections):
    matched_person = match_face_to_person(face_bbox, person_detections)
    if matched_person is None:
        return None

    for idx, person in enumerate(person_detections):
        if person is matched_person:
            return idx

    return None


def person_face_counts(detections, person_detections):
    counts = {}
    if detections is None:
        return counts

    for detection in detections:
        idx = matched_person_index(detection["bbox"], person_detections)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1

    return counts


def duplicate_of_used_person(person_idx, person_detections, used_person_indices):
    person_bbox = person_detections[person_idx]["bbox"]

    for used_idx in used_person_indices:
        used_bbox = person_detections[used_idx]["bbox"]
        if person_boxes_overlap(person_bbox, used_bbox):
            return True
        if center_proximity_score(person_bbox, used_bbox, max_distance=45) >= 0.75:
            return True

    return False


def bbox_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def person_boxes_overlap(a, b):
    small_area = min(bbox_area(a), bbox_area(b))
    covered_small_box = intersection_area(a, b) / small_area if small_area else 0
    return (
        iou(a, b) >= PERSON_DUPLICATE_IOU_THRESHOLD
        or covered_small_box >= 0.18
    )


def duplicate_of_existing_tracker(person_bbox, trackers, ignored_tracker_ids=None):
    ignored_tracker_ids = ignored_tracker_ids or set()

    for tid, trk in trackers.items():
        if tid in ignored_tracker_ids:
            continue

        existing_bbox = trk.last_person_bbox or trk.last_bbox
        if existing_bbox is None:
            continue

        if person_boxes_overlap(person_bbox, existing_bbox):
            return True
        if center_proximity_score(person_bbox, existing_bbox, max_distance=70) >= 0.35:
            return True

    return False


def center_proximity_score(face_bbox, tracker_bbox, max_distance=120):
    face_cx, face_cy = bbox_center(face_bbox)
    trk_cx, trk_cy = bbox_center(tracker_bbox)
    distance = ((face_cx - trk_cx) ** 2 + (face_cy - trk_cy) ** 2) ** 0.5
    return max(0.0, 1.0 - (distance / max_distance))


def duplicate_face_for_used_person(face_bbox, person_idx, person_tracker_ids, trackers):
    if person_idx is None or person_idx not in person_tracker_ids:
        return False

    existing_bbox = trackers[person_tracker_ids[person_idx]].last_bbox
    face_height = max(face_bbox[3] - face_bbox[1], 1)
    existing_height = max(existing_bbox[3] - existing_bbox[1], 1)
    max_distance = max(35, 1.4 * max(face_height, existing_height))

    return (
        iou(face_bbox, existing_bbox) >= 0.10
        or center_proximity_score(face_bbox, existing_bbox, max_distance=max_distance) > 0
    )


def tracker_detection_match_score(detection_bbox, person_bbox, trk, use_person_match):
    face_iou = iou(detection_bbox, trk.last_bbox)
    person_iou = iou(person_bbox, trk.last_person_bbox) if (
        use_person_match and trk.last_person_bbox is not None
    ) else 0

    proximity = center_proximity_score(detection_bbox, trk.last_bbox, max_distance=160)
    continuity_score = 0
    if proximity >= 0.30:
        continuity_score = PERSON_IOU_MATCH_THRESHOLD + (0.30 * proximity)
    if trk.last_person_bbox is not None and center_in_bbox(detection_bbox, trk.last_person_bbox):
        continuity_score += 0.10

    return max(face_iou, person_iou, continuity_score), face_iou


def identity_in_use_by_other_tracker(identity, tid, frame_identity_claims):
    if identity is None:
        return False

    owner_tid = frame_identity_claims.get(identity)
    return owner_tid is not None and owner_tid != tid


def update_trackers_from_person_detections(
    trackers,
    person_detections,
    frame_shape,
    excluded_person_indices=None,
):
    matches = {}
    used_person_indices = set()
    excluded_person_indices = excluded_person_indices or set()

    for tid, trk in list(trackers.items()):
        if trk.last_person_bbox is None:
            continue

        best_idx = None
        best_score = 0
        for idx, person in enumerate(person_detections):
            if idx in used_person_indices or idx in excluded_person_indices:
                continue
            score = iou(person["bbox"], trk.last_person_bbox)
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None or best_score <= PERSON_IOU_MATCH_THRESHOLD:
            continue

        person_bbox = person_detections[best_idx]["bbox"]
        trk.update_person_only(
            person_bbox,
            occluded=person_is_occluded(person_bbox, person_detections, best_idx),
            frame_shape=frame_shape,
        )
        matches[tid] = best_idx
        used_person_indices.add(best_idx)

    return matches


def log_tracker_row(
    logger,
    frame_id,
    trk,
    bbox,
    status,
    failure="",
    face_detected=0,
    person_bbox=None,
    recognition_status="TRACKER_ONLY",
):
    logger.log({
        "frame": frame_id,
        "tracker_id": trk.id,
        "identity_id": trk.identity_id,
        "recognized_identity": trk.identity_id,
        "recognition_score": None,
        "recognition_status": recognition_status,
        "face_score": None,
        "person_score": None,
        "fused_score": None,
        "cumulative_identity_score": trk.identity_votes.get(trk.identity_id) if trk.identity_id is not None else None,
        "person_bbox": json.dumps(list(map(int, person_bbox))) if person_bbox else None,
        "recognition_correct_by_tracker": True,
        "snapshot_path": None,
        "face_detected": face_detected,
        "tracker_active": 1,
        "bbox": json.dumps(list(map(int, bbox))) if bbox else None,
        "area_ratio": None,
        "brightness": None,
        "landmark_conf": None,
        "scale_pass": None,
        "brightness_pass": None,
        "landmark_pass": None,
        "status": status,
        "failure": failure,
        "tracker_invalid": None,
    })


def process_person_detection(
    frame,
    person,
    person_idx,
    person_detections,
    trackers,
    tracker_gen,
    person_recognizer,
    identity_manager,
    logger,
    frame_id,
    frame_identity_claims=None,
):
    person_bbox = person["bbox"]
    occluded = person_is_occluded(person_bbox, person_detections, person_idx)
    if occluded:
        return None

    tid = next(tracker_gen)
    trk = FaceTracker(tid, person_bbox, person_bbox=person_bbox)
    trk.update_person_only(person_bbox, occluded=occluded, frame_shape=frame.shape)
    trackers[tid] = trk

    identity = None
    person_score = None
    fused_score = None
    rec_status = "PERSON_ONLY"
    if frame_identity_claims is None:
        frame_identity_claims = {}

    if not occluded and not trk.exited_frame:
        person_crop = crop_bbox(frame, person_bbox)
        person_emb = person_recognizer.embedding(person_crop)
        if person_emb is not None:
            identity, fused_score, _, person_score, _ = identity_manager.recognize(person_emb=person_emb)
            if (
                identity is None
                or person_score < PERSON_RECOGNITION_THRESHOLD
                or identity_in_use_by_other_tracker(identity, tid, frame_identity_claims)
            ):
                identity = identity_manager.register(person_emb=person_emb)
                rec_status = "PERSON_ONLY_NEW_IDENTITY"
            else:
                identity_manager.update(identity, person_emb=person_emb)
                rec_status = "PERSON_ONLY_RECOGNITION"
            trk.update_identity(identity, score=fused_score or person_score or 1.0)
            frame_identity_claims[identity] = tid

    logger.log({
        "frame": frame_id,
        "tracker_id": tid,
        "identity_id": trk.identity_id,
        "recognized_identity": identity,
        "recognition_score": fused_score,
        "recognition_status": rec_status,
        "face_score": None,
        "person_score": person_score,
        "fused_score": fused_score,
        "cumulative_identity_score": trk.identity_votes.get(trk.identity_id) if trk.identity_id is not None else None,
        "person_bbox": json.dumps(list(map(int, person_bbox))),
        "recognition_correct_by_tracker": True,
        "snapshot_path": None,
        "face_detected": 0,
        "tracker_active": 1,
        "bbox": json.dumps(list(map(int, person_bbox))),
        "area_ratio": None,
        "brightness": None,
        "landmark_conf": None,
        "scale_pass": None,
        "brightness_pass": None,
        "landmark_pass": None,
        "status": "PERSON_ONLY",
        "failure": "",
        "tracker_invalid": None,
    })

    return tid


def process_detection(
    frame,
    detection,
    detections,
    person_detections,
    trackers,
    tracker_gen,
    face_recognizer,
    person_recognizer,
    identity_manager,
    logger,
    frame_id,
    prev_frame,
    prev_bbox,
    invalid_counter,
    face_person_counts=None,
    used_tracker_ids=None,
    frame_identity_claims=None,
):
    cs = []
    best_id = None
    best_iou = 0
    best_match_score = 0

    matched_person = match_face_to_person(detection["bbox"], person_detections)
    matched_person_idx = matched_person_index(detection["bbox"], person_detections)
    if matched_person is not None:
        person_bbox = matched_person["bbox"]
    else:
        person_bbox = expand_face_to_person_bbox(detection["bbox"], frame.shape)

    occluded = detection_is_occluded(
        detection["bbox"],
        person_bbox,
        detections,
        person_detections,
        matched_person,
    )
    use_person_match = (
        matched_person_idx is None
        or face_person_counts is None
        or face_person_counts.get(matched_person_idx, 0) <= 1
    )
    use_person_embedding_for_identity = use_person_match
    used_tracker_ids = used_tracker_ids or set()
    if frame_identity_claims is None:
        frame_identity_claims = {}

    for tid, trk in trackers.items():
        if tid in used_tracker_ids:
            continue

        match_score, face_iou = tracker_detection_match_score(
            detection["bbox"],
            person_bbox,
            trk,
            use_person_match,
        )
        if match_score > best_match_score:
            best_id = tid
            best_iou = face_iou
            best_match_score = match_score

    if best_iou > IOU_MATCH_THRESHOLD or best_match_score > PERSON_IOU_MATCH_THRESHOLD:
        trackers[best_id].update(detection["bbox"], person_bbox=person_bbox, occluded=occluded, frame_shape=frame.shape)
        tid = best_id
    else:
        for tid, trk in trackers.items():
            if tid in used_tracker_ids:
                continue

            match_score, face_iou = tracker_detection_match_score(
                detection["bbox"],
                person_bbox,
                trk,
                use_person_match,
            )
            if match_score > best_match_score:
                best_id = tid
                best_iou = face_iou
                best_match_score = match_score
            else:
                save_track_snapshot(prev_frame, frame, trk.last_bbox, detection["bbox"], invalid_counter[0])
                cs.append(invalid_counter[0])
                invalid_counter[0] += 1

        print(f"Frame {frame_id}, No matching tracker found, Best IOU: {best_iou}, Best match: {best_match_score}")
        tid = next(tracker_gen)
        trackers[tid] = FaceTracker(tid, detection["bbox"], person_bbox=person_bbox)
        trackers[tid].update(detection["bbox"], person_bbox=person_bbox, occluded=occluded, frame_shape=frame.shape)

    trk = trackers[tid]

    s_ok, ar = check_scale(detection["bbox"], frame.shape)
    b_ok, br = check_brightness(frame, detection["bbox"])
    l_ok, lc = check_landmarks(detection["landmarks"])

    fails = []
    if not s_ok:
        fails.append("SCALE")
    if not b_ok:
        fails.append("BRIGHTNESS")
    if not l_ok:
        fails.append("LANDMARK")

    status = "ACCEPTABLE" if not fails else "NOT_ACCEPTABLE"

    recognition_correct = None
    identity = None
    rec_score = None
    face_score = None
    person_score = None
    fused_score = None
    rec_status = "NONE"
    snapshot = None
    update_tracker_identity = False
    if status == "ACCEPTABLE":
        face_crop = crop_bbox(frame, detection["bbox"])
        person_crop = crop_bbox(frame, person_bbox)

        face_emb = face_recognizer.embedding(face_crop)
        person_emb = person_recognizer.embedding(person_crop) if use_person_embedding_for_identity else None

        if face_emb is not None or person_emb is not None:
            identity, fused_score, face_score, person_score, rec_score = identity_manager.recognize(
                face_emb=face_emb,
                person_emb=person_emb,
            )

            if (
                identity is not None
                and identity_in_use_by_other_tracker(identity, tid, frame_identity_claims)
                and identity != trk.identity_id
            ):
                if trk.trusted_continuity():
                    identity = trk.identity_id
                    identity_manager.update(identity, face_emb=face_emb, person_emb=person_emb)
                    rec_status = "TRACKER_CONTINUITY_OVERRIDE"
                    recognition_correct = True
                    update_tracker_identity = True
                elif face_score >= FACE_ONLY_RECOGNITION_THRESHOLD and not use_person_embedding_for_identity:
                    identity = identity_manager.register(face_emb=face_emb)
                    rec_status = "FRAME_ID_CONFLICT_NEW_IDENTITY"
                    recognition_correct = True
                    update_tracker_identity = True
                    trk.clear_continuity_break()
                else:
                    identity = identity_manager.register(face_emb=face_emb, person_emb=person_emb)
                    rec_status = "FRAME_ID_CONFLICT_NEW_IDENTITY"
                    recognition_correct = True
                    update_tracker_identity = True
                    trk.clear_continuity_break()
            elif identity is None and trk.trusted_continuity():
                identity = trk.identity_id
                identity_manager.update(identity, face_emb=face_emb, person_emb=person_emb)
                rec_status = "TRACKER_CONTINUITY_LOW_SCORE"
                recognition_correct = True
                update_tracker_identity = True
            elif identity is None:
                identity = identity_manager.register(face_emb=face_emb, person_emb=person_emb)
                rec_status = "NEW_IDENTITY"
                recognition_correct = True
                update_tracker_identity = True
                trk.clear_continuity_break()
            elif trk.identity_id is None:
                identity_manager.update(identity, face_emb=face_emb, person_emb=person_emb)
                rec_status = "INITIAL_RECOGNITION"
                recognition_correct = True
                update_tracker_identity = True
                trk.clear_continuity_break()
            elif identity == trk.identity_id:
                identity_manager.update(identity, face_emb=face_emb, person_emb=person_emb)
                rec_status = "CONFIRMED"
                recognition_correct = False
                update_tracker_identity = True
                trk.clear_continuity_break()
            elif trk.trusted_continuity():
                identity = trk.identity_id
                identity_manager.update(identity, face_emb=face_emb, person_emb=person_emb)
                rec_status = "TRACKER_CONTINUITY_OVERRIDE"
                recognition_correct = True
                update_tracker_identity = True
            else:
                rec_status = "IDENTITY_CONFLICT"
                recognition_correct = False

            if update_tracker_identity:
                trk.update_identity(identity, score=fused_score or rec_score or 0.0)
                frame_identity_claims[identity] = tid
            snapshot = save_snapshot(frame, detection["bbox"], identity, False, "")
        else:
            print(f"Frame {frame_id}, Tracker {tid}, Failed to compute embeddings")
    else:
        reason = "Scale" if not s_ok else ("Brightness" if not b_ok else "Landmark")
        snapshot = save_snapshot(frame, detection["bbox"], identity, True, reason)

    logger.log({
        "frame": frame_id,
        "tracker_id": tid,
        "identity_id": trk.identity_id,
        "recognized_identity": identity,
        "recognition_score": rec_score,
        "recognition_status": rec_status,
        "face_score": face_score,
        "person_score": person_score,
        "fused_score": fused_score,
        "cumulative_identity_score": trk.identity_votes.get(trk.identity_id) if trk.identity_id is not None else None,
        "person_bbox": json.dumps(list(map(int, person_bbox))) if person_bbox else None,
        "recognition_correct_by_tracker": recognition_correct,
        "snapshot_path": snapshot,
        "face_detected": 1,
        "tracker_active": 1,
        "bbox": json.dumps(list(map(int, detection["bbox"]))),
        "area_ratio": ar,
        "brightness": br,
        "landmark_conf": lc,
        "scale_pass": s_ok,
        "brightness_pass": b_ok,
        "landmark_pass": l_ok,
        "status": status,
        "failure": ",".join(fails),
        "tracker_invalid": ",".join(str(c) for c in cs),
    })

    return tid, frame.copy(), detection["bbox"], matched_person_idx


def main():
    parser = argparse.ArgumentParser(description="Face + person ReID video tracker")
    parser.add_argument("--video", default="./inputs/clip2.mp4")
    parser.add_argument("--output", default="log.csv")
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)

    detector = RetinaFaceDetector()
    person_detector = PersonDetector()
    face_recognizer = FaceRecognizer()
    person_recognizer = PersonRecognizer()
    identity_manager = IdentityManager()

    logger = CSVLogger(args.output)

    trackers = {}
    tracker_gen = itertools.count()
    frame_id = 0
    prev_frame = None
    prev_bbox = None
    invalid_counter = [0]

    while True:
        ret, frame = video.read()
        if not ret:
            break

        detections = detector.detect(frame)
        person_detections = person_detector.detect(frame)

        num_faces = len(detections) if detections is not None else 0
        print(f"For frame num {frame_id}, num of faces {num_faces}, num of persons {len(person_detections)}")

        if detections is None:
            frame_identity_claims = {}
            person_matched_trackers = update_trackers_from_person_detections(
                trackers,
                person_detections,
                frame.shape,
            )

            logged_tracker_ids = set()
            used_person_indices = set(person_matched_trackers.values())

            for tid, person_idx in person_matched_trackers.items():
                trk = trackers[tid]
                person_bbox = person_detections[person_idx]["bbox"]
                if trk.identity_id is not None:
                    frame_identity_claims[trk.identity_id] = tid
                log_tracker_row(
                    logger,
                    frame_id,
                    trk,
                    person_bbox,
                    status="PERSON_ONLY_TRACK",
                    face_detected=0,
                    person_bbox=person_bbox,
                    recognition_status="PERSON_ONLY_TRACK",
                )
                logged_tracker_ids.add(tid)

            for person_idx, person in enumerate(person_detections):
                if person_idx in used_person_indices:
                    continue
                if duplicate_of_used_person(person_idx, person_detections, used_person_indices):
                    continue
                if duplicate_of_existing_tracker(
                    person["bbox"],
                    trackers,
                ):
                    continue
                tid = process_person_detection(
                    frame,
                    person,
                    person_idx,
                    person_detections,
                    trackers,
                    tracker_gen,
                    person_recognizer,
                    identity_manager,
                    logger,
                    frame_id,
                    frame_identity_claims=frame_identity_claims,
                )
                if tid is not None:
                    logged_tracker_ids.add(tid)

            for tid, trk in list(trackers.items()):
                if tid not in person_matched_trackers and tid not in logged_tracker_ids:
                    predicted_bbox = trk.mark_missed(frame.shape)
                    if trk.can_predict_without_detection():
                        log_tracker_row(
                            logger,
                            frame_id,
                            trk,
                            predicted_bbox,
                            status="TRACKER_PREDICTED",
                            failure="DETECTORS_MISSED",
                            face_detected=0,
                            person_bbox=trk.last_person_bbox,
                            recognition_status="TRACKER_PREDICTED",
                        )
                        logged_tracker_ids.add(tid)
                    elif not trk.alive():
                        del trackers[tid]

            if not logged_tracker_ids:
                snapshot = save_snapshot(frame, None, None, True, "NO_FACE_DETECTED")
                logger.log({
                    "frame": frame_id,
                    "tracker_id": None,
                    "identity_id": None,
                    "recognized_identity": None,
                    "recognition_score": None,
                    "recognition_status": None,
                    "face_score": None,
                    "person_score": None,
                    "fused_score": None,
                    "cumulative_identity_score": None,
                    "person_bbox": None,
                    "recognition_correct_by_tracker": None,
                    "snapshot_path": snapshot,
                    "face_detected": 0,
                    "tracker_active": 0,
                    "bbox": None,
                    "area_ratio": None,
                    "brightness": None,
                    "landmark_conf": None,
                    "scale_pass": None,
                    "brightness_pass": None,
                    "landmark_pass": None,
                    "status": None,
                    "failure": "NO_DETECTIONS_NO_TRACKS",
                    "tracker_invalid": None,
                })
        else:
            seen_tracker_ids = set()
            used_person_indices = set()
            person_tracker_ids = {}
            frame_identity_claims = {}
            face_person_counts = person_face_counts(detections, person_detections)

            for detection in detections:
                duplicate_person_idx = matched_person_index(detection["bbox"], person_detections)
                if duplicate_face_for_used_person(
                    detection["bbox"],
                    duplicate_person_idx,
                    person_tracker_ids,
                    trackers,
                ):
                    continue

                tid, prev_frame, prev_bbox, person_idx = process_detection(
                    frame,
                    detection,
                    detections,
                    person_detections,
                    trackers,
                    tracker_gen,
                    face_recognizer,
                    person_recognizer,
                    identity_manager,
                    logger,
                    frame_id,
                    prev_frame,
                    prev_bbox,
                    invalid_counter,
                    face_person_counts=face_person_counts,
                    used_tracker_ids=seen_tracker_ids,
                    frame_identity_claims=frame_identity_claims,
                )
                seen_tracker_ids.add(tid)
                if person_idx is not None:
                    used_person_indices.add(person_idx)
                    person_tracker_ids[person_idx] = tid

            unmatched_trackers = {
                tid: trk for tid, trk in trackers.items() if tid not in seen_tracker_ids
            }
            person_matched_trackers = update_trackers_from_person_detections(
                unmatched_trackers,
                person_detections,
                frame.shape,
                excluded_person_indices=used_person_indices,
            )
            for tid, person_idx in person_matched_trackers.items():
                trk = trackers[tid]
                used_person_indices.add(person_idx)
                if trk.identity_id is not None:
                    frame_identity_claims[trk.identity_id] = tid
                log_tracker_row(
                    logger,
                    frame_id,
                    trk,
                    person_detections[person_idx]["bbox"],
                    status="PERSON_ONLY_TRACK",
                    face_detected=0,
                    person_bbox=person_detections[person_idx]["bbox"],
                    recognition_status="PERSON_ONLY_TRACK",
                )

            for person_idx, person in enumerate(person_detections):
                if person_idx in used_person_indices:
                    continue
                if duplicate_of_used_person(person_idx, person_detections, used_person_indices):
                    continue
                if duplicate_of_existing_tracker(
                    person["bbox"],
                    trackers,
                ):
                    continue
                tid = process_person_detection(
                    frame,
                    person,
                    person_idx,
                    person_detections,
                    trackers,
                    tracker_gen,
                    person_recognizer,
                    identity_manager,
                    logger,
                    frame_id,
                    frame_identity_claims=frame_identity_claims,
                )
                if tid is not None:
                    seen_tracker_ids.add(tid)

            for tid, trk in list(trackers.items()):
                if tid not in seen_tracker_ids and tid not in person_matched_trackers:
                    predicted_bbox = trk.mark_missed(frame.shape)
                    if trk.can_predict_without_detection():
                        log_tracker_row(
                            logger,
                            frame_id,
                            trk,
                            predicted_bbox,
                            status="TRACKER_PREDICTED",
                            failure="DETECTORS_MISSED",
                            face_detected=0,
                            person_bbox=trk.last_person_bbox,
                            recognition_status="TRACKER_PREDICTED",
                        )
                    elif not trk.alive():
                        del trackers[tid]

        frame_id += 1

    video.release()
    print(f"Done. Log saved to {args.output}")


if __name__ == "__main__":
    main()
