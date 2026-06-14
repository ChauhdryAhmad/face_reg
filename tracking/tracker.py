import cv2
import numpy as np
from config import FRAME_EDGE_MARGIN, MAX_MISSING_FRAMES, UNCONFIRMED_MAX_MISSING_FRAMES


class FaceTracker:

    def __init__(self, tracker_id, bbox, person_bbox=None):

        self.id = tracker_id
        self.missing = 0
        self.last_bbox = list(map(int, bbox))
        self.last_person_bbox = person_bbox
        self.bbox_velocity = np.zeros(4, dtype=np.float32)
        self.last_update_kind = "face"

        self.identity_id = None
        self.identity_votes = {}
        self.occluded = False
        self.exited_frame = False
        self.continuity_broken = False

        self.kf = cv2.KalmanFilter(8, 4)
        self._init_kf(bbox)

    def _init_kf(self, bbox):

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1

        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)

        initial_state = np.array(
            [cx, cy, w, h, 0, 0, 0, 0],
            dtype=np.float32
        )
        self.kf.statePre = initial_state.copy()
        self.kf.statePost = initial_state.copy()

    def _reset_kf_state(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        state = np.array(
            [
                cx,
                cy,
                w,
                h,
                self.bbox_velocity[0],
                self.bbox_velocity[1],
                self.bbox_velocity[2],
                self.bbox_velocity[3],
            ],
            dtype=np.float32,
        )
        self.kf.statePre = state.copy()
        self.kf.statePost = state.copy()

    def update(self, bbox, person_bbox=None, occluded=False, frame_shape=None):

        bbox = list(map(int, bbox))
        previous_bbox = np.array(self.last_bbox, dtype=np.float32)
        new_bbox = np.array(bbox, dtype=np.float32)
        delta = np.clip(new_bbox - previous_bbox, -80, 80)
        if self.last_update_kind == "person":
            self.bbox_velocity = np.zeros(4, dtype=np.float32)
        else:
            self.bbox_velocity = (self.bbox_velocity * 0.6) + (delta * 0.4)

        self.missing = 0
        self.last_bbox = bbox
        self.last_update_kind = "face"
        self.last_person_bbox = person_bbox
        self.occluded = bool(occluded)
        if frame_shape is not None:
            self.exited_frame = self._outside_frame(person_bbox or bbox, frame_shape)
        if self.occluded or self.exited_frame:
            self.continuity_broken = True

        meas = np.array([
            [(bbox[0] + bbox[2]) / 2],
            [(bbox[1] + bbox[3]) / 2],
            [bbox[2] - bbox[0]],
            [bbox[3] - bbox[1]]
        ], dtype=np.float32)

        self.kf.correct(meas)
        self._reset_kf_state(bbox)

    def update_person_only(self, person_bbox, occluded=False, frame_shape=None):
        person_bbox = list(map(int, person_bbox))

        self.missing = 0
        self.last_bbox = person_bbox
        self.last_update_kind = "person"
        self.last_person_bbox = person_bbox
        self.bbox_velocity = np.zeros(4, dtype=np.float32)
        self.occluded = bool(occluded)
        if frame_shape is not None:
            self.exited_frame = self._outside_frame(person_bbox, frame_shape)
        if self.occluded or self.exited_frame:
            self.continuity_broken = True
        self._reset_kf_state(person_bbox)

    def predicted_bbox(self):
        predicted = np.array(self.last_bbox, dtype=np.float32) + self.bbox_velocity
        x1, y1, x2, y2 = predicted.tolist()

        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1

        return [int(x1), int(y1), int(x2), int(y2)]

    def predict(self):

        self.missing += 1
        self.last_bbox = self.predicted_bbox()
        self.bbox_velocity *= 0.85
        return self.last_bbox

    def alive(self):
        max_missing = MAX_MISSING_FRAMES if self.identity_id is not None else UNCONFIRMED_MAX_MISSING_FRAMES
        return self.missing <= max_missing

    def mark_missed(self, frame_shape):
        predicted = self.predict()
        self.exited_frame = self._outside_frame(
            predicted,
            frame_shape,
        )
        if self.exited_frame:
            self.continuity_broken = True
        return predicted

    def can_predict_without_detection(self):
        return (
            not self.exited_frame
            and self.alive()
        )

    def trusted_continuity(self):
        return (
            self.identity_id is not None
            and self.missing == 0
            and not self.occluded
            and not self.exited_frame
            and not self.continuity_broken
        )

    def clear_continuity_break(self):
        self.continuity_broken = False

    def update_identity(self, identity, score=1.0):

        if identity not in self.identity_votes:
            self.identity_votes[identity] = 0.0

        self.identity_votes[identity] += score

        self.identity_id = max(
            self.identity_votes,
            key=self.identity_votes.get
        )

    @staticmethod
    def _outside_frame(bbox, frame_shape):
        if bbox is None:
            return False

        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        return (
            x2 < -FRAME_EDGE_MARGIN
            or y2 < -FRAME_EDGE_MARGIN
            or x1 > width + FRAME_EDGE_MARGIN
            or y1 > height + FRAME_EDGE_MARGIN
        )










# # tracking/kalman_tracker.py

# import cv2
# import numpy as np
# from config import MAX_MISSING_FRAMES

# class FaceTracker:
#     def __init__(self, face_id, bbox):
#         self.id = face_id
#         self.kf = cv2.KalmanFilter(8, 4)
#         self._init_kf(bbox)
#         self.missing = 0
#         self.last_bbox = bbox

#     def _init_kf(self, bbox):
#         x1, y1, x2, y2 = bbox
#         cx, cy = (x1+x2)/2, (y1+y2)/2
#         w, h = x2-x1, y2-y1

#         self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
#         for i in range(4):
#             self.kf.transitionMatrix[i, i+4] = 1

#         self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
#         self.kf.statePre = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float32)

#     def update(self, bbox):
#         self.missing = 0
#         self.last_bbox = bbox
#         meas = np.array([
#             [(bbox[0]+bbox[2])/2],
#             [(bbox[1]+bbox[3])/2],
#             [bbox[2]-bbox[0]],
#             [bbox[3]-bbox[1]]
#         ], np.float32)
#         self.kf.correct(meas)

#     def predict(self):
#         self.missing += 1
#         p = self.kf.predict()
#         cx, cy, w, h = p[:4].flatten()
#         return [
#             int(cx-w/2), int(cy-h/2),
#             int(cx+w/2), int(cy+h/2)
#         ]

#     def alive(self):
#         return self.missing <= MAX_MISSING_FRAMES
