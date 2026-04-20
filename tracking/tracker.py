import cv2
import numpy as np
from config import MAX_MISSING_FRAMES


class FaceTracker:

    def __init__(self, tracker_id, bbox):

        self.id = tracker_id
        self.missing = 0
        self.last_bbox = bbox

        self.identity_id = None
        self.identity_votes = {}

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

        self.kf.statePre = np.array(
            [cx, cy, w, h, 0, 0, 0, 0],
            dtype=np.float32
        )

    def update(self, bbox):

        self.missing = 0
        self.last_bbox = bbox

        meas = np.array([
            [(bbox[0] + bbox[2]) / 2],
            [(bbox[1] + bbox[3]) / 2],
            [bbox[2] - bbox[0]],
            [bbox[3] - bbox[1]]
        ], dtype=np.float32)

        self.kf.correct(meas)

    def predict(self):

        self.missing += 1
        p = self.kf.predict()

        cx, cy, w, h = p[:4].flatten()

        return [
            int(cx - w / 2),
            int(cy - h / 2),
            int(cx + w / 2),
            int(cy + h / 2)
        ]

    def alive(self):
        return self.missing <= MAX_MISSING_FRAMES

    def update_identity(self, identity):

        if identity not in self.identity_votes:
            self.identity_votes[identity] = 0

        self.identity_votes[identity] += 1

        self.identity_id = max(
            self.identity_votes,
            key=self.identity_votes.get
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