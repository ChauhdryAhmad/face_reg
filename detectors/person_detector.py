from ultralytics import YOLO

from config import PERSON_DETECTOR_CONF, PERSON_DETECTOR_MODEL


class PersonDetector:

    def __init__(self, model_name=PERSON_DETECTOR_MODEL, conf=PERSON_DETECTOR_CONF):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf)
        detections = []

        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if names.get(cls_id) != "person":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(box.conf.item()),
                })

        return detections
