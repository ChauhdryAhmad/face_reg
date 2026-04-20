from retinaface import RetinaFace

class RetinaFaceDetector:

    def detect(self, frame):

        detections = RetinaFace.detect_faces(frame)
        results = []

        if not detections:
            print("Not Detected")
            return None

        for _, d in detections.items():

            results.append({
                "bbox": d["facial_area"],
                "conf": d["score"],
                "landmarks": d["landmarks"]
            })

        return results

