import os



from deepface import DeepFace

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
import numpy as np
import os
import json
from config import RECOGNITION_THRESHOLD


class FaceRecognizer:

    def __init__(self, db="faces_db"):

        # DeepFace model
        self.model_name = "ArcFace"
        self.detector_backend = "skip"

        os.makedirs(db, exist_ok=True)

        self.db = db
        self.emb_file = f"{db}/embeddings.npy"
        self.meta_file = f"{db}/meta.json"

        if os.path.exists(self.emb_file):
            self.embeddings = list(np.load(self.emb_file))
            self.meta = json.load(open(self.meta_file))
        else:
            self.embeddings = []
            self.meta = {}

    def embedding(self, face):

        try:
            result = DeepFace.represent(
                img_path=face,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )

            if len(result) == 0:
                return None

            emb = np.array(result[0]["embedding"])
            return emb

        except Exception as e:
            print("Embedding error:", e)
            return None

    def recognize(self, emb):

        if not self.embeddings:
            return None, 0

        sims = [
            np.dot(emb, e) /
            (np.linalg.norm(emb) * np.linalg.norm(e))
            for e in self.embeddings
        ]

        best = np.argmax(sims)

        if sims[best] > RECOGNITION_THRESHOLD:
            return best, sims[best]

        return None, sims[best]

    def register(self, emb):

        idx = len(self.embeddings)

        self.embeddings.append(emb)
        self.meta[str(idx)] = {"id": idx}

        np.save(self.emb_file, np.array(self.embeddings))
        json.dump(self.meta, open(self.meta_file, "w"))

        return idx