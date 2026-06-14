import json
import os

import numpy as np

from config import (
    FACE_SCORE_WEIGHT,
    FUSED_RECOGNITION_THRESHOLD,
    MAX_EMBEDDINGS_PER_IDENTITY,
    PERSON_SCORE_WEIGHT,
)


class IdentityManager:

    def __init__(self, db="identity_db"):
        os.makedirs(db, exist_ok=True)

        self.db = db
        self.face_file = f"{db}/face_embeddings.npy"
        self.person_file = f"{db}/person_embeddings.npy"
        self.meta_file = f"{db}/meta.json"

        if os.path.exists(self.face_file):
            self.face_embeddings = list(np.load(self.face_file, allow_pickle=True))
        else:
            self.face_embeddings = []

        if os.path.exists(self.person_file):
            self.person_embeddings = list(np.load(self.person_file, allow_pickle=True))
        else:
            self.person_embeddings = []

        if len(self.person_embeddings) < len(self.face_embeddings):
            self.person_embeddings.extend([None] * (len(self.face_embeddings) - len(self.person_embeddings)))
        elif len(self.face_embeddings) < len(self.person_embeddings):
            self.face_embeddings.extend([None] * (len(self.person_embeddings) - len(self.face_embeddings)))

        if os.path.exists(self.meta_file):
            self.meta = json.load(open(self.meta_file))
        else:
            self.meta = {}

        self._migrate_legacy_face_db()

        self.face_embeddings = [self._as_samples(v) for v in self.face_embeddings]
        self.person_embeddings = [self._as_samples(v) for v in self.person_embeddings]

    def _migrate_legacy_face_db(self):
        legacy_dir = "faces_db"
        legacy_emb = f"{legacy_dir}/embeddings.npy"
        legacy_meta = f"{legacy_dir}/meta.json"

        if self.face_embeddings or not os.path.exists(legacy_emb):
            return

        legacy_faces = list(np.load(legacy_emb))
        self.face_embeddings = legacy_faces
        self.person_embeddings = [None] * len(legacy_faces)

        if os.path.exists(legacy_meta):
            self.meta = json.load(open(legacy_meta))

        self._save()

    @staticmethod
    def _cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _as_samples(value):
        if value is None:
            return []

        if isinstance(value, (list, tuple)):
            samples = []
            for sample in value:
                samples.extend(IdentityManager._as_samples(sample))
            return samples

        arr = np.asarray(value)
        if arr.dtype == object:
            try:
                arr = arr.astype(np.float32)
            except (TypeError, ValueError):
                samples = []
                for sample in arr.tolist():
                    samples.extend(IdentityManager._as_samples(sample))
                return samples

        if arr.ndim == 1:
            return [arr.astype(np.float32)]

        if arr.ndim == 2:
            return [sample.astype(np.float32) for sample in arr]

        return [arr.reshape(-1).astype(np.float32)]

    @staticmethod
    def _append_sample(samples, emb):
        if emb is None:
            return samples

        samples = list(samples or [])
        samples.append(np.asarray(emb, dtype=np.float32))
        return samples[-MAX_EMBEDDINGS_PER_IDENTITY:]

    def _best_sample_score(self, emb, samples):
        if emb is None or not samples:
            return 0.0

        return max(self._cosine(emb, sample) for sample in samples)

    def _score_identity(self, idx, face_emb, person_emb):
        parts = []
        weights = []

        if face_emb is not None and self.face_embeddings[idx]:
            parts.append(self._best_sample_score(face_emb, self.face_embeddings[idx]))
            weights.append(FACE_SCORE_WEIGHT)

        if person_emb is not None and self.person_embeddings[idx]:
            parts.append(self._best_sample_score(person_emb, self.person_embeddings[idx]))
            weights.append(PERSON_SCORE_WEIGHT)

        if not parts:
            return 0.0

        total_weight = sum(weights)
        return sum(score * weight for score, weight in zip(parts, weights)) / total_weight

    def recognize(self, face_emb=None, person_emb=None):
        n = max(len(self.face_embeddings), len(self.person_embeddings))
        if n == 0:
            return None, 0.0, 0.0, 0.0, 0.0

        cumulative_scores = []

        for idx in range(n):
            cumulative_scores.append(self._score_identity(idx, face_emb, person_emb))

        best_idx = int(np.argmax(cumulative_scores))
        best_score = cumulative_scores[best_idx]

        face_score = self._best_sample_score(face_emb, self.face_embeddings[best_idx]) if (
            face_emb is not None and best_idx < len(self.face_embeddings)
        ) else 0.0

        person_score = self._best_sample_score(person_emb, self.person_embeddings[best_idx]) if (
            person_emb is not None and best_idx < len(self.person_embeddings)
        ) else 0.0

        if best_score >= FUSED_RECOGNITION_THRESHOLD:
            return best_idx, best_score, face_score, person_score, best_score

        return None, best_score, face_score, person_score, best_score

    def register(self, face_emb=None, person_emb=None):
        idx = max(len(self.face_embeddings), len(self.person_embeddings))

        if idx >= len(self.face_embeddings):
            self.face_embeddings.append(self._append_sample([], face_emb))
        else:
            self.face_embeddings[idx] = self._append_sample([], face_emb)

        if idx >= len(self.person_embeddings):
            self.person_embeddings.append(self._append_sample([], person_emb))
        else:
            self.person_embeddings[idx] = self._append_sample([], person_emb)

        self.meta[str(idx)] = {"id": idx}
        self._save()
        return idx

    def update(self, idx, face_emb=None, person_emb=None):
        if face_emb is not None:
            if idx < len(self.face_embeddings):
                self.face_embeddings[idx] = self._append_sample(self.face_embeddings[idx], face_emb)
            else:
                self.face_embeddings.append(self._append_sample([], face_emb))

        if person_emb is not None:
            if idx < len(self.person_embeddings):
                self.person_embeddings[idx] = self._append_sample(self.person_embeddings[idx], person_emb)
            else:
                self.person_embeddings.append(self._append_sample([], person_emb))

        self._save()

    def _save(self):
        np.save(self.face_file, np.array(self.face_embeddings, dtype=object))
        np.save(self.person_file, np.array(self.person_embeddings, dtype=object))
        json.dump(self.meta, open(self.meta_file, "w"))
