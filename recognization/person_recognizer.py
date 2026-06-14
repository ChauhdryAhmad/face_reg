import cv2
import numpy as np
import torch
from torchreid.reid.utils.feature_extractor import FeatureExtractor

from config import PERSON_REID_MODEL


class PersonRecognizer:

    def __init__(self, model_name=PERSON_REID_MODEL, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=device,
            verbose=False,
        )

    def embedding(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return None

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        features = self.extractor([rgb])

        if features is None or len(features) == 0:
            return None

        emb = features[0]
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()

        return np.asarray(emb, dtype=np.float32).flatten()
