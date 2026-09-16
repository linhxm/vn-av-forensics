"""YuNet single-face geometry; visibility/continuity gates are not identity verification."""

import cv2
import numpy as np

from vn_av_data.common.runtime import require_file


class FaceTracker:
    def __init__(self, path, confidence=0.85, min_size=64):
        require_file(path, "YuNet model; run relations-setup --only media")
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), confidence)
        self.min_size = min_size
        self.previous = None

    def inspect(self, frame):
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        count = 0 if faces is None else len(faces)
        if count != 1:
            self.previous = None
            return {"valid": False, "reason": "no_face" if count == 0 else "multiple_faces"}
        face = faces[0]
        x, y, w, h = face[:4].astype(float)
        points = face[4:14].reshape(5, 2)
        mouth = points[3:5]
        valid = (
            min(w, h) >= self.min_size
            and x >= 0
            and y >= 0
            and x + w <= width
            and y + h <= height
            and np.all(mouth[:, 0] > x)
            and np.all(mouth[:, 0] < x + w)
            and np.all(mouth[:, 1] > y + h * 0.4)
            and np.all(mouth[:, 1] < y + h)
            and np.linalg.norm(mouth[0] - mouth[1]) >= 0.12 * w
        )
        box = np.array([x, y, w, h])
        continuous = True
        if self.previous is not None:
            p = self.previous
            distance = np.linalg.norm(box[:2] + box[2:] / 2 - p[:2] - p[2:] / 2)
            continuous = bool(distance < max(w, h) * 0.6 and 0.5 < w / p[2] < 2)
        self.previous = box if valid else None
        return {
            "valid": bool(valid and continuous),
            "reason": "ok" if valid and continuous else "face_geometry_or_track_jump",
            "box": box.tolist(),
        }

    def __call__(self, frames):
        self.previous = None
        crops, valid = [], []
        for frame in frames:
            info = self.inspect(frame)
            valid.append(info["valid"])
            if info["valid"]:
                x, y, w, h = map(round, info["box"])
                crop = frame[y : y + h, x : x + w]
                crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (336, 336))
            else:
                crop = np.zeros((336, 336, 3), np.uint8)
            crops.append(crop)
        return np.stack(crops), np.asarray(valid, bool)
