import cv2
import numpy as np
import torch
import tempfile
import os
import logging
from collections import deque
from typing import Optional, Callable, List, Tuple, Dict

from ultralytics import YOLO
from config import MIN_CONF

logger = logging.getLogger(__name__)


class PersonTrack:
    """Tek bir kişinin tüm video boyunca takibini tutar."""

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.positions: deque = deque(maxlen=120)   # son 120 frame konum geçmişi
        self.timestamps: deque = deque(maxlen=120)
        self.first_ts: Optional[float] = None
        self.last_ts: float = 0.0
        self.best_frame: Optional[np.ndarray] = None
        self.best_score: float = -1.0
        self.flagged: bool = False
        self.flag_reason: str = ""

    def update(self, bbox: np.ndarray, frame: np.ndarray, ts: float) -> None:
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.positions.append((cx, cy))
        self.timestamps.append(ts)

        score = self._frame_score(frame, bbox)
        if score > self.best_score:
            self.best_score = score
            self.best_frame = frame.copy()

    @staticmethod
    def _frame_score(frame: np.ndarray, bbox: np.ndarray) -> float:
        """Keskinlik × alan oranı — yüksek = daha net görüntü."""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        area_ratio = ((x2 - x1) * (y2 - y1)) / (frame.shape[0] * frame.shape[1])
        return float(sharpness * (area_ratio ** 0.5))

    @property
    def dwell_time(self) -> float:
        if self.first_ts is None or self.last_ts == 0.0:
            return 0.0
        return self.last_ts - self.first_ts

    @property
    def movement_range(self) -> float:
        """Son position penceresindeki X+Y standart sapması."""
        if len(self.positions) < 4:
            return 999.0
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        return float(np.std(xs) + np.std(ys))

    @property
    def recent_velocity(self) -> float:
        """Son 8 frame için piksel/saniye."""
        if len(self.positions) < 8:
            return 0.0
        p = list(self.positions)[-8:]
        t = list(self.timestamps)[-8:]
        dx = p[-1][0] - p[0][0]
        dy = p[-1][1] - p[0][1]
        dt = max(t[-1] - t[0], 0.001)
        return float(np.sqrt(dx ** 2 + dy ** 2) / dt)


class TheftDetector:
    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        device: str = "auto",
        dwell_threshold: float = 8.0,
        movement_threshold: float = 60.0,
    ):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        logger.info("Cihaz: %s", self.device)
        self.model = YOLO(model_name)
        self.dwell_threshold = dwell_threshold
        self.movement_threshold = movement_threshold
        self.tracks: Dict[int, PersonTrack] = {}
        self.flagged_tracks: List[PersonTrack] = []

    def reset(self) -> None:
        self.tracks = {}
        self.flagged_tracks = []

    # ------------------------------------------------------------------
    def process_video(
        self,
        video_path: str,
        progress_fn: Optional[Callable[[float], None]] = None,
    ) -> Tuple[str, List[Dict]]:
        self.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path = tmp.name
        tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        events = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ts = frame_idx / fps
            annotated, event = self._process_frame(frame, ts)
            writer.write(annotated)
            if event:
                events.append(event)

            frame_idx += 1
            if progress_fn and frame_idx % 8 == 0:
                progress_fn(frame_idx / total)

        cap.release()
        writer.release()

        seen_ids: set = set()
        results = []
        for track in self.flagged_tracks:
            if track.track_id not in seen_ids and track.best_frame is not None:
                seen_ids.add(track.track_id)
                results.append({
                    "track_id": track.track_id,
                    "frame": track.best_frame,
                    "reason": track.flag_reason,
                    "dwell_time": track.dwell_time,
                })

        return out_path, results, frame_idx

    # ------------------------------------------------------------------
    def _process_frame(
        self, frame: np.ndarray, ts: float
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        results = self.model.track(
            frame,
            device=self.device,
            classes=[0],
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        annotated = frame.copy()
        event: Optional[Dict] = None

        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return self._overlay_info(annotated, ts, 0), event

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for box, tid, conf in zip(boxes, ids, confs):
            if conf < MIN_CONF:
                continue

            if tid not in self.tracks:
                self.tracks[tid] = PersonTrack(tid)

            track = self.tracks[tid]
            track.update(box, frame, ts)

            reason = self._check_suspicious(track)
            if reason and not track.flagged:
                track.flagged = True
                track.flag_reason = reason
                self.flagged_tracks.append(track)
                event = {"track_id": int(tid), "ts": ts, "reason": reason}

            x1, y1, x2, y2 = map(int, box)
            color = (30, 30, 210) if track.flagged else (30, 190, 80)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            tag = "SUPHELi!" if track.flagged else "Normal"
            label = f"[{tid}] {tag}  {conf:.0%}"
            ly = max(y1 - 7, 14)
            cv2.putText(annotated, label, (x1, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        return self._overlay_info(annotated, ts, len(ids)), event

    @staticmethod
    def _overlay_info(frame: np.ndarray, ts: float, n_persons: int) -> np.ndarray:
        import time as _t
        ts_str = _t.strftime("%H:%M:%S", _t.gmtime(ts))
        cv2.putText(frame, f"T:{ts_str}  Kisi:{n_persons}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)
        return frame

    def _check_suspicious(self, track: PersonTrack) -> Optional[str]:
        # Uzun süre aynı bölgede bekleme
        if track.dwell_time >= self.dwell_threshold:
            if track.movement_range < self.movement_threshold:
                return f"Uzun sure bekledi ({track.dwell_time:.1f}s, hareket:{track.movement_range:.0f}px)"

        # Bekleme sonrası ani hızlı hareket
        if track.dwell_time >= 5.0 and track.recent_velocity > 260:
            return f"Ani hizli hareket ({track.recent_velocity:.0f}px/s)"

        return None
