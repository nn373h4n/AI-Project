import cv2
import numpy as np
import torch
import tempfile
import os
import subprocess
import logging
from collections import deque
from typing import Optional, Callable, List, Tuple, Dict

import imageio_ffmpeg
from ultralytics import YOLO
from config import MIN_CONF

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

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

        # H.264 çıktısı — sabit klasöre kaydet (Gradio'nun temp temizliğinden korunsun)
        os.makedirs("output", exist_ok=True)
        out_path = os.path.abspath("output/annotated.mp4")

        ffmpeg_cmd = [
            _FFMPEG, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path,
        ]
        ffproc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        events = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ts = frame_idx / fps
            annotated, event = self._process_frame(frame, ts)
            ffproc.stdin.write(annotated.tobytes())
            if event:
                events.append(event)

            frame_idx += 1
            if progress_fn and frame_idx % 8 == 0:
                progress_fn(frame_idx / total)

        cap.release()
        ffproc.stdin.close()
        ffproc.wait()

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

            self._draw_track(annotated, box, track, conf)

        return self._overlay_info(annotated, ts, len(ids)), event

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_track(img: np.ndarray, box: np.ndarray, track: "PersonTrack", conf: float) -> None:
        x1, y1, x2, y2 = map(int, box)
        tid = track.track_id

        if track.flagged:
            color     = (0, 0, 220)        # kırmızı (BGR)
            lbl_bg    = (0, 0, 170)
            thickness = 3
            label     = f" ID:{tid}  SUPHE! "

            # yarı saydam kırmızı dolgu
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 150), -1)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

            # alt alarm bandı
            ay = min(y2 + 20, img.shape[0] - 4)
            cv2.putText(img, "! HIRSIZLIK ALARMI !", (x1, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 2)
        else:
            color     = (40, 210, 40)      # yeşil (BGR)
            lbl_bg    = (20, 150, 20)
            thickness = 2
            label     = f" ID:{tid}  IZLENIYOR "

        # — köşe bracket kutusu —
        cl = max(10, min(18, (x2 - x1) // 4, (y2 - y1) // 4))
        corners = [
            [(x1, y1 + cl), (x1, y1),  (x1 + cl, y1)],
            [(x2 - cl, y1), (x2, y1),  (x2, y1 + cl)],
            [(x1, y2 - cl), (x1, y2),  (x1 + cl, y2)],
            [(x2 - cl, y2), (x2, y2),  (x2, y2 - cl)],
        ]
        for pts in corners:
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, thickness)

        # — hareket izi (son 25 konum) —
        pos_list = list(track.positions)[-25:]
        for i, (cx, cy) in enumerate(pos_list):
            alpha = (i + 1) / len(pos_list)
            r = max(1, int(3 * alpha))
            fade = tuple(int(c * alpha * 0.9) for c in color)
            cv2.circle(img, (int(cx), int(cy)), r, fade, -1)

        # — etiket (renkli zemin üzerine) —
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, ft = 0.44, 1
        (tw, th), bl = cv2.getTextSize(label, font, fs, ft)
        ly = max(y1 - 4, th + 6)
        cv2.rectangle(img, (x1, ly - th - 6), (x1 + tw + 2, ly + bl), lbl_bg, -1)
        cv2.putText(img, label, (x1 + 2, ly - 2), font, fs, (255, 255, 255), ft)

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
