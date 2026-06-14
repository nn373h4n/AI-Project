import cv2
import logging
import os
import subprocess
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import imageio_ffmpeg
import numpy as np
import torch
from ultralytics import YOLO

from config import DEFAULT_FRAME_SKIP, MIN_CONF, POSE_MODEL

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
logger  = logging.getLogger(__name__)


class PersonTrack:
    def __init__(self, track_id: int):
        self.track_id   = track_id
        self.positions:  deque = deque(maxlen=120)
        self.timestamps: deque = deque(maxlen=120)
        self.first_ts:   Optional[float] = None
        self.last_ts:    float = 0.0
        self.best_frame: Optional[np.ndarray] = None
        self.best_score: float = -1.0
        self.flagged:    bool  = False
        self.flag_reason: str  = ""
        # Pose
        self.pose_gesture: Optional[str] = None
        self.pose_suspicious_count: int  = 0

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
        if len(self.positions) < 4:
            return 999.0
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        return float(np.std(xs) + np.std(ys))

    @property
    def recent_velocity(self) -> float:
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
        model_name: str = "yolo11m.pt",
        device: str = "auto",
        dwell_threshold: float = 8.0,
        movement_threshold: float = 60.0,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        use_pose: bool = True,
    ):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        logger.info("Cihaz: %s", self.device)

        self.model = YOLO(model_name)

        self.pose_model: Optional[YOLO] = None
        if use_pose:
            try:
                self.pose_model = YOLO(POSE_MODEL)
                logger.info("Poz modeli yüklendi: %s", POSE_MODEL)
            except Exception as exc:
                logger.warning("Poz modeli yüklenemedi (%s), devam ediliyor.", exc)

        self.dwell_threshold    = dwell_threshold
        self.movement_threshold = movement_threshold
        self.frame_skip         = max(1, frame_skip)
        self.tracks:         Dict[int, PersonTrack] = {}
        self.flagged_tracks: List[PersonTrack]      = []
        self._frame_counter: int                    = 0
        self._cached_results                        = None

    def reset(self) -> None:
        self.tracks          = {}
        self.flagged_tracks  = []
        self._frame_counter  = 0
        self._cached_results = None

    # ──────────────────────────────────────────────────────────────────────────
    def process_video(
        self,
        video_path: str,
        progress_fn: Optional[Callable[[float], None]] = None,
    ) -> Tuple[str, List[Dict], int]:
        self.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        os.makedirs("output", exist_ok=True)
        out_path = os.path.abspath("output/annotated.mp4")

        ffmpeg_cmd = [
            _FFMPEG, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ]
        ffproc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        events:    List[Dict] = []
        frame_idx: int        = 0

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
        results: List[Dict] = []
        for track in self.flagged_tracks:
            if track.track_id not in seen_ids and track.best_frame is not None:
                seen_ids.add(track.track_id)
                results.append({
                    "track_id":   track.track_id,
                    "frame":      track.best_frame,
                    "reason":     track.flag_reason,
                    "dwell_time": track.dwell_time,
                })

        return out_path, results, frame_idx

    # ──────────────────────────────────────────────────────────────────────────
    def _process_frame(
        self, frame: np.ndarray, ts: float
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        self._frame_counter += 1
        should_detect = (self._frame_counter % self.frame_skip == 0) or (self._cached_results is None)

        if should_detect:
            results = self.model.track(
                frame,
                device=self.device,
                classes=[0],
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )
            self._cached_results = results

        annotated = frame.copy()
        event: Optional[Dict] = None

        r = self._cached_results[0]
        if r.boxes is None or r.boxes.id is None:
            return self._overlay_info(annotated, ts, 0), event

        boxes = r.boxes.xyxy.cpu().numpy()
        ids   = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        # Pose estimation on detection frames (every other detection frame)
        if should_detect and self.pose_model and self._frame_counter % (self.frame_skip * 2) == 0:
            self._run_pose(frame, boxes, ids)

        for box, tid, conf in zip(boxes, ids, confs):
            if conf < MIN_CONF:
                continue
            if tid not in self.tracks:
                self.tracks[tid] = PersonTrack(tid)

            track = self.tracks[tid]
            if should_detect:
                track.update(box, frame, ts)

                reason = self._check_suspicious(track)
                if reason and not track.flagged:
                    track.flagged     = True
                    track.flag_reason = reason
                    self.flagged_tracks.append(track)
                    event = {"track_id": int(tid), "ts": ts, "reason": reason}

            self._draw_track(annotated, box, track, conf)

        return self._overlay_info(annotated, ts, len(ids)), event

    # ──────────────────────────────────────────────────────────────────────────
    def _run_pose(self, frame: np.ndarray, boxes: np.ndarray, track_ids: np.ndarray) -> None:
        try:
            pose_res = self.pose_model(frame, verbose=False, device=self.device, classes=[0])
        except Exception:
            return
        if not pose_res or pose_res[0].keypoints is None:
            return

        p0 = pose_res[0]
        if p0.boxes is None or len(p0.boxes.xyxy) == 0:
            return

        pose_boxes = p0.boxes.xyxy.cpu().numpy()
        kpts_all   = p0.keypoints.xy.cpu().numpy()

        for pb, kpts in zip(pose_boxes, kpts_all):
            best_tid  = None
            best_iou  = 0.25
            for box, tid in zip(boxes, track_ids):
                iou = self._bbox_iou(pb, box)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = int(tid)

            if best_tid is None or best_tid not in self.tracks:
                continue

            gesture = self._analyze_pose(kpts)
            if gesture:
                t = self.tracks[best_tid]
                t.pose_gesture = gesture
                t.pose_suspicious_count += 1

    @staticmethod
    def _analyze_pose(kpts: np.ndarray) -> Optional[str]:
        """COCO-17 keypoints → suspicious gesture label or None."""
        if len(kpts) < 13:
            return None
        lw, rw = kpts[9], kpts[10]
        lh, rh = kpts[11], kpts[12]
        ls, rs = kpts[5], kpts[6]

        if (lw[0] == 0 and lw[1] == 0) and (rw[0] == 0 and rw[1] == 0):
            return None

        shoulder_w = max(abs(float(rs[0]) - float(ls[0])), 1.0)

        def dist(a, b):
            return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))

        lw_hip = dist(lw, lh) / shoulder_w
        rw_hip = dist(rw, rh) / shoulder_w

        # Wrists close to hips → concealing / pocket stuffing
        if lw_hip < 0.45 or rw_hip < 0.45:
            return "cep_hareketi"

        # Wrists below hips → reaching into low shelf or bag
        if float(lw[1]) > float(lh[1]) + 25 or float(rw[1]) > float(rh[1]) + 25:
            return "asagi_uzanma"

        return None

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        xb1, yb1, xb2, yb2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        xi1 = max(xa1, xb1); yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2); yi2 = min(ya2, yb2)
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        if inter == 0:
            return 0.0
        union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
        return inter / union if union > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_track(
        img: np.ndarray, box: np.ndarray, track: "PersonTrack", conf: float
    ) -> None:
        x1, y1, x2, y2 = map(int, box)
        tid = track.track_id

        if track.flagged:
            color, lbl_bg, thickness = (0, 0, 220), (0, 0, 170), 3
            label = f" ID:{tid}  SUPHE! "
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 150), -1)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            ay = min(y2 + 20, img.shape[0] - 4)
            cv2.putText(img, "! HIRSIZLIK ALARMI !", (x1, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 2)
        else:
            color, lbl_bg, thickness = (40, 210, 40), (20, 150, 20), 2
            # Show pose gesture if any
            if track.pose_gesture:
                lbl_icon = "🤏" if track.pose_gesture == "cep_hareketi" else "👇"
                label = f" ID:{tid}  IZLENIYOR "
            else:
                label = f" ID:{tid}  IZLENIYOR "

        # Corner bracket box
        cl = max(10, min(18, (x2 - x1) // 4, (y2 - y1) // 4))
        corners = [
            [(x1, y1 + cl), (x1, y1),      (x1 + cl, y1)],
            [(x2 - cl, y1), (x2, y1),      (x2, y1 + cl)],
            [(x1, y2 - cl), (x1, y2),      (x1 + cl, y2)],
            [(x2 - cl, y2), (x2, y2),      (x2, y2 - cl)],
        ]
        for pts in corners:
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, thickness)

        # Pose indicator dot on top-right corner
        if track.pose_gesture and not track.flagged:
            dot_c = (0, 140, 255)  # orange for pose alert
            cv2.circle(img, (x2 - 4, y1 + 4), 4, dot_c, -1)

        # Movement trail (last 25 positions)
        pos_list = list(track.positions)[-25:]
        for i, (cx, cy) in enumerate(pos_list):
            alpha = (i + 1) / len(pos_list)
            r = max(1, int(3 * alpha))
            fade = tuple(int(c * alpha * 0.9) for c in color)
            cv2.circle(img, (int(cx), int(cy)), r, fade, -1)

        # Label background + text
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
        # Dwell + stillness
        if track.dwell_time >= self.dwell_threshold:
            if track.movement_range < self.movement_threshold:
                return (f"Uzun sure bekledi "
                        f"({track.dwell_time:.1f}s, hareket:{track.movement_range:.0f}px)")

        # Dwell + sudden fast movement
        if track.dwell_time >= 5.0 and track.recent_velocity > 260:
            return f"Ani hizli hareket ({track.recent_velocity:.0f}px/s)"

        # Pose: repeated suspicious gesture
        if track.pose_suspicious_count >= 3 and track.dwell_time >= 3.0:
            gesture_label = {
                "cep_hareketi": "cep/gizleme hareketi",
                "asagi_uzanma": "urune uzanma",
            }.get(track.pose_gesture or "", "suphe hareketi")
            return (f"Poz: {gesture_label} "
                    f"({track.pose_suspicious_count}x, {track.dwell_time:.1f}s)")

        return None
