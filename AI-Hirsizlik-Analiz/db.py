import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_BASE   = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE, "data", "analiz.db")
IMG_DIR = os.path.join(_BASE, "data", "detections")


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            video_file      TEXT,
            started_at      TEXT,
            finished_at     TEXT,
            total_frames    INTEGER,
            total_persons   INTEGER,
            total_suspects  INTEGER,
            model_name      TEXT,
            device          TEXT
        );
        CREATE TABLE IF NOT EXISTS detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id  INTEGER REFERENCES analyses(id),
            track_id     INTEGER,
            reason       TEXT,
            dwell_time   REAL,
            video_ts     REAL,
            detected_at  TEXT,
            image_path   TEXT DEFAULT '',
            confirmed    INTEGER DEFAULT -1
        );
        """)


def log_analysis(
    video_file: str,
    started_at: str,
    finished_at: str,
    total_frames: int,
    total_persons: int,
    total_suspects: int,
    model_name: str,
    device: str,
) -> int:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "INSERT INTO analyses "
            "(video_file,started_at,finished_at,total_frames,total_persons,total_suspects,model_name,device) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (video_file, started_at, finished_at, total_frames, total_persons, total_suspects, model_name, device),
        )
        return cur.lastrowid


def log_detection(
    analysis_id: int,
    track_id: int,
    reason: str,
    dwell_time: float,
    video_ts: float,
    image_path: str = "",
) -> int:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "INSERT INTO detections (analysis_id,track_id,reason,dwell_time,video_ts,detected_at,image_path) "
            "VALUES (?,?,?,?,?,?,?)",
            (analysis_id, track_id, reason, dwell_time, video_ts, datetime.now().isoformat(), image_path),
        )
        return cur.lastrowid


def update_label(detection_id: int, confirmed: int) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("UPDATE detections SET confirmed=? WHERE id=?", (confirmed, detection_id))


def get_unlabeled(limit: int = 50) -> List[Dict]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT d.id, d.track_id, d.reason, d.dwell_time, d.image_path, d.detected_at, a.video_file
               FROM detections d JOIN analyses a ON a.id=d.analysis_id
               WHERE d.confirmed=-1 AND d.image_path != ''
               ORDER BY d.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_accuracy_stats() -> Dict:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN confirmed=1 THEN 1 ELSE 0 END) as true_pos,
                SUM(CASE WHEN confirmed=0 THEN 1 ELSE 0 END) as false_pos,
                SUM(CASE WHEN confirmed=-1 THEN 1 ELSE 0 END) as unlabeled
               FROM detections"""
        ).fetchone()
        return dict(row) if row else {"total": 0, "true_pos": 0, "false_pos": 0, "unlabeled": 0}


def get_daily_stats(days: int = 30) -> List[Dict]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT
                strftime('%Y-%m-%d', a.started_at) as day,
                COUNT(DISTINCT a.id) as analyses,
                COALESCE(SUM(a.total_frames),0)   as frames,
                COALESCE(SUM(a.total_persons),0)  as persons,
                COALESCE(SUM(a.total_suspects),0) as suspects
               FROM analyses a
               WHERE date(a.started_at) >= date('now', ?)
               GROUP BY day ORDER BY day""",
            (f"-{days} days",),
        ).fetchall()
        return [dict(r) for r in rows]


def get_detection_trend(days: int = 30) -> List[Dict]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT strftime('%Y-%m-%d', detected_at) as day,
                      COUNT(*) as total,
                      SUM(CASE WHEN confirmed=1 THEN 1 ELSE 0 END) as confirmed_tp
               FROM detections
               WHERE date(detected_at) >= date('now', ?)
               GROUP BY day ORDER BY day""",
            (f"-{days} days",),
        ).fetchall()
        return [dict(r) for r in rows]


def save_detection_image(frame_bgr: np.ndarray, analysis_id: int, track_id: int) -> str:
    fname = f"det_{analysis_id}_{track_id}_{datetime.now().strftime('%H%M%S%f')[:14]}.jpg"
    path = os.path.join(IMG_DIR, fname)
    cv2.imwrite(path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def export_dataset(export_dir: str) -> Tuple[int, int]:
    import shutil
    tp_dir = os.path.join(export_dir, "theft")
    fp_dir = os.path.join(export_dir, "normal")
    os.makedirs(tp_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, image_path, confirmed FROM detections WHERE confirmed != -1 AND image_path != ''"
        ).fetchall()

    tp_count = fp_count = 0
    for row in rows:
        src = row["image_path"]
        if not os.path.exists(src):
            continue
        if row["confirmed"] == 1:
            shutil.copy(src, os.path.join(tp_dir, os.path.basename(src)))
            tp_count += 1
        else:
            shutil.copy(src, os.path.join(fp_dir, os.path.basename(src)))
            fp_count += 1

    return tp_count, fp_count
