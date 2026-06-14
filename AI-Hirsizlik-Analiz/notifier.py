import os
import tempfile
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


def send_telegram_alert(
    token: str,
    chat_id: str,
    frame: np.ndarray,
    track_id: int,
    reason: str,
    dwell_time: float,
    video_ts: float,
) -> bool:
    """Şüpheli kişinin en kaliteli karesini Telegram'a gönderir."""
    if not token or not chat_id:
        logger.warning("Telegram bilgileri eksik, gönderim atlandı.")
        return False

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()

    try:
        cv2.imwrite(tmp.name, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        simdi = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        video_sn = f"{int(video_ts // 60):02d}:{int(video_ts % 60):02d}"
        caption = (
            f"SUPHE: Kisi #{track_id}\n"
            f"Sebep: {reason}\n"
            f"Bekleme: {dwell_time:.1f} saniye\n"
            f"Video konum: {video_sn}\n"
            f"Tarih/Saat: {simdi}"
        )

        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(tmp.name, "rb") as photo:
            resp = requests.post(
                url,
                files={"photo": photo},
                data={"chat_id": chat_id, "caption": caption, "parse_mode": ""},
                timeout=20,
            )

        if resp.ok:
            logger.info("Telegram: ID:%s gonderildi.", track_id)
            return True
        else:
            logger.error("Telegram hata: %s", resp.text)
            return False

    except Exception as exc:
        logger.error("Telegram gonderilemedi: %s", exc)
        return False
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def test_connection(token: str, chat_id: str) -> tuple:
    """Telegram bağlantısını test eder. (ok: bool, mesaj: str)"""
    if not token or not chat_id:
        return False, "Token veya Chat ID bos."
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{token}/getMe", timeout=8
        )
        if r.ok:
            name = r.json().get("result", {}).get("username", "?")
            return True, f"Baglanti basarili — Bot: @{name}"
        return False, f"Hata: {r.status_code}"
    except Exception as e:
        return False, f"Baglanti hatasi: {e}"
