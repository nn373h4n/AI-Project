import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

DEFAULT_MODEL          = "yolo11m.pt"
POSE_MODEL             = "yolo11n-pose.pt"
DEFAULT_DWELL_THRESHOLD = 8.0
DEFAULT_MOVE_THRESHOLD  = 60.0
DEFAULT_FRAME_SKIP      = 3
MIN_CONF               = 0.35
