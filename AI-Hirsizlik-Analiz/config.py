import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

DEFAULT_MODEL = "yolov8m.pt"
DEFAULT_DWELL_THRESHOLD = 8.0   # saniye — kişi bu kadar süreden uzun durursa şüpheli
DEFAULT_MOVE_THRESHOLD = 60.0   # piksel std — bu kadardan az hareket ederse şüpheli
MIN_CONF = 0.35                 # minimum tespit güven skoru
