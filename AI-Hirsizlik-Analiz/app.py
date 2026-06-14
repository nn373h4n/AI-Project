import logging
import os
import time
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from config import (
    DEFAULT_DWELL_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_MOVE_THRESHOLD,
    TELEGRAM_CHAT_ID,
    TELEGRAM_TOKEN,
)
from detector import TheftDetector
from notifier import send_telegram_alert, test_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #080808 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

.gradio-container { max-width: 1160px !important; padding: 20px !important; }

/* --- header --- */
#hdr {
    padding: 16px 0 12px;
    border-bottom: 1px solid #1b1b1b;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
#hdr-title {
    font-size: .95rem;
    font-weight: 500;
    color: #b8b8b8;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin: 0;
}
#hdr-sub {
    font-size: .6rem;
    color: #2e2e2e;
    letter-spacing: 2px;
    margin-top: 5px;
    text-transform: uppercase;
}
.status-row {
    display: flex;
    gap: 16px;
    align-items: center;
}
.s-chip {
    font-size: .58rem;
    color: #2e2e2e;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 5px;
    text-transform: uppercase;
}
.dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #2ecc71;
    box-shadow: 0 0 5px #2ecc71;
    animation: blink 2.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.15} }

/* --- blocks --- */
.block, .gr-form, .form, .panel {
    background: #0c0c0c !important;
    border: 1px solid #181818 !important;
    border-radius: 0 !important;
}

.label-wrap span, .block label span {
    color: #383838 !important;
    font-size: .6rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* --- inputs --- */
input[type=text], input[type=password], textarea, select {
    background: #060606 !important;
    border: 1px solid #1e1e1e !important;
    color: #aaa !important;
    font-size: .78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 0 !important;
    padding: 8px !important;
}
input:focus, textarea:focus {
    border-color: #2a2a2a !important;
    box-shadow: none !important;
    outline: none !important;
}

/* --- buttons --- */
button {
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .68rem !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    transition: background .15s, color .15s !important;
}
button.primary {
    background: transparent !important;
    border: 1px solid #d4a017 !important;
    color: #d4a017 !important;
}
button.primary:hover {
    background: #d4a017 !important;
    color: #000 !important;
}
button.secondary {
    background: transparent !important;
    border: 1px solid #232323 !important;
    color: #383838 !important;
}
button.secondary:hover {
    border-color: #3a3a3a !important;
    color: #666 !important;
}

/* --- sliders --- */
input[type=range] { accent-color: #d4a017 !important; background: transparent !important; border: none !important; }

/* --- accordion --- */
.gr-accordion { border-color: #181818 !important; background: #090909 !important; }
.gr-accordion summary { color: #444 !important; font-size: .65rem !important; letter-spacing: 1.5px !important; }

/* --- log --- */
#log-box textarea {
    color: #3dba6e !important;
    background: #030303 !important;
    font-size: .68rem !important;
    line-height: 1.75 !important;
    border-color: #111 !important;
}

/* --- gallery --- */
.gallery-item { border: 1px solid #7b1212 !important; border-radius: 0 !important; }
.gallery-item img { border-radius: 0 !important; }

/* --- video --- */
video { border: 1px solid #1a1a1a !important; border-radius: 0 !important; }

/* --- progress --- */
.progress-bar { background: #d4a017 !important; border-radius: 0 !important; }

/* --- section titles --- */
.sec-title {
    font-size: .58rem;
    color: #303030;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    border-bottom: 1px solid #141414;
    padding-bottom: 6px;
    margin-bottom: 4px;
}

/* --- alert strip --- */
.alert-strip {
    border: 1px solid #7b1212;
    color: #c0392b;
    font-size: .65rem;
    letter-spacing: 1px;
    padding: 8px 12px;
    margin-top: 6px;
    display: none;
}
"""

# ---------------------------------------------------------------------------
_detector: TheftDetector | None = None


def _get_detector(model_choice: str, dwell: float, move: float) -> TheftDetector:
    model_map = {
        "Hizli  (nano)": "yolov8n.pt",
        "Dengeli (medium)": "yolov8m.pt",
        "Hassas (large)": "yolov8l.pt",
    }
    global _detector
    _detector = TheftDetector(
        model_name=model_map.get(model_choice, DEFAULT_MODEL),
        dwell_threshold=dwell,
        movement_threshold=move,
    )
    return _detector


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def analyze(
    video_file,
    tg_token: str,
    tg_chat: str,
    model_choice: str,
    dwell: float,
    move: float,
    progress=gr.Progress(track_tqdm=False),
):
    logs = []

    if video_file is None:
        return None, [], "Hata: video yuklenmedi."

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logs.append(f"[{_now()}] Analiz basladi  |  Cihaz: {device_str.upper()}")
    logs.append(f"[{_now()}] Model: {model_choice}")

    det = _get_detector(model_choice, dwell, move)

    def prog_cb(p: float):
        progress(p, desc=f"Analiz ediliyor  {p*100:.0f}%")

    try:
        out_video, flagged = det.process_video(video_file, progress_fn=prog_cb)
    except Exception as exc:
        return None, [], f"[HATA] {exc}"

    logs.append(f"[{_now()}] Bitti. {len(flagged)} suphe tespiti.")

    gallery = []
    for item in flagged:
        tid = item["track_id"]
        frame_bgr = item["frame"]
        reason = item["reason"]
        dwell_t = item["dwell_time"]
        vid_ts = det.tracks[tid].last_ts

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        gallery.append((pil_img, f"#{tid} — {reason[:40]}"))

        if tg_token and tg_chat:
            ok = send_telegram_alert(tg_token, tg_chat, frame_bgr, tid, reason, dwell_t, vid_ts)
            durum = "gonderildi" if ok else "gonderilemedi"
            logs.append(f"[{_now()}] Telegram #{tid}: {durum}")
        else:
            logs.append(f"[{_now()}] #{tid} tespit edildi (Telegram ayarsiz)")

    progress(1.0, desc="Tamamlandi")
    return out_video, gallery, "\n".join(logs)


def check_telegram(token: str, chat: str):
    ok, msg = test_connection(token, chat)
    prefix = "OK" if ok else "HATA"
    return f"[{prefix}] {msg}"


# ---------------------------------------------------------------------------
def build_ui():
    device_label = "CUDA" if torch.cuda.is_available() else "CPU"

    with gr.Blocks(
        css=CSS,
        title="Hirsizlik Analiz",
        theme=gr.themes.Base(
            primary_hue="neutral",
            secondary_hue="neutral",
            neutral_hue="neutral",
        ),
    ) as demo:

        # Header
        gr.HTML(f"""
        <div id="hdr">
          <div>
            <div id="hdr-title">Hirsizlik Analiz Sistemi</div>
            <div id="hdr-sub">Computer Vision &nbsp;·&nbsp; GPU Hizlandirilmis &nbsp;·&nbsp; Anlik Tespit</div>
          </div>
          <div class="status-row">
            <div class="s-chip"><span class="dot"></span> Sistem Hazir</div>
            <div class="s-chip">YOLOv8 + ByteTrack</div>
            <div class="s-chip">{device_label}</div>
          </div>
        </div>
        """)

        # --- main row ---
        with gr.Row(equal_height=False):
            # left column: upload + settings
            with gr.Column(scale=2, min_width=320):
                gr.HTML('<div class="sec-title">Video Girdisi</div>')
                video_input = gr.Video(label="Analiz edilecek video", height=220)

                with gr.Accordion("Ayarlar", open=False):
                    with gr.Group():
                        tg_token = gr.Textbox(
                            label="Telegram Bot Token",
                            type="password",
                            value=TELEGRAM_TOKEN,
                            placeholder="123456:ABC-...",
                        )
                        tg_chat = gr.Textbox(
                            label="Telegram Chat ID",
                            value=TELEGRAM_CHAT_ID,
                            placeholder="-100123456789",
                        )
                        tg_test_btn = gr.Button("Baglanti Test Et", variant="secondary", size="sm")
                        tg_status = gr.Textbox(label="Telegram Durumu", interactive=False, lines=1)

                    gr.HTML('<hr style="border-color:#141414;margin:10px 0">')

                    model_radio = gr.Radio(
                        choices=["Hizli  (nano)", "Dengeli (medium)", "Hassas (large)"],
                        value="Dengeli (medium)",
                        label="Model",
                    )
                    dwell_sl = gr.Slider(
                        2, 25, value=DEFAULT_DWELL_THRESHOLD, step=0.5,
                        label="Bekleme esigi (saniye)",
                    )
                    move_sl = gr.Slider(
                        10, 200, value=DEFAULT_MOVE_THRESHOLD, step=5,
                        label="Hareket esigi (piksel std)",
                    )

                with gr.Row():
                    run_btn = gr.Button("ANALİZ BAŞLAT", variant="primary")
                    clr_btn = gr.Button("Temizle", variant="secondary")

            # right column: output video
            with gr.Column(scale=3):
                gr.HTML('<div class="sec-title">Islenmi Video</div>')
                video_output = gr.Video(label="Annotated output", height=380)

        # --- suspects gallery ---
        gr.HTML('<div class="sec-title" style="margin-top:18px">Suphe Tespitleri — En Net Kare</div>')
        gallery = gr.Gallery(
            label="",
            columns=5,
            rows=2,
            height=260,
            object_fit="cover",
            elem_id="suspect-gallery",
        )

        # --- log ---
        gr.HTML('<div class="sec-title" style="margin-top:14px">Sistem Logu</div>')
        log_box = gr.Textbox(
            label="",
            lines=7,
            interactive=False,
            elem_id="log-box",
        )

        # --- wiring ---
        run_btn.click(
            fn=analyze,
            inputs=[video_input, tg_token, tg_chat, model_radio, dwell_sl, move_sl],
            outputs=[video_output, gallery, log_box],
        )

        clr_btn.click(
            fn=lambda: (None, None, [], ""),
            outputs=[video_input, video_output, gallery, log_box],
        )

        tg_test_btn.click(
            fn=check_telegram,
            inputs=[tg_token, tg_chat],
            outputs=[tg_status],
        )

    return demo


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_port=7860,
        server_name="0.0.0.0",
        share=False,
        show_error=True,
    )
