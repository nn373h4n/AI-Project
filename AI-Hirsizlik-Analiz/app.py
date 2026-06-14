import logging
import os
import warnings
import torch
from datetime import datetime

import cv2
import gradio as gr
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="gradio")

from config import (
    DEFAULT_DWELL_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_MOVE_THRESHOLD,
    TELEGRAM_CHAT_ID,
    TELEGRAM_TOKEN,
)
from detector import TheftDetector
from notifier import send_telegram_alert, test_connection

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Industrial Surveillance aesthetic
# Fonts: Share Tech Mono (display/data) · IBM Plex Mono (body/labels)
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
:root {
  --bg:        #04060f;
  --s1:        #070a18;
  --s2:        #0b1024;
  --bd:        #111a2e;
  --bd2:       #1c2d4a;
  --acc:       #f5a520;
  --acc-g:     rgba(245,165,32,.18);
  --red:       #c0392b;
  --red-g:     rgba(192,57,43,.18);
  --grn:       #00c17a;
  --grn-g:     rgba(0,193,122,.15);
  --data:      #00e5a0;
  --txt:       #6a7c96;
  --txt-hi:    #b0c2da;
  --txt-lo:    #1e2d45;
  --f-d:       'Share Tech Mono', monospace;
  --f-b:       'IBM Plex Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }
body { background: var(--bg) !important; margin: 0; }

/* dot-grid + radial ambient */
.gradio-container {
  background:
    radial-gradient(ellipse 70% 50% at 50% -10%, rgba(18,35,80,.35) 0%, transparent 65%),
    radial-gradient(ellipse 30% 20% at 85% 90%, rgba(245,165,32,.04) 0%, transparent 60%),
    radial-gradient(circle 1px at 50% 50%, #111a2e 1px, transparent 0),
    var(--bg) !important;
  background-size: auto, auto, 36px 36px !important;
  max-width: 1240px !important;
  padding: 28px 32px !important;
  font-family: var(--f-b) !important;
}

/* ── HEADER ─────────────────────────────── */
#hdr {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: flex-end;
  padding: 0 0 20px;
  margin-bottom: 36px;
  border-bottom: 1px solid var(--bd);
  position: relative;
  overflow: hidden;
}
#hdr::before {
  content: '';
  position: absolute;
  top: -40px; left: -32px;
  width: 320px; height: 120px;
  background: radial-gradient(ellipse, rgba(245,165,32,.06) 0%, transparent 70%);
  pointer-events: none;
}
#hdr::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0;
  width: 90px; height: 2px;
  background: linear-gradient(90deg, var(--acc), transparent);
}
.eyebrow {
  font-family: var(--f-b);
  font-size: .5rem;
  color: var(--txt-lo);
  letter-spacing: 5px;
  text-transform: uppercase;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.eyebrow::before {
  content: '';
  display: inline-block;
  width: 18px; height: 1px;
  background: var(--acc);
  box-shadow: 0 0 8px var(--acc);
}
#hdr-title {
  font-family: var(--f-d);
  font-size: 2rem;
  color: var(--txt-hi);
  letter-spacing: 4px;
  margin: 0;
  line-height: 1;
  text-transform: uppercase;
}
#hdr-title span {
  color: var(--acc);
  text-shadow: 0 0 32px var(--acc-g), 0 0 60px rgba(245,165,32,.08);
}
#hdr-right {
  text-align: right;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 10px;
}
#live-clock {
  font-family: var(--f-d);
  font-size: 1.15rem;
  color: var(--txt);
  letter-spacing: 3px;
  min-width: 200px;
}
.chips {
  display: flex; gap: 6px;
}
.chip {
  font-family: var(--f-b);
  font-size: .46rem;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  border: 1px solid var(--bd);
  color: var(--txt-lo);
  padding: 3px 10px;
  display: flex; align-items: center; gap: 5px;
  transition: border-color .2s, color .2s;
}
.chip.live {
  border-color: var(--grn);
  color: var(--grn);
}
.chip.live:hover {
  background: var(--grn-g);
}
.ldot {
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--grn);
  box-shadow: 0 0 7px var(--grn);
  animation: lb 1.9s ease-in-out infinite;
  flex-shrink: 0;
}
@keyframes lb { 0%,100%{opacity:1} 50%{opacity:.08} }

/* ── SECTION HEADER ─────────────────────── */
.sec {
  font-family: var(--f-b);
  font-size: .48rem;
  color: var(--txt-lo);
  letter-spacing: 3.5px;
  text-transform: uppercase;
  padding-bottom: 10px;
  margin-bottom: 14px;
  border-bottom: 1px solid var(--bd);
  display: flex; align-items: center; gap: 10px;
}
.sec-mark {
  font-family: var(--f-d);
  font-size: .75rem;
  color: var(--acc);
  line-height: 1;
}

/* ── GRADIO BLOCKS ───────────────────────── */
.block, .gr-form, .form, .panel, .gap {
  background: var(--s1) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 0 !important;
}

/* top-right accent line */
.block {
  border-top-color: var(--bd2) !important;
}

/* ── LABELS ──────────────────────────────── */
.label-wrap span, .block label > span {
  font-family: var(--f-b) !important;
  color: var(--txt-lo) !important;
  font-size: .52rem !important;
  text-transform: uppercase !important;
  letter-spacing: 2.5px !important;
}

/* ── INPUTS ──────────────────────────────── */
input[type=text], input[type=password], textarea {
  background: rgba(4,6,15,.95) !important;
  border: 1px solid var(--bd) !important;
  color: var(--txt) !important;
  font-family: var(--f-b) !important;
  font-size: .76rem !important;
  border-radius: 0 !important;
  padding: 9px 13px !important;
  transition: border-color .2s, color .2s !important;
}
input[type=text]:focus, input[type=password]:focus, textarea:focus {
  border-color: var(--bd2) !important;
  color: var(--txt-hi) !important;
  box-shadow: inset 0 0 0 1px rgba(245,165,32,.06) !important;
  outline: none !important;
}

/* ── BUTTONS ─────────────────────────────── */
button {
  font-family: var(--f-b) !important;
  font-size: .58rem !important;
  letter-spacing: 3.5px !important;
  text-transform: uppercase !important;
  border-radius: 0 !important;
  cursor: pointer !important;
  transition: all .2s !important;
}
button.primary {
  background: transparent !important;
  border: 1px solid var(--acc) !important;
  color: var(--acc) !important;
  position: relative !important;
  overflow: hidden !important;
}
button.primary:hover {
  background: var(--acc) !important;
  color: #000 !important;
  box-shadow: 0 0 28px var(--acc-g) !important;
}
button.secondary {
  background: transparent !important;
  border: 1px solid var(--bd) !important;
  color: var(--txt-lo) !important;
}
button.secondary:hover {
  border-color: var(--bd2) !important;
  color: var(--txt) !important;
}

/* ── SLIDERS ─────────────────────────────── */
input[type=range] {
  accent-color: var(--acc) !important;
  background: transparent !important;
  border: none !important;
}

/* ── ACCORDION ───────────────────────────── */
.accordion {
  background: var(--s2) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 0 !important;
}
.accordion > button {
  font-size: .54rem !important;
  letter-spacing: 2px !important;
  color: var(--txt-lo) !important;
}

/* ── VIDEO ───────────────────────────────── */
video {
  border: 1px solid var(--bd) !important;
  border-radius: 0 !important;
  display: block !important;
}
/* scanline overlay on video wrappers */
#vid-in .block::after, #vid-out .block::after {
  content: '' !important;
  position: absolute !important;
  inset: 0 !important;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 3px,
    rgba(0,0,0,.05) 3px,
    rgba(0,0,0,.05) 4px
  ) !important;
  pointer-events: none !important;
  z-index: 5 !important;
}

/* corner brackets on video areas */
#vid-in, #vid-out {
  position: relative;
}
#vid-in::before, #vid-in::after,
#vid-out::before, #vid-out::after {
  content: '';
  position: absolute;
  width: 18px; height: 18px;
  border-color: rgba(245,165,32,.5);
  border-style: solid;
  z-index: 10;
  pointer-events: none;
}
#vid-in::before,  #vid-out::before  { top:6px;  left:6px;  border-width:1px 0 0 1px; }
#vid-in::after,   #vid-out::after   { bottom:6px; right:6px; border-width:0 1px 1px 0; }

/* ── GALLERY ─────────────────────────────── */
.gallery-item {
  border: 1px solid rgba(192,57,43,.6) !important;
  border-radius: 0 !important;
  overflow: hidden !important;
  position: relative !important;
}
.gallery-item img {
  filter: contrast(1.08) saturate(.8) !important;
  border-radius: 0 !important;
  transition: filter .3s !important;
}
.gallery-item:hover img { filter: contrast(1.12) saturate(.95) !important; }

/* ── LOG TERMINAL ────────────────────────── */
#log-box textarea {
  font-family: var(--f-d) !important;
  font-size: .68rem !important;
  color: var(--data) !important;
  background: #020409 !important;
  border-color: var(--bd) !important;
  line-height: 1.9 !important;
  letter-spacing: .5px !important;
  text-shadow: 0 0 12px rgba(0,229,160,.15) !important;
}

/* ── PROGRESS ────────────────────────────── */
.progress-bar, [class*=progress-level] {
  background: linear-gradient(90deg, var(--acc), rgba(245,165,32,.6)) !important;
  box-shadow: 0 0 14px var(--acc-g) !important;
  border-radius: 0 !important;
}

/* ── STATS GRID ──────────────────────────── */
#stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--bd);
  border: 1px solid var(--bd);
  margin: 24px 0;
}
.sc {
  background: var(--s1);
  padding: 16px 20px;
  display: flex; flex-direction: column; gap: 6px;
  position: relative;
  overflow: hidden;
}
.sc::after {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 2px; height: 100%;
  background: var(--bd2);
}
.sc:first-child::after { background: var(--acc); box-shadow: 0 0 10px var(--acc-g); }
.sc:nth-child(3)::after { background: var(--red); box-shadow: 0 0 10px var(--red-g); }
.sc-lbl {
  font-family: var(--f-b);
  font-size: .47rem;
  color: var(--txt-lo);
  letter-spacing: 2.5px;
  text-transform: uppercase;
}
.sc-val {
  font-family: var(--f-d);
  font-size: 1.5rem;
  color: var(--txt-hi);
  line-height: 1;
  letter-spacing: 1px;
}
.sc-val.d { color: var(--red);  text-shadow: 0 0 20px var(--red-g); }
.sc-val.g { color: var(--grn);  text-shadow: 0 0 20px var(--grn-g); }
.sc-val.a { color: var(--acc);  text-shadow: 0 0 20px var(--acc-g); }
.sc-sub {
  font-family: var(--f-b);
  font-size: .45rem;
  color: var(--txt-lo);
  letter-spacing: 1px;
  margin-top: 2px;
}

/* ── DIVIDER ─────────────────────────────── */
.div {
  border: none;
  border-top: 1px solid var(--bd);
  margin: 22px 0;
}

/* ── SELECT ──────────────────────────────── */
select {
  background: var(--bg) !important;
  border: 1px solid var(--bd) !important;
  color: var(--txt) !important;
  font-family: var(--f-b) !important;
  border-radius: 0 !important;
}

/* ── TELEGRAM PANEL ──────────────────────── */
#tg-panel {
  background: var(--s2);
  border: 1px solid var(--bd2);
  padding: 16px 18px;
  margin-top: 14px;
  position: relative;
  overflow: hidden;
}
#tg-panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 2px;
  background: linear-gradient(90deg, #2196F3, transparent);
}
.tg-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}
.tg-icon {
  font-size: 1.1rem;
  line-height: 1;
}
.tg-title {
  font-family: var(--f-b);
  font-size: .54rem;
  color: #5baef7;
  letter-spacing: 2.5px;
  text-transform: uppercase;
}
.tg-status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #555;
  margin-left: auto;
  transition: background .3s, box-shadow .3s;
}
.tg-status-dot.ok  { background: var(--grn); box-shadow: 0 0 8px var(--grn-g); }
.tg-status-dot.err { background: var(--red); box-shadow: 0 0 8px var(--red-g); }
"""

# ─────────────────────────────────────────────────────────────────────────────
FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono'
    '&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">'
)

HEADER = """
<div id="hdr">
  <div>
    <div class="eyebrow">Guvenlik Analiz Sistemi · CAM/AI/GPU</div>
    <h1 id="hdr-title">HIRSIZLIk <span>ANALİZ</span></h1>
  </div>
  <div id="hdr-right">
    <div id="live-clock">--:--:--</div>
    <div class="chips">
      <div class="chip live"><span class="ldot"></span>CANLI</div>
      <div class="chip">GPU</div>
      <div class="chip">YOLOv8</div>
      <div class="chip">ByteTrack</div>
    </div>
  </div>
</div>
<script>
(function(){
  function tick(){
    var el=document.getElementById('live-clock');
    if(!el)return;
    var d=new Date();
    var t=[d.getHours(),d.getMinutes(),d.getSeconds()]
           .map(function(n){return String(n).padStart(2,'0')}).join(':');
    var dt=d.toLocaleDateString('tr-TR',{day:'2-digit',month:'2-digit',year:'numeric'});
    el.textContent=t+' · '+dt;
  }
  tick(); setInterval(tick,1000);
})();
</script>
"""

STATS_EMPTY = """
<div id="stats-row">
  <div class="sc">
    <div class="sc-lbl">Analiz Edilen Kare</div>
    <div class="sc-val a">—</div>
    <div class="sc-sub">Bekliyor</div>
  </div>
  <div class="sc">
    <div class="sc-lbl">Tespit Edilen Kisi</div>
    <div class="sc-val">—</div>
    <div class="sc-sub">Bekliyor</div>
  </div>
  <div class="sc">
    <div class="sc-lbl">Suphe Tespiti</div>
    <div class="sc-val d">—</div>
    <div class="sc-sub">Bekliyor</div>
  </div>
  <div class="sc">
    <div class="sc-lbl">Islem Cihazi</div>
    <div class="sc-val g">—</div>
    <div class="sc-sub">Hazir</div>
  </div>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_det: TheftDetector | None = None


def _make(model_choice: str, dwell: float, move: float) -> TheftDetector:
    global _det
    m = {"Hizli (nano)": "yolov8n.pt",
         "Dengeli (medium)": "yolov8m.pt",
         "Hassas (large)": "yolov8l.pt"}
    _det = TheftDetector(
        model_name=m.get(model_choice, DEFAULT_MODEL),
        dwell_threshold=dwell,
        movement_threshold=move,
    )
    return _det


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def analyze(
    video_file,
    tg_token: str,
    tg_chat: str,
    model_choice: str,
    dwell: float,
    move: float,
    progress=gr.Progress(),
):
    logs = []

    if video_file is None:
        return _video_html(None), [], "> HATA: video dosyasi secilmedi.", STATS_EMPTY

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logs.append(f"> [{_ts()}] Sistem basladi  |  Cihaz:{device}  Model:{model_choice}")
    logs.append(f"> [{_ts()}] Girdi: {os.path.basename(str(video_file))}")

    det = _make(model_choice, dwell, move)

    def prog(p: float):
        progress(p, desc=f"Islem  {p*100:.0f}%")

    try:
        out_video, flagged, frame_count = det.process_video(video_file, progress_fn=prog)
    except Exception as exc:
        return _video_html(None), [], f"> HATA: {exc}", STATS_EMPTY

    total = len(det.tracks)
    logs.append(f"> [{_ts()}] Tamamlandi  kare:{frame_count}  kisi:{total}  suphe:{len(flagged)}")

    gallery = []
    for item in flagged:
        tid = item["track_id"]
        bgr = item["frame"]
        reason = item["reason"]
        dwell_t = item["dwell_time"]
        vid_ts = det.tracks[tid].last_ts

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gallery.append((Image.fromarray(rgb), f"#{tid}  {reason[:48]}"))

        if tg_token and tg_chat:
            ok = send_telegram_alert(tg_token, tg_chat, bgr, tid, reason, dwell_t, vid_ts)
            logs.append(f"> [{_ts()}] Telegram #{tid}: {'GONDERILDI' if ok else 'HATA'}")
        else:
            logs.append(f"> [{_ts()}] #{tid} tespit — Telegram yapilandirilmamis")

    if not flagged:
        logs.append(f"> [{_ts()}] Supe davranis tespit edilmedi.")

    danger_cls = "d" if flagged else "g"
    stats = f"""
<div id="stats-row">
  <div class="sc">
    <div class="sc-lbl">Analiz Edilen Kare</div>
    <div class="sc-val a">{frame_count:,}</div>
    <div class="sc-sub">@25–60 fps</div>
  </div>
  <div class="sc">
    <div class="sc-lbl">Tespit Edilen Kisi</div>
    <div class="sc-val">{total}</div>
    <div class="sc-sub">Benzersiz kimlik</div>
  </div>
  <div class="sc">
    <div class="sc-lbl">Suphe Tespiti</div>
    <div class="sc-val {danger_cls}">{len(flagged)}</div>
    <div class="sc-sub">{"Bildirim gonderildi" if flagged and tg_token else "Kayit alindi"}</div>
  </div>
  <div class="sc">
    <div class="sc-lbl">Islem Cihazi</div>
    <div class="sc-val g">{device}</div>
    <div class="sc-sub">YOLOv8 · ByteTrack</div>
  </div>
</div>
"""
    progress(1.0, desc="Tamamlandi")
    return _video_html(out_video), gallery, "\n".join(logs), stats


_OUTPUT_DIR = os.path.abspath("output")


def _video_html(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return (
            '<div style="height:420px;display:flex;align-items:center;'
            'justify-content:center;color:#243040;font-family:monospace;'
            'font-size:.7rem;letter-spacing:2px;background:#070b15;'
            'border:1px solid #111928">ANALİZ BEKLENİYOR</div>'
        )
    import base64
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    size_mb = round(os.path.getsize(path) / 1024 / 1024, 1)
    return (
        f'<video controls autoplay muted '
        f'style="width:100%;max-height:420px;background:#000;display:block" '
        f'src="data:video/mp4;base64,{b64}"></video>'
        f'<div style="font-family:monospace;font-size:.5rem;color:#243040;'
        f'letter-spacing:2px;margin-top:4px">'
        f'OUTPUT/annotated.mp4 &nbsp;·&nbsp; {size_mb} MB</div>'
    )


def check_tg(token: str, chat: str):
    ok, msg = test_connection(token, chat)
    dot_cls = "ok" if ok else "err"
    status  = f"> [{'OK' if ok else 'HATA'}] {msg}"
    # inject dot class into the HTML badge via a returned pair
    badge = f'<div class="tg-head"><span class="tg-icon">✈</span><span class="tg-title">Telegram Bildirimi</span><span class="tg-status-dot {dot_cls}"></span></div>'
    return status, badge


# ─────────────────────────────────────────────────────────────────────────────
def build_ui():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    gr.set_static_paths([_OUTPUT_DIR])

    with gr.Blocks(
        css=CSS,
        title="Hirsizlik Analiz",
        theme=gr.themes.Base(
            primary_hue="neutral",
            secondary_hue="neutral",
            neutral_hue="neutral",
        ),
    ) as demo:

        gr.HTML(FONTS)
        gr.HTML(HEADER)

        # ── üst satır: upload sol + output sag ──────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=340):
                gr.HTML('<div class="sec"><span class="sec-mark">//</span> Video Girdisi</div>')
                video_in = gr.Video(label="Analiz edilecek kayit", height=260)

                with gr.Row():
                    run_btn = gr.Button("ANALİZ BAŞLAT", variant="primary", size="lg")
                    clr_btn = gr.Button("Sifirla", variant="secondary")

            with gr.Column(scale=2, min_width=400):
                gr.HTML('<div class="sec"><span class="sec-mark">//</span> Analiz Ciktisi</div>')
                video_out = gr.HTML(_video_html(None))

        # ── Telegram + Ayarlar yan yana ─────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                tg_badge = gr.HTML(
                    '<div class="tg-head">'
                    '<span class="tg-icon">✈</span>'
                    '<span class="tg-title">Telegram Bildirimi</span>'
                    '<span class="tg-status-dot"></span>'
                    '</div>'
                )
                with gr.Group(elem_id="tg-panel"):
                    with gr.Row():
                        tg_token = gr.Textbox(label="Bot Token", type="password",
                                              value=TELEGRAM_TOKEN,
                                              placeholder="123456:ABC-DEF...", scale=3)
                        tg_chat  = gr.Textbox(label="Chat ID", value=TELEGRAM_CHAT_ID,
                                              placeholder="-100xxxxxxx", scale=2)
                    tg_btn = gr.Button("Baglanti Test Et", variant="secondary", size="sm")
                    tg_st  = gr.Textbox(label="Durum", interactive=False, lines=1,
                                        value="> Token ve chat id girin, test edin")

            with gr.Column(scale=1):
                with gr.Accordion("Model & Parametreler", open=False):
                    model_r  = gr.Radio(["Hizli (nano)", "Dengeli (medium)", "Hassas (large)"],
                                        value="Dengeli (medium)", label="YOLOv8 agirlik")
                    dwell_sl = gr.Slider(2, 30, value=DEFAULT_DWELL_THRESHOLD, step=0.5,
                                         label="Bekleme esigi (saniye)")
                    move_sl  = gr.Slider(10, 200, value=DEFAULT_MOVE_THRESHOLD, step=5,
                                         label="Hareket esigi (piksel std)")

        # ── stats ────────────────────────────────────────────────────────────
        stats_box = gr.HTML(STATS_EMPTY)

        # ── gallery ──────────────────────────────────────────────────────────
        gr.HTML('<div class="sec"><span class="sec-mark">//</span> Suphe Tespitleri — En Kaliteli Kare</div>')
        gallery = gr.Gallery(
            label="",
            columns=5,
            rows=2,
            height=290,
            object_fit="cover",
        )

        # ── log ──────────────────────────────────────────────────────────────
        gr.HTML('<div class="sec"><span class="sec-mark">//</span> Sistem Logu</div>')
        log_box = gr.Textbox(label="", lines=8, interactive=False, elem_id="log-box")

        # ── wiring ───────────────────────────────────────────────────────────
        run_btn.click(
            fn=analyze,
            inputs=[video_in, tg_token, tg_chat, model_r, dwell_sl, move_sl],
            outputs=[video_out, gallery, log_box, stats_box],
        )
        clr_btn.click(
            fn=lambda: (_video_html(None), None, [], "", STATS_EMPTY),
            outputs=[video_out, video_in, gallery, log_box, stats_box],
        )
        tg_btn.click(
            fn=check_tg,
            inputs=[tg_token, tg_chat],
            outputs=[tg_st, tg_badge],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_ui().launch(
        server_port=7860,
        server_name="0.0.0.0",
        share=False,
        show_error=True,
    )
