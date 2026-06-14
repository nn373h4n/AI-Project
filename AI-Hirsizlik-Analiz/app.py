import base64
import logging
import os
import subprocess
import threading
import warnings
from datetime import datetime

import cv2
import gradio as gr
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="gradio")

from config import (
    DEFAULT_DWELL_THRESHOLD,
    DEFAULT_FRAME_SKIP,
    DEFAULT_MODEL,
    DEFAULT_MOVE_THRESHOLD,
    TELEGRAM_CHAT_ID,
    TELEGRAM_TOKEN,
)
from db import (
    export_dataset,
    get_accuracy_stats,
    get_unlabeled,
    init_db,
    log_analysis,
    log_detection,
    save_detection_image,
    update_label,
)
from detector import TheftDetector
from notifier import send_telegram_alert, test_connection
from reporter import make_report_html

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
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

/* ── HEADER ─────────────── */
#hdr {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: flex-end;
  padding: 0 0 20px;
  margin-bottom: 28px;
  border-bottom: 1px solid var(--bd);
  position: relative; overflow: hidden;
}
#hdr::before {
  content:''; position:absolute; top:-40px; left:-32px;
  width:320px; height:120px;
  background:radial-gradient(ellipse,rgba(245,165,32,.06) 0%,transparent 70%);
  pointer-events:none;
}
#hdr::after {
  content:''; position:absolute; bottom:-1px; left:0;
  width:90px; height:2px;
  background:linear-gradient(90deg,var(--acc),transparent);
}
.eyebrow {
  font-family:var(--f-b); font-size:.5rem; color:var(--txt-lo);
  letter-spacing:5px; text-transform:uppercase; margin-bottom:10px;
  display:flex; align-items:center; gap:10px;
}
.eyebrow::before {
  content:''; display:inline-block; width:18px; height:1px;
  background:var(--acc); box-shadow:0 0 8px var(--acc);
}
#hdr-title {
  font-family:var(--f-d); font-size:2rem; color:var(--txt-hi);
  letter-spacing:4px; margin:0; line-height:1; text-transform:uppercase;
}
#hdr-title span { color:var(--acc); text-shadow:0 0 32px var(--acc-g); }
#hdr-right {
  text-align:right; display:flex; flex-direction:column;
  align-items:flex-end; gap:10px;
}
#live-clock {
  font-family:var(--f-d); font-size:1.15rem; color:var(--txt);
  letter-spacing:3px; min-width:200px;
}
.chips { display:flex; gap:6px; }
.chip {
  font-family:var(--f-b); font-size:.46rem; letter-spacing:1.8px;
  text-transform:uppercase; border:1px solid var(--bd); color:var(--txt-lo);
  padding:3px 10px; display:flex; align-items:center; gap:5px;
  transition:border-color .2s,color .2s;
}
.chip.live { border-color:var(--grn); color:var(--grn); }
.chip.live:hover { background:var(--grn-g); }
.ldot {
  width:4px; height:4px; border-radius:50%; background:var(--grn);
  box-shadow:0 0 7px var(--grn); animation:lb 1.9s ease-in-out infinite; flex-shrink:0;
}
@keyframes lb { 0%,100%{opacity:1} 50%{opacity:.08} }

/* ── SECTION HEADER ──────── */
.sec {
  font-family:var(--f-b); font-size:.48rem; color:var(--txt-lo);
  letter-spacing:3.5px; text-transform:uppercase; padding-bottom:10px;
  margin-bottom:14px; border-bottom:1px solid var(--bd);
  display:flex; align-items:center; gap:10px;
}
.sec-mark { font-family:var(--f-d); font-size:.75rem; color:var(--acc); line-height:1; }

/* ── TABS ────────────────── */
.tabs>.tab-nav {
  background:var(--s2) !important;
  border-bottom:1px solid var(--bd) !important;
  border-radius:0 !important;
  padding:0 !important;
}
.tabs>.tab-nav>button {
  font-family:var(--f-b) !important; font-size:.52rem !important;
  letter-spacing:2.5px !important; text-transform:uppercase !important;
  color:var(--txt-lo) !important; border-radius:0 !important;
  border:none !important; border-bottom:2px solid transparent !important;
  padding:12px 22px !important; background:transparent !important;
  transition:color .2s,border-color .2s !important;
}
.tabs>.tab-nav>button.selected {
  color:var(--acc) !important; border-bottom-color:var(--acc) !important;
}
.tabs>.tab-nav>button:hover:not(.selected) { color:var(--txt) !important; }
.tabitem { padding:20px 0 0 0 !important; background:transparent !important; }

/* ── BLOCKS ──────────────── */
.block,.gr-form,.form,.panel,.gap {
  background:var(--s1) !important; border:1px solid var(--bd) !important;
  border-radius:0 !important;
}
.block { border-top-color:var(--bd2) !important; }

/* ── LABELS ──────────────── */
.label-wrap span,.block label>span {
  font-family:var(--f-b) !important; color:var(--txt-lo) !important;
  font-size:.52rem !important; text-transform:uppercase !important;
  letter-spacing:2.5px !important;
}

/* ── INPUTS ──────────────── */
input[type=text],input[type=password],textarea {
  background:rgba(4,6,15,.95) !important; border:1px solid var(--bd) !important;
  color:var(--txt) !important; font-family:var(--f-b) !important;
  font-size:.76rem !important; border-radius:0 !important;
  padding:9px 13px !important; transition:border-color .2s,color .2s !important;
}
input[type=text]:focus,input[type=password]:focus,textarea:focus {
  border-color:var(--bd2) !important; color:var(--txt-hi) !important;
  box-shadow:inset 0 0 0 1px rgba(245,165,32,.06) !important; outline:none !important;
}

/* ── BUTTONS ──────────────── */
button {
  font-family:var(--f-b) !important; font-size:.58rem !important;
  letter-spacing:3.5px !important; text-transform:uppercase !important;
  border-radius:0 !important; cursor:pointer !important; transition:all .2s !important;
}
button.primary {
  background:transparent !important; border:1px solid var(--acc) !important;
  color:var(--acc) !important; position:relative !important; overflow:hidden !important;
}
button.primary:hover {
  background:var(--acc) !important; color:#000 !important;
  box-shadow:0 0 28px var(--acc-g) !important;
}
button.secondary {
  background:transparent !important; border:1px solid var(--bd) !important;
  color:var(--txt-lo) !important;
}
button.secondary:hover { border-color:var(--bd2) !important; color:var(--txt) !important; }

/* ── SLIDERS ──────────────── */
input[type=range] { accent-color:var(--acc) !important; background:transparent !important; border:none !important; }

/* ── ACCORDION ────────────── */
.accordion { background:var(--s2) !important; border:1px solid var(--bd) !important; border-radius:0 !important; }
.accordion>button { font-size:.54rem !important; letter-spacing:2px !important; color:var(--txt-lo) !important; }

/* ── VIDEO ────────────────── */
video { border:1px solid var(--bd) !important; border-radius:0 !important; display:block !important; }
#vid-in .block::after,#vid-out .block::after {
  content:'' !important; position:absolute !important; inset:0 !important;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.05) 3px,rgba(0,0,0,.05) 4px) !important;
  pointer-events:none !important; z-index:5 !important;
}
#vid-in,#vid-out { position:relative; }
#vid-in::before,#vid-in::after,#vid-out::before,#vid-out::after {
  content:''; position:absolute; width:18px; height:18px;
  border-color:rgba(245,165,32,.5); border-style:solid; z-index:10; pointer-events:none;
}
#vid-in::before,#vid-out::before { top:6px; left:6px; border-width:1px 0 0 1px; }
#vid-in::after,#vid-out::after { bottom:6px; right:6px; border-width:0 1px 1px 0; }

/* ── GALLERY ──────────────── */
.gallery-item { border:1px solid rgba(192,57,43,.6) !important; border-radius:0 !important; overflow:hidden !important; }
.gallery-item img { filter:contrast(1.08) saturate(.8) !important; border-radius:0 !important; transition:filter .3s !important; }
.gallery-item:hover img { filter:contrast(1.12) saturate(.95) !important; }

/* ── LOG ──────────────────── */
#log-box textarea {
  font-family:var(--f-d) !important; font-size:.68rem !important;
  color:var(--data) !important; background:#020409 !important;
  border-color:var(--bd) !important; line-height:1.9 !important;
  letter-spacing:.5px !important; text-shadow:0 0 12px rgba(0,229,160,.15) !important;
}

/* ── PROGRESS ──────────────── */
.progress-bar,[class*=progress-level] {
  background:linear-gradient(90deg,var(--acc),rgba(245,165,32,.6)) !important;
  box-shadow:0 0 14px var(--acc-g) !important; border-radius:0 !important;
}

/* ── STATS GRID ────────────── */
#stats-row {
  display:grid; grid-template-columns:repeat(4,1fr);
  gap:1px; background:var(--bd); border:1px solid var(--bd); margin:24px 0;
}
.sc {
  background:var(--s1); padding:16px 20px;
  display:flex; flex-direction:column; gap:6px;
  position:relative; overflow:hidden;
}
.sc::after {
  content:''; position:absolute; top:0; left:0;
  width:2px; height:100%; background:var(--bd2);
}
.sc:first-child::after { background:var(--acc); box-shadow:0 0 10px var(--acc-g); }
.sc:nth-child(3)::after { background:var(--red); box-shadow:0 0 10px var(--red-g); }
.sc-lbl { font-family:var(--f-b); font-size:.47rem; color:var(--txt-lo); letter-spacing:2.5px; text-transform:uppercase; }
.sc-val { font-family:var(--f-d); font-size:1.5rem; color:var(--txt-hi); line-height:1; letter-spacing:1px; }
.sc-val.d { color:var(--red);  text-shadow:0 0 20px var(--red-g); }
.sc-val.g { color:var(--grn);  text-shadow:0 0 20px var(--grn-g); }
.sc-val.a { color:var(--acc);  text-shadow:0 0 20px var(--acc-g); }
.sc-sub { font-family:var(--f-b); font-size:.45rem; color:var(--txt-lo); letter-spacing:1px; margin-top:2px; }

/* ── TELEGRAM PANEL ─────────── */
#tg-panel { background:var(--s2); border:1px solid var(--bd2); padding:16px 18px; margin-top:14px; position:relative; overflow:hidden; }
#tg-panel::before { content:''; position:absolute; top:0; left:0; width:100%; height:2px; background:linear-gradient(90deg,#2196F3,transparent); }
.tg-head { display:flex; align-items:center; gap:10px; margin-bottom:14px; }
.tg-icon { font-size:1.1rem; line-height:1; }
.tg-title { font-family:var(--f-b); font-size:.54rem; color:#5baef7; letter-spacing:2.5px; text-transform:uppercase; }
.tg-status-dot { width:6px; height:6px; border-radius:50%; background:#555; margin-left:auto; transition:background .3s,box-shadow .3s; }
.tg-status-dot.ok  { background:var(--grn); box-shadow:0 0 8px var(--grn-g); }
.tg-status-dot.err { background:var(--red); box-shadow:0 0 8px var(--red-g); }

/* ── LABEL CARD ─────────────── */
#label-img .block { border-color:var(--bd2) !important; min-height:280px; }
.label-info {
  font-family:var(--f-b); font-size:.72rem; color:var(--txt);
  background:var(--s2); border:1px solid var(--bd); padding:14px 16px; margin-top:8px;
}
.label-info .li-id { font-family:var(--f-d); font-size:.9rem; color:var(--acc); margin-bottom:8px; }
.label-info .li-row { margin-bottom:4px; }
.label-info .li-dim { color:var(--txt-lo); font-size:.62rem; }
.acc-grid {
  display:grid; grid-template-columns:1fr 1fr; gap:1px;
  background:var(--bd); border:1px solid var(--bd);
}
.acc-cell {
  background:var(--s1); padding:12px 16px;
  display:flex; flex-direction:column; gap:4px;
}
.acc-lbl { font-family:var(--f-b); font-size:.44rem; letter-spacing:2px; text-transform:uppercase; color:var(--txt-lo); }
.acc-val { font-family:var(--f-d); font-size:1.3rem; color:var(--txt-hi); }
.train-log {
  font-family:var(--f-d) !important; font-size:.65rem !important;
  color:#00e5a0 !important; background:#020409 !important;
  border-color:var(--bd) !important; line-height:1.8 !important;
}
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
    <div class="eyebrow">Güvenlik Analiz Sistemi · CAM/AI/GPU</div>
    <h1 id="hdr-title">HIRSIZLIk <span>ANALİZ</span></h1>
  </div>
  <div id="hdr-right">
    <div id="live-clock">--:--:--</div>
    <div class="chips">
      <div class="chip live"><span class="ldot"></span>CANLI</div>
      <div class="chip">GPU</div>
      <div class="chip">YOLOv11</div>
      <div class="chip">ByteTrack</div>
      <div class="chip">Poz</div>
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
    el.textContent=t+' · '+dt;
  }
  tick(); setInterval(tick,1000);
})();
</script>
"""

STATS_EMPTY = """
<div id="stats-row">
  <div class="sc"><div class="sc-lbl">Analiz Edilen Kare</div><div class="sc-val a">—</div><div class="sc-sub">Bekliyor</div></div>
  <div class="sc"><div class="sc-lbl">Tespit Edilen Kişi</div><div class="sc-val">—</div><div class="sc-sub">Bekliyor</div></div>
  <div class="sc"><div class="sc-lbl">Şüphe Tespiti</div><div class="sc-val d">—</div><div class="sc-sub">Bekliyor</div></div>
  <div class="sc"><div class="sc-lbl">İşlem Cihazı</div><div class="sc-val g">—</div><div class="sc-sub">Hazır</div></div>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_det: TheftDetector | None = None
_OUTPUT_DIR = os.path.abspath("output")


def _make(model_choice: str, dwell: float, move: float, use_pose: bool) -> TheftDetector:
    global _det
    m = {
        "Hızlı (nano)":    "yolo11n.pt",
        "Dengeli (medium)": "yolo11m.pt",
        "Hassas (large)":   "yolo11l.pt",
    }
    _det = TheftDetector(
        model_name=m.get(model_choice, DEFAULT_MODEL),
        dwell_threshold=dwell,
        movement_threshold=move,
        frame_skip=DEFAULT_FRAME_SKIP,
        use_pose=use_pose,
    )
    return _det


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _video_html(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return (
            '<div style="height:420px;display:flex;align-items:center;'
            'justify-content:center;color:#243040;font-family:monospace;'
            'font-size:.7rem;letter-spacing:2px;background:#070b15;'
            'border:1px solid #111928">ANALİZ BEKLENİYOR</div>'
        )
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    size_mb = round(os.path.getsize(path) / 1024 / 1024, 1)
    return (
        f'<video controls autoplay muted '
        f'style="width:100%;max-height:420px;background:#000;display:block" '
        f'src="data:video/mp4;base64,{b64}"></video>'
        f'<div style="font-family:monospace;font-size:.5rem;color:#243040;'
        f'letter-spacing:2px;margin-top:4px">'
        f'output/annotated.mp4 · {size_mb} MB</div>'
    )


def analyze(
    video_file,
    tg_token: str,
    tg_chat: str,
    model_choice: str,
    dwell: float,
    move: float,
    use_pose: bool,
    progress=gr.Progress(),
):
    logs = []
    if video_file is None:
        return _video_html(None), [], "> HATA: video dosyası seçilmedi.", STATS_EMPTY

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logs.append(f"> [{_ts()}] Başladı  |  Cihaz:{device}  Model:{model_choice}  Poz:{'Açık' if use_pose else 'Kapalı'}")
    logs.append(f"> [{_ts()}] Girdi: {os.path.basename(str(video_file))}")

    det         = _make(model_choice, dwell, move, use_pose)
    started_at  = datetime.now().isoformat()

    def prog(p: float):
        progress(p, desc=f"İşlem {p*100:.0f}%")

    try:
        out_video, flagged, frame_count = det.process_video(video_file, progress_fn=prog)
    except Exception as exc:
        return _video_html(None), [], f"> HATA: {exc}", STATS_EMPTY

    finished_at = datetime.now().isoformat()
    total       = len(det.tracks)
    logs.append(f"> [{_ts()}] Tamamlandı  kare:{frame_count}  kişi:{total}  şüphe:{len(flagged)}")

    # Log to DB
    try:
        analysis_id = log_analysis(
            str(video_file), started_at, finished_at,
            frame_count, total, len(flagged), model_choice, device,
        )
    except Exception:
        analysis_id = -1

    gallery = []
    for item in flagged:
        tid      = item["track_id"]
        bgr      = item["frame"]
        reason   = item["reason"]
        dwell_t  = item["dwell_time"]
        vid_ts   = det.tracks[tid].last_ts if tid in det.tracks else 0.0

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gallery.append((Image.fromarray(rgb), f"#{tid}  {reason[:48]}"))

        # Save detection image + log to DB
        try:
            img_path = save_detection_image(bgr, analysis_id, tid)
            log_detection(analysis_id, tid, reason, dwell_t, vid_ts, img_path)
        except Exception:
            pass

        if tg_token and tg_chat:
            ok = send_telegram_alert(tg_token, tg_chat, bgr, tid, reason, dwell_t, vid_ts)
            logs.append(f"> [{_ts()}] Telegram #{tid}: {'GÖNDERILDI' if ok else 'HATA'}")
        else:
            logs.append(f"> [{_ts()}] #{tid} tespit — Telegram yapılandırılmamış")

    if not flagged:
        logs.append(f"> [{_ts()}] Şüpheli davranış tespit edilmedi.")

    dc  = "d" if flagged else "g"
    stats = f"""
<div id="stats-row">
  <div class="sc">
    <div class="sc-lbl">Analiz Edilen Kare</div>
    <div class="sc-val a">{frame_count:,}</div>
    <div class="sc-sub">@25–60 fps</div>
  </div>
  <div class="sc">
    <div class="sc-lbl">Tespit Edilen Kişi</div>
    <div class="sc-val">{total}</div>
    <div class="sc-sub">Benzersiz kimlik</div>
  </div>
  <div class="sc">
    <div class="sc-lbl">Şüphe Tespiti</div>
    <div class="sc-val {dc}">{len(flagged)}</div>
    <div class="sc-sub">{"Bildirim gönderildi" if flagged and tg_token else "Kayıt alındı"}</div>
  </div>
  <div class="sc">
    <div class="sc-lbl">İşlem Cihazı</div>
    <div class="sc-val g">{device}</div>
    <div class="sc-sub">YOLOv11 · ByteTrack · Poz</div>
  </div>
</div>
"""
    progress(1.0, desc="Tamamlandı")
    return _video_html(out_video), gallery, "\n".join(logs), stats


def check_tg(token: str, chat: str):
    ok, msg = test_connection(token, chat)
    dot_cls = "ok" if ok else "err"
    status  = f"> [{'OK' if ok else 'HATA'}] {msg}"
    badge   = (
        '<div class="tg-head">'
        '<span class="tg-icon">✈</span>'
        '<span class="tg-title">Telegram Bildirimi</span>'
        f'<span class="tg-status-dot {dot_cls}"></span>'
        '</div>'
    )
    return status, badge


# ─── Rapor tab ───────────────────────────────────────────────────────────────
def generate_report(period: str):
    try:
        return make_report_html(period)
    except Exception as exc:
        return f'<p style="color:#c0392b;font-family:monospace">Rapor hatası: {exc}</p>'


# ─── Öğrenme tab ─────────────────────────────────────────────────────────────
def _acc_html() -> str:
    s       = get_accuracy_stats()
    labeled = (s.get("true_pos", 0) or 0) + (s.get("false_pos", 0) or 0)
    prec    = ((s.get("true_pos", 0) or 0) / labeled * 100) if labeled > 0 else 0.0
    fp_rate = ((s.get("false_pos", 0) or 0) / labeled * 100) if labeled > 0 else 0.0
    score   = max(0.0, min(100.0, prec))

    bar_w   = int(score)
    bar_col = "#00c17a" if score >= 80 else ("#f5a520" if score >= 50 else "#c0392b")

    return f"""
<div style="font-family:'IBM Plex Mono',monospace">
  <div class="acc-grid" style="margin-bottom:12px">
    <div class="acc-cell">
      <div class="acc-lbl">Toplam Tespit</div>
      <div class="acc-val">{s.get('total',0)}</div>
    </div>
    <div class="acc-cell">
      <div class="acc-lbl">Etiketlendi</div>
      <div class="acc-val">{labeled}</div>
    </div>
    <div class="acc-cell">
      <div class="acc-lbl" style="color:#00c17a">Doğru Pozitif</div>
      <div class="acc-val" style="color:#00c17a">{s.get('true_pos',0)}</div>
    </div>
    <div class="acc-cell">
      <div class="acc-lbl" style="color:#c0392b">Yanlış Pozitif</div>
      <div class="acc-val" style="color:#c0392b">{s.get('false_pos',0)}</div>
    </div>
  </div>
  <div style="font-size:.44rem;letter-spacing:2px;text-transform:uppercase;color:#1e2d45;margin-bottom:6px">
    Kesinlik Skoru
  </div>
  <div style="background:#070a18;border:1px solid #111a2e;height:22px;position:relative;margin-bottom:4px">
    <div style="width:{bar_w}%;height:100%;background:{bar_col};opacity:.8;transition:width .5s"></div>
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:1.4rem;color:{bar_col};
              text-shadow:0 0 16px {bar_col}44">
    {prec:.1f}% <span style="font-size:.7rem;color:#1e2d45">kesinlik &nbsp;/&nbsp; {fp_rate:.1f}% YP oranı</span>
  </div>
</div>
"""


def _det_info_html(det: dict | None, idx: int, total: int) -> str:
    if not det:
        return (
            '<div class="label-info">'
            '<div class="li-id">—</div>'
            '<div class="li-dim">Yenilemek için ↑ Yenile butonuna basın</div>'
            '</div>'
        )
    return (
        f'<div class="label-info">'
        f'<div class="li-id">Kişi #{det["track_id"]}  <span style="color:#1e2d45;font-size:.6rem">'
        f'[{idx+1}/{total}]</span></div>'
        f'<div class="li-row">{det["reason"]}</div>'
        f'<div class="li-row">Bekleme: {det["dwell_time"]:.1f}s</div>'
        f'<div class="li-row li-dim">{det["detected_at"][:19]}</div>'
        f'<div class="li-row li-dim">{os.path.basename(det.get("video_file","?"))}</div>'
        f'</div>'
    )


def load_unlabeled_fn():
    rows = get_unlabeled(60)
    idx  = 0
    if not rows:
        return rows, idx, None, _det_info_html(None, 0, 0), _acc_html()
    det  = rows[0]
    img  = det["image_path"] if os.path.exists(det.get("image_path", "")) else None
    return rows, idx, img, _det_info_html(det, 0, len(rows)), _acc_html()


def _show_at(rows, idx):
    if not rows:
        return None, _det_info_html(None, 0, 0)
    idx = max(0, min(idx, len(rows) - 1))
    det = rows[idx]
    img = det["image_path"] if os.path.exists(det.get("image_path", "")) else None
    return img, _det_info_html(det, idx, len(rows))


def label_fn(rows, idx, confirmed: int):
    if rows and 0 <= idx < len(rows):
        try:
            update_label(rows[idx]["id"], confirmed)
        except Exception:
            pass
    new_idx = min(idx + 1, len(rows) - 1) if rows else 0
    img, info = _show_at(rows, new_idx)
    return new_idx, img, info, _acc_html()


def nav_fn(rows, idx, delta: int):
    new_idx = max(0, min(idx + delta, len(rows) - 1)) if rows else 0
    img, info = _show_at(rows, new_idx)
    return new_idx, img, info


def export_ds_fn():
    ds_dir = os.path.abspath("data/dataset")
    try:
        tp, fp = export_dataset(ds_dir)
        cmd = f"yolo classify train model=yolo11n-cls.pt data={ds_dir} epochs=50 imgsz=128"
        return (
            f'<div style="font-family:monospace;color:#00c17a;padding:12px;background:#020409;border:1px solid #111a2e">'
            f'Dataset dışa aktarıldı → <b>{ds_dir}</b><br>'
            f'Doğru: <b>{tp}</b>  Yanlış: <b>{fp}</b><br><br>'
            f'<span style="color:#6a7c96">Fine-tune için çalıştırın:</span><br>'
            f'<span style="color:#f5a520">{cmd}</span>'
            f'</div>'
        )
    except Exception as exc:
        return f'<div style="color:#c0392b;font-family:monospace;padding:8px">{exc}</div>'


_train_log_lock = threading.Lock()
_train_log_lines: list = []


def start_training_fn():
    ds_dir = os.path.abspath("data/dataset")
    tp_dir = os.path.join(ds_dir, "theft")
    if not os.path.isdir(tp_dir) or len(os.listdir(tp_dir)) == 0:
        return "> HATA: Önce 'Dataset Dışa Aktar' yapın ve etiketlenmiş veri olduğundan emin olun."

    global _train_log_lines
    _train_log_lines = [f"> [{_ts()}] Eğitim başlatılıyor..."]

    def _run():
        cmd = [
            "python", "-m", "ultralytics",
            "classify", "train",
            f"model=yolo11n-cls.pt",
            f"data={ds_dir}",
            "epochs=50", "imgsz=128", "batch=8",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.strip()
                if line:
                    with _train_log_lock:
                        _train_log_lines.append(f"> {line}")
            proc.wait()
            with _train_log_lock:
                _train_log_lines.append(f"> [{_ts()}] Eğitim tamamlandı. Çıkış kodu: {proc.returncode}")
        except Exception as exc:
            with _train_log_lock:
                _train_log_lines.append(f"> HATA: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return "\n".join(_train_log_lines)


def poll_train_log_fn():
    with _train_log_lock:
        return "\n".join(_train_log_lines[-60:]) if _train_log_lines else "> Eğitim bekleniyor..."


# ─────────────────────────────────────────────────────────────────────────────
def build_ui():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    try:
        init_db()
    except Exception as exc:
        logging.warning("DB init hatası: %s", exc)

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

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # TAB 1 — ANALİZ
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("Analiz"):

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=340):
                        gr.HTML('<div class="sec"><span class="sec-mark">//</span> Video Girdisi</div>')
                        video_in = gr.Video(label="Analiz edilecek kayıt", height=260, elem_id="vid-in")
                        with gr.Row():
                            run_btn = gr.Button("ANALİZ BAŞLAT", variant="primary", size="lg")
                            clr_btn = gr.Button("Sıfırla", variant="secondary")

                    with gr.Column(scale=2, min_width=400):
                        gr.HTML('<div class="sec"><span class="sec-mark">//</span> Analiz Çıktısı</div>')
                        video_out = gr.HTML(_video_html(None), elem_id="vid-out")

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
                            tg_btn = gr.Button("Bağlantı Test Et", variant="secondary", size="sm")
                            tg_st  = gr.Textbox(label="Durum", interactive=False, lines=1,
                                                value="> Token ve chat id girin, test edin")

                    with gr.Column(scale=1):
                        with gr.Accordion("Model & Parametreler", open=False):
                            model_r  = gr.Radio(
                                ["Hızlı (nano)", "Dengeli (medium)", "Hassas (large)"],
                                value="Dengeli (medium)", label="YOLOv11 Ağırlık",
                            )
                            use_pose_cb = gr.Checkbox(label="Poz Tahmini (hareket analizi)", value=True)
                            dwell_sl = gr.Slider(2, 30, value=DEFAULT_DWELL_THRESHOLD, step=0.5,
                                                 label="Bekleme eşiği (saniye)")
                            move_sl  = gr.Slider(10, 200, value=DEFAULT_MOVE_THRESHOLD, step=5,
                                                 label="Hareket eşiği (piksel std)")

                stats_box = gr.HTML(STATS_EMPTY)

                gr.HTML('<div class="sec"><span class="sec-mark">//</span> Şüphe Tespitleri — En Kaliteli Kare</div>')
                gallery = gr.Gallery(label="", columns=5, rows=2, height=290, object_fit="cover")

                gr.HTML('<div class="sec"><span class="sec-mark">//</span> Sistem Logu</div>')
                log_box = gr.Textbox(label="", lines=8, interactive=False, elem_id="log-box")

                # Wiring
                run_btn.click(
                    fn=analyze,
                    inputs=[video_in, tg_token, tg_chat, model_r, dwell_sl, move_sl, use_pose_cb],
                    outputs=[video_out, gallery, log_box, stats_box],
                )
                clr_btn.click(
                    fn=lambda: (_video_html(None), None, [], "", STATS_EMPTY),
                    outputs=[video_out, video_in, gallery, log_box, stats_box],
                )
                tg_btn.click(fn=check_tg, inputs=[tg_token, tg_chat], outputs=[tg_st, tg_badge])

            # ══════════════════════════════════════════════════════════════════
            # TAB 2 — RAPORLAR
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("Raporlar"):

                gr.HTML('<div class="sec"><span class="sec-mark">//</span> Periyot Seçimi</div>')
                with gr.Row():
                    period_r   = gr.Radio(
                        ["gunluk", "haftalik", "aylik", "yillik"],
                        value="haftalik",
                        label="Rapor Periyodu",
                    )
                    rp_btn = gr.Button("Raporu Oluştur", variant="primary")

                report_html = gr.HTML(
                    '<div style="height:200px;display:flex;align-items:center;justify-content:center;'
                    'color:#1e2d45;font-family:monospace;font-size:.7rem;letter-spacing:2px">'
                    'Rapor oluşturmak için bir periyot seçin ve butona basın</div>'
                )

                rp_btn.click(fn=generate_report, inputs=[period_r], outputs=[report_html])

            # ══════════════════════════════════════════════════════════════════
            # TAB 3 — ÖĞRENME
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("Öğrenme"):

                gr.HTML(
                    '<div class="sec"><span class="sec-mark">//</span> Tespit Etiketleme</div>'
                    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.62rem;'
                    'color:#1e2d45;margin-bottom:18px;line-height:1.8">'
                    'Analiz sonrası kaydedilen tespitleri inceleyin. '
                    'Gerçek hırsızlık ise <b style="color:#00c17a">Doğru</b>, '
                    'yanlış alarm ise <b style="color:#c0392b">Yanlış</b> olarak işaretleyin. '
                    'Sistem zamanla kendi skor hesabını günceller.</div>'
                )

                det_state  = gr.State([])
                idx_state  = gr.State(0)

                with gr.Row():
                    with gr.Column(scale=1):
                        lbl_refresh = gr.Button("Yenile (DB\'den Yükle)", variant="secondary")
                        lbl_img     = gr.Image(label="Tespit Karesi", height=280, elem_id="label-img")
                        lbl_info    = gr.HTML(_det_info_html(None, 0, 0))
                        with gr.Row():
                            prev_btn  = gr.Button("← Geri",  variant="secondary", size="sm")
                            true_btn  = gr.Button("✓ Doğru", variant="primary",   size="sm")
                            false_btn = gr.Button("✗ Yanlış", variant="secondary", size="sm")
                            next_btn  = gr.Button("İleri →", variant="secondary", size="sm")

                    with gr.Column(scale=1):
                        gr.HTML('<div class="sec" style="margin-top:0"><span class="sec-mark">//</span> Doğruluk Skoru</div>')
                        acc_html   = gr.HTML(_acc_html())

                        gr.HTML('<div class="sec"><span class="sec-mark">//</span> Dataset & Eğitim</div>')
                        exp_btn    = gr.Button("Dataset Dışa Aktar", variant="secondary")
                        exp_out    = gr.HTML("")
                        train_btn  = gr.Button("Modeli İyileştir (Fine-tune)", variant="primary")
                        train_log  = gr.Textbox(label="Eğitim Logu", lines=8, interactive=False,
                                                elem_id="train-log", elem_classes=["train-log"])
                        poll_btn   = gr.Button("Log Yenile", variant="secondary", size="sm")

                # Öğrenme wiring
                lbl_refresh.click(
                    fn=load_unlabeled_fn,
                    outputs=[det_state, idx_state, lbl_img, lbl_info, acc_html],
                )
                true_btn.click(
                    fn=lambda rows, idx: label_fn(rows, idx, 1),
                    inputs=[det_state, idx_state],
                    outputs=[idx_state, lbl_img, lbl_info, acc_html],
                )
                false_btn.click(
                    fn=lambda rows, idx: label_fn(rows, idx, 0),
                    inputs=[det_state, idx_state],
                    outputs=[idx_state, lbl_img, lbl_info, acc_html],
                )
                prev_btn.click(
                    fn=lambda rows, idx: nav_fn(rows, idx, -1),
                    inputs=[det_state, idx_state],
                    outputs=[idx_state, lbl_img, lbl_info],
                )
                next_btn.click(
                    fn=lambda rows, idx: nav_fn(rows, idx, +1),
                    inputs=[det_state, idx_state],
                    outputs=[idx_state, lbl_img, lbl_info],
                )
                exp_btn.click(fn=export_ds_fn, outputs=[exp_out])
                train_btn.click(fn=start_training_fn, outputs=[train_log])
                poll_btn.click(fn=poll_train_log_fn, outputs=[train_log])

    return demo


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_ui().launch(
        server_port=7860,
        server_name="0.0.0.0",
        share=False,
        show_error=True,
    )
