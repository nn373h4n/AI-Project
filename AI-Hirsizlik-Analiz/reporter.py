import base64
import io
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from db import get_accuracy_stats, get_daily_stats, get_detection_trend

_BG   = "#04060f"
_S1   = "#070a18"
_BD   = "#111a2e"
_BD2  = "#1c2d4a"
_ACC  = "#f5a520"
_RED  = "#c0392b"
_GRN  = "#00c17a"
_TXT  = "#6a7c96"
_TXTH = "#b0c2da"


def _setup_fig(figsize=(10, 3.6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    ax.set_facecolor(_S1)
    for spine in ax.spines.values():
        spine.set_color(_BD)
    ax.tick_params(colors=_TXT, labelsize=7)
    ax.xaxis.label.set_color(_TXT)
    ax.yaxis.label.set_color(_TXT)
    ax.title.set_color(_TXTH)
    ax.grid(True, color=_BD, linewidth=0.5, linestyle="--", alpha=0.5)
    plt.tight_layout(pad=1.4)
    return fig, ax


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _empty_b64(msg: str) -> str:
    fig, ax = _setup_fig()
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=_TXT,
            fontsize=9, fontfamily="monospace", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    return _to_b64(fig)


def suspects_chart(days: int = 7) -> str:
    rows = get_daily_stats(days)
    if not rows:
        return _empty_b64("Veri yok — henüz analiz yapılmadı")

    dates    = [r["day"][5:] for r in rows]
    suspects = [r["suspects"] or 0 for r in rows]
    persons  = [r["persons"]  or 0 for r in rows]

    fig, ax = _setup_fig()
    x, w = np.arange(len(dates)), 0.36
    ax.bar(x - w / 2, persons,  w, color=_BD2,   label="Kişi",  zorder=2)
    ax.bar(x + w / 2, suspects, w, color=_RED,    label="Şüphe", alpha=0.88, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Adet", fontsize=7)
    ax.set_title(f"Son {days} Gün — Kişi & Şüphe Tespiti", fontsize=9, pad=10)

    leg = ax.legend(facecolor=_S1, edgecolor=_BD, fontsize=7)
    for t in leg.get_texts():
        t.set_color(_TXT)

    for i, v in enumerate(suspects):
        if v > 0:
            ax.text(x[i] + w / 2, v + 0.05, str(v),
                    ha="center", va="bottom", color=_RED, fontsize=7)

    return _to_b64(fig)


def accuracy_chart(days: int = 30) -> str:
    rows = get_detection_trend(days)
    if not rows:
        return _empty_b64("Etiketlenmiş veri yok")

    dates     = [r["day"][5:] for r in rows]
    totals    = [r["total"] or 0 for r in rows]
    confirmed = [r["confirmed_tp"] or 0 for r in rows]

    # Running cumulative precision
    prec = []
    cum_tp = cum_tot = 0
    for tp, tot in zip(confirmed, totals):
        cum_tp  += tp
        cum_tot += tot
        prec.append((cum_tp / cum_tot * 100) if cum_tot > 0 else 0)

    fig, ax = _setup_fig()
    x = np.arange(len(dates))
    ax.plot(x, prec, color=_ACC, linewidth=1.6, marker="o", markersize=3, zorder=3)
    ax.fill_between(x, prec, alpha=0.08, color=_ACC)
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Kesinlik %", fontsize=7)
    ax.set_title("Kümülatif Tespit Kesinliği (Etiketlenen)", fontsize=9, pad=10)
    ax.axhline(y=80, color=_GRN, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(0.01, 0.77, "80 % hedef", transform=ax.transAxes,
            color=_GRN, fontsize=6, va="bottom")

    return _to_b64(fig)


def make_report_html(period: str = "haftalik") -> str:
    days_map = {"gunluk": 1, "haftalik": 7, "aylik": 30, "yillik": 365}
    days = days_map.get(period, 7)

    acc   = get_accuracy_stats()
    daily = get_daily_stats(days)

    total_analyses = sum(r.get("analyses", 0) or 0 for r in daily)
    total_suspects = sum(r.get("suspects", 0) or 0 for r in daily)
    total_persons  = sum(r.get("persons",  0) or 0 for r in daily)
    labeled    = (acc.get("true_pos", 0) or 0) + (acc.get("false_pos", 0) or 0)
    precision  = ((acc.get("true_pos", 0) or 0) / labeled * 100) if labeled > 0 else 0

    s_chart = suspects_chart(days)
    a_chart = accuracy_chart(min(days, 30))

    cell = ("background:#070a18;padding:14px 18px;"
            "border-left:2px solid #1c2d4a")
    lbl  = ("font-size:.44rem;letter-spacing:2px;text-transform:uppercase;"
            "color:#1e2d45;margin-bottom:6px")
    val  = "font-size:1.65rem;font-family:'Share Tech Mono',monospace"
    sec  = ("font-size:.44rem;letter-spacing:3px;text-transform:uppercase;"
            "color:#1e2d45;border-bottom:1px solid #111a2e;padding-bottom:7px;margin-bottom:10px")

    period_label = {"gunluk": "Günlük", "haftalik": "Haftalık",
                    "aylik": "Aylık", "yillik": "Yıllık"}.get(period, period.capitalize())

    return f"""
<div style="font-family:'IBM Plex Mono',monospace;color:#6a7c96;padding:4px 0">
  <div style="font-size:.5rem;letter-spacing:4px;text-transform:uppercase;
              color:#f5a520;margin-bottom:16px">{period_label} Rapor</div>

  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
              background:#111a2e;margin-bottom:20px">
    <div style="{cell}">
      <div style="{lbl}">Analiz Sayısı</div>
      <div style="{val};color:#b0c2da">{total_analyses}</div>
    </div>
    <div style="{cell}">
      <div style="{lbl}">Toplam Kişi</div>
      <div style="{val};color:#b0c2da">{total_persons}</div>
    </div>
    <div style="{cell};border-left-color:#c0392b">
      <div style="{lbl}">Şüphe Tespiti</div>
      <div style="{val};color:#c0392b;text-shadow:0 0 16px rgba(192,57,43,.3)">{total_suspects}</div>
    </div>
    <div style="{cell};border-left-color:#f5a520">
      <div style="{lbl}">Kesinlik</div>
      <div style="{val};color:#f5a520;text-shadow:0 0 16px rgba(245,165,32,.2)">{precision:.1f}%</div>
    </div>
  </div>

  <div style="margin-bottom:18px">
    <div style="{sec}">// Şüphe Grafiği</div>
    <img src="data:image/png;base64,{s_chart}"
         style="width:100%;border:1px solid #111a2e;display:block">
  </div>

  <div>
    <div style="{sec}">// Doğruluk Trendi</div>
    <img src="data:image/png;base64,{a_chart}"
         style="width:100%;border:1px solid #111a2e;display:block">
  </div>
</div>
"""
