#!/usr/bin/env python3
"""
TriNetra-AMRF — Tactical Threat Intelligence Console (Phases 6-8)
================================================================

Run the fusion detector + ByteTrack (with the TRC gate) on a synchronized
RGB + thermal clip and *see* what the system does, in an olive-green military
console: a STATUS board, a COMMAND terminal of live events, a vertical SYS
gauge of thermal reliability (TRC), the optical/thermal feed, a SECTOR SCAN
radar that plots the current frame's tracks, a THREAT board with its reasons,
an alert LOG, a SIGNAL activity meter and a TARGETS count by class. Tabs below
hold the track tables, timelines, trajectories and the alert log (snapshots and
JSON download).

Two ways to use it:
  * "LOAD EXAMPLE"     instant (pre-computed run saved earlier)
  * "EXECUTE"          pick a dataset clip or upload your own pair; optionally
                       damage the thermal stream to watch TRC fall and a sensor
                       alert fire

Left half of each frame = RGB with tracks; right half = thermal with the same
boxes. Playback is a frame scrubber with a Play button (browsers cannot play the
OpenCV-written MP4s, and no ffmpeg is installed). Dataset replay only: no live
camera is connected.

Usage:
    python app/tracking_ui.py                 # http://127.0.0.1:7862
    python app/tracking_ui.py --device cpu
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.video import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, clip_paths, process_video  # noqa: E402
from utils.visualize_tracks import _color_for  # noqa: E402

RUNS = PROJECT_ROOT / "runs" / "tracking_ui"
EXAMPLE = PROJECT_ROOT / "runs" / "tracking" / "smoke_seq07"
TABLE_COLS = ["ID", "Class", "Conf", "x (centre)", "y (feet)", "Behaviour flags (whole run)"]
SUMMARY_COLS = ["ID", "Class", "First frame", "Last frame", "Frames tracked", "Mean TRC", "Behaviour flags"]
ALERT_COLS = ["#", "Frame", "Kind", "Level", "Alert", "Why"]
LEVEL_NAMES = {1: "benign", 2: "low", 3: "elevated", 4: "high", 5: "critical"}

_cfg = yaml.safe_load(open(DEFAULT_CONFIG, encoding="utf-8"))
CLASS_NAMES = {c["id"]: c["name"] for c in _cfg["datasets"]["categories"]}
NIGHT_RGB = float(_cfg.get("risk_model", {}).get("threat", {}).get("night_rgb_mean", 70.0))
SEQUENCES = [f"Sequence {i:02d} · {'night' if i <= 10 else 'day'}" for i in range(1, 15)]

# ── look: olive military console ─────────────────────────────────────────────
BG, PANEL, PANEL2, BORDER = "#16211a", "#1d2b20", "#233426", "#5f7d56"
TEXT, MUTED, BRIGHT = "#bcd3a0", "#7d966c", "#e2f0c6"
AMBER, RED, ACCENT = "#e3b448", "#e0644f", "#a8c97f"
LEVEL_HEX = {1: "#8fbf6a", 2: "#bfd36a", 3: AMBER, 4: "#e88a3c", 5: RED}

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.lime,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Share Tech Mono"), "ui-monospace", "monospace"],
    font_mono=[gr.themes.GoogleFont("Share Tech Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(**{f"{k}{s}": v for k, v in {
    "body_background_fill": BG,
    "body_text_color": TEXT,
    "body_text_color_subdued": MUTED,
    "background_fill_primary": PANEL,
    "background_fill_secondary": PANEL2,
    "block_background_fill": PANEL,
    "block_border_color": BORDER,
    "block_border_width": "1px",
    "block_label_background_fill": PANEL2,
    "block_label_text_color": MUTED,
    "block_title_text_color": TEXT,
    "border_color_primary": BORDER,
    "input_background_fill": "#142017",
    "input_border_color": BORDER,
    "button_primary_background_fill": "#3e5a33",
    "button_primary_background_fill_hover": "#4d6e40",
    "button_primary_text_color": BRIGHT,
    "button_primary_border_color": ACCENT,
    "button_secondary_background_fill": PANEL2,
    "button_secondary_background_fill_hover": "#2c412e",
    "button_secondary_text_color": TEXT,
    "button_secondary_border_color": BORDER,
    "slider_color": ACCENT,
    "checkbox_background_color_selected": ACCENT,
    "table_even_background_fill": PANEL,
    "table_odd_background_fill": PANEL2,
    "table_border_color": BORDER,
    "color_accent_soft": "#2c412e",
    "link_text_color": ACCENT,
}.items() for s in ("", "_dark")})

SCREWS = (f"radial-gradient(circle at 9px 9px, #0e150f 3px, {BORDER} 4px, transparent 5px),"
          f"radial-gradient(circle at calc(100% - 9px) 9px, #0e150f 3px, {BORDER} 4px, transparent 5px),"
          f"radial-gradient(circle at 9px calc(100% - 9px), #0e150f 3px, {BORDER} 4px, transparent 5px),"
          f"radial-gradient(circle at calc(100% - 9px) calc(100% - 9px), #0e150f 3px, {BORDER} 4px, transparent 5px)")

CSS = f"""
.gradio-container {{max-width: 1720px !important; background:
  repeating-linear-gradient(0deg, rgba(0,0,0,.10) 0 1px, transparent 1px 3px),
  radial-gradient(1200px 700px at 50% 0%, #22321f, {BG} 70%) !important;
  border:2px solid #31452c; border-radius:22px; box-shadow: inset 0 0 0 6px #121b14, inset 0 0 60px rgba(0,0,0,.6);}}
.gradio-container * {{text-transform:uppercase; letter-spacing:.06em;}}
.gradio-container input, .gradio-container textarea, .tn-why, .tn-log .r, .tn-term {{text-transform:none;}}
footer {{display:none !important;}}
.mil {{position:relative; border:1px solid {BORDER}; border-radius:10px; padding:16px 18px 18px;
  background:{SCREWS}, linear-gradient(180deg, {PANEL}, #1a271d); box-shadow: inset 0 0 0 1px rgba(0,0,0,.35), inset 0 0 30px rgba(0,0,0,.35);}}
.mil-h {{font-family:'Stardos Stencil','Share Tech Mono',monospace; font-size:20px; color:{BRIGHT}; letter-spacing:.14em;
  border-bottom:1px solid {BORDER}; padding:0 0 8px 14px; margin:0 0 10px; display:flex; justify-content:space-between; align-items:baseline;}}
.mil-h small {{font-family:'Share Tech Mono',monospace; font-size:12px; color:{MUTED};}}
.tn-top {{display:grid; grid-template-columns: 1fr auto; gap:14px;}}
.tn-title {{display:flex; align-items:center; gap:16px; font-family:'Stardos Stencil','Share Tech Mono',monospace;
  font-size:30px; color:{BRIGHT}; letter-spacing:.22em; padding:6px 10px;}}
.tn-title .ico {{display:grid; grid-template-columns:repeat(3,7px); gap:3px;}}
.tn-title .ico i {{display:block; width:7px; height:7px; background:{ACCENT};}}
.tn-title span {{font-family:'Share Tech Mono',monospace; font-size:13px; color:{MUTED}; letter-spacing:.12em;}}
.tn-clock {{font-family:'Share Tech Mono',monospace; font-size:34px; color:{BRIGHT}; letter-spacing:.12em; padding:4px 18px; min-width:190px; text-align:center;}}
.tn-rows div {{display:flex; justify-content:space-between; gap:10px; border-bottom:1px solid rgba(95,125,86,.35);
  padding:7px 4px; font-size:16px;}}
.tn-rows div b {{color:{BRIGHT}; font-weight:400;}}
.tn-rows div em {{font-style:normal; color:{MUTED};}}
.tn-term {{font-size:15px; line-height:1.75; color:{BRIGHT}; min-height:150px;}}
.tn-term .ok {{color:{TEXT};}} .tn-term .warn {{color:{AMBER};}} .tn-term .bad {{color:{RED};}}
.tn-cursor {{display:inline-block; width:9px; height:15px; background:{BRIGHT}; vertical-align:-2px; animation:tnblink 1s steps(1) infinite;}}
@keyframes tnblink {{50% {{opacity:0}}}}
.tn-gauge {{display:flex; gap:10px; height:300px;}}
.tn-gauge .scale {{display:flex; flex-direction:column; justify-content:space-between; font-size:13px; color:{MUTED}; text-align:right;}}
.tn-gauge .tube {{position:relative; width:30px; border:1px solid {BORDER}; background:
  repeating-linear-gradient(0deg, transparent 0 7px, rgba(95,125,86,.35) 7px 8px), #142017;}}
.tn-gauge .fill {{position:absolute; left:3px; right:3px; bottom:3px; background:
  repeating-linear-gradient(0deg, currentColor 0 5px, transparent 5px 8px);}}
.tn-gauge .mark {{position:absolute; left:-6px; right:-6px; border-top:1px dashed {RED};}}
.tn-big {{font-size:26px; color:{BRIGHT}; text-align:center; margin-top:10px;}}
.tn-radar svg {{width:100%; height:auto; display:block;}}
.tn-sweep {{transform-origin:150px 150px; animation:tnsweep 4s linear infinite;}}
@keyframes tnsweep {{to {{transform:rotate(360deg)}}}}
.tn-lvl {{font-family:'Stardos Stencil','Share Tech Mono',monospace; font-size:58px; line-height:1;}}
.tn-pips {{display:flex; gap:5px; margin:10px 0;}}
.tn-pips span {{flex:1; height:10px; border:1px solid {BORDER};}}
.tn-why {{margin:0; padding-left:18px; color:{TEXT}; font-size:14px; line-height:1.6;}}
.tn-pulse {{animation:tnpulse 1.4s ease-in-out infinite;}}
@keyframes tnpulse {{50% {{box-shadow: inset 0 0 0 1px rgba(0,0,0,.35), 0 0 18px rgba(224,100,79,.55);}}}}
.tn-flash {{margin-top:10px; font-size:13px; color:#1b120c; background:{RED}; padding:4px 10px; display:inline-block;}}
.tn-log {{max-height:250px; overflow-y:auto;}}
.tn-log > div {{border-bottom:1px solid rgba(95,125,86,.35); padding:7px 2px; font-size:14px;}}
.tn-log .t {{color:{BRIGHT};}} .tn-log .m {{color:{MUTED}; font-size:12px;}} .tn-log .r {{color:{TEXT}; font-size:12.5px; margin-top:2px;}}
.tn-sig {{display:flex; align-items:flex-end; gap:5px; height:70px; padding:0 4px;}}
.tn-sig span {{flex:1; background:{ACCENT};}}
.tn-tg {{display:grid; grid-template-columns:repeat(3,1fr); gap:8px;}}
.tn-tg div {{border:1px solid rgba(95,125,86,.45); padding:6px 8px; text-align:center;}}
.tn-tg b {{display:block; font-size:24px; color:{BRIGHT}; font-weight:400;}}
.tn-tg em {{font-style:normal; font-size:11px; color:{MUTED};}}
.tn-caption {{font-size:13px; color:{MUTED}; padding:4px 6px;}}
.tn-caption b {{color:{BRIGHT}; font-weight:400;}} .tn-dmg {{color:{AMBER};}}
.tn-status {{font-size:13px; color:{TEXT}; border:1px dashed {BORDER}; padding:10px 12px; line-height:1.6;}}
.tn-status.err {{border-color:{RED}; color:#f3b3a6;}}
.tn-frame img {{filter: sepia(.18) saturate(.9); border:1px solid {BORDER};}}
.tn-side {{background:{SCREWS}, linear-gradient(180deg, {PANEL}, #1a271d) !important; border:1px solid {BORDER} !important;
  border-radius:10px !important; padding:14px !important;}}
.tn-feedwrap {{background:{SCREWS}, linear-gradient(180deg, {PANEL}, #1a271d) !important; border:1px solid {BORDER} !important;
  border-radius:10px !important; padding:12px !important;}}
button.primary {{border:1px solid {ACCENT} !important; font-size:17px !important; letter-spacing:.18em !important;}}
.tab-nav button, button[role=tab] {{font-family:'Stardos Stencil','Share Tech Mono',monospace !important; letter-spacing:.12em;}}
@media (max-width: 1100px) {{ .tn-top {{grid-template-columns:1fr;}} }}
"""

HEAD = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Stardos+Stencil:wght@400;700&family=Share+Tech+Mono&display=swap">
<script>
setInterval(function () {
  var d = new Date(), p = function (n) { return String(n).padStart(2, '0'); };
  var s = p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
  document.querySelectorAll('.tn-clock').forEach(function (e) { e.textContent = s; });
}, 1000);
</script>
"""

TOPBAR = """
<div class="tn-top">
  <div class="mil"><div class="tn-title"><div class="ico"><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <div>TRINETRA &nbsp;TACTICAL INTERFACE<br><span>RGB-THERMAL THREAT INTELLIGENCE · DATASET REPLAY (M3FD) · NO LIVE CAMERA</span></div></div></div>
  <div class="mil"><div class="tn-clock">--:--:--</div></div>
</div>
"""


# ── helpers ──────────────────────────────────────────────────────────────────
def _read_frame(path: str, idx: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, f = cap.read()
    cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


def _load_state(prefix: Path, start: int, vis_video: str, stress=None, label="") -> dict:
    records = [json.loads(l) for l in Path(str(prefix) + ".frames.jsonl").read_text(encoding="utf-8").splitlines()]
    tracks = json.loads(Path(str(prefix) + ".tracks.json").read_text(encoding="utf-8"))
    alerts_path = Path(str(prefix) + ".alerts.json")
    alerts = json.loads(alerts_path.read_text(encoding="utf-8")) if alerts_path.exists() else []
    cap = cv2.VideoCapture(str(vis_video))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 768)
    cap.release()
    return {"video": str(prefix) + ".mp4", "records": records, "tracks": tracks, "start": int(start),
            "vis_video": str(vis_video), "stress": list(stress) if stress else None, "label": label,
            "alerts": alerts, "alerts_path": str(alerts_path) if alerts_path.exists() else None, "size": size}


def _alerts_df(state, upto_frame=None) -> pd.DataFrame:
    rows = [[a["alert_id"], a["frame"], a["kind"], f"{a['level']} ({a['label']})", a["title"], "; ".join(a["reasons"])]
            for a in (state or {}).get("alerts", []) if upto_frame is None or a["frame"] <= upto_frame]
    return pd.DataFrame(rows[::-1], columns=ALERT_COLS)          # newest first


def _gallery(state):
    return [(a["snapshot"], f"#{a['alert_id']} frame {a['frame']}: {a['title']}")
            for a in (state or {}).get("alerts", []) if a.get("snapshot") and Path(a["snapshot"]).exists()]


def _esc(s) -> str:
    return html.escape(str(s))


# ── panels ───────────────────────────────────────────────────────────────────
def _status_board(state=None, rec=None, i=0) -> str:
    if not rec:
        rows = [("LINK", "STANDBY"), ("STATUS", "IDLE"), ("SOURCE", "—"), ("SYS", "AWAITING DATA"), ("FRAME", "—")]
        color = MUTED
    else:
        t = rec.get("threat") or {}
        trc = float(rec["trc"])
        sys_txt = "NORMAL" if trc >= 0.7 else "MARGINAL" if trc >= 0.5 else "DEGRADED"
        rgb = t.get("rgb_mean")
        scene = ("NIGHT" if rgb < NIGHT_RGB else "DAY") if rgb is not None else "—"
        src = state["label"].split(" · frames")[0].replace("Sequence ", "SEQ ")
        rows = [("LINK", "ONLINE"), ("STATUS", "ACTIVE"), ("SOURCE", _esc(src)),
                ("SYS", f"{sys_txt} · TRC {trc:.2f}"), ("SCENE", f"{scene} · LUX {rgb:.0f}" if rgb is not None else scene),
                ("FRAME", f"{rec['frame']:04d} / {state['records'][-1]['frame']:04d}")]
        color = BRIGHT
    body = "".join(f'<div><b style="color:{color}">{k}</b><em>{v}</em></div>' for k, v in rows)
    return f'<div class="mil"><p class="mil-h">STATUS <small>TN-01</small></p><div class="tn-rows">{body}</div></div>'


def _command(state=None, rec=None) -> str:
    if not rec:
        lines = ['<span class="ok">&gt; SYSTEM BOOT ........ OK</span>', '<span class="ok">&gt; FUSION MODEL ....... LOADED</span>',
                 '<span class="ok">&gt; AWAITING MISSION INPUT</span>']
    else:
        t = rec.get("threat") or {}
        trc = float(rec["trc"])
        lvl = int(t.get("level", 0)) if t else 0
        lines = ['<span class="ok">&gt; LINK ESTABLISHED</span>',
                 '<span class="ok">&gt; RGB+THERMAL SYNC ... OK</span>',
                 (f'<span class="ok">&gt; THERMAL TRC {trc:.2f} .. RELIABLE</span>' if trc >= 0.5 else
                  f'<span class="bad">&gt; THERMAL TRC {trc:.2f} .. DEGRADED</span>'),
                 f'<span class="ok">&gt; TRACKING {len(rec["tracks"]):02d} TARGETS</span>']
        if t.get("thermal_only"):
            lines.append(f'<span class="warn">&gt; {t["thermal_only"]} HIDDEN FROM RGB · THERMAL ONLY</span>')
        if lvl:
            cls = "bad" if lvl >= 4 else "warn" if lvl == 3 else "ok"
            lines.append(f'<span class="{cls}">&gt; THREAT LEVEL {lvl} · {LEVEL_NAMES[lvl].upper()}</span>')
        for a in state.get("alerts", []):
            if a["frame"] == rec["frame"]:
                lines.append(f'<span class="bad">&gt; ALERT #{a["alert_id"]} {_esc(a["title"]).upper()}</span>')
    return (f'<div class="mil"><p class="mil-h">COMMAND <small>&lt;&lt;&lt;</small></p>'
            f'<div class="tn-term">{"<br>".join(lines)}<br>&gt; SYSTEM READY <span class="tn-cursor"></span></div></div>')


def _gauge(rec=None) -> str:
    trc = float(rec["trc"]) if rec else 0.0
    col = ACCENT if trc >= 0.7 else AMBER if trc >= 0.5 else RED
    fill = max(0.0, min(1.0, trc)) * 100
    scale = "".join(f"<span>{v:.2f}</span>" for v in (1.0, 0.75, 0.5, 0.25, 0.0))
    return (f'<div class="mil"><p class="mil-h">SYS <small>TRC</small></p><div class="tn-gauge">'
            f'<div class="scale">{scale}</div><div class="tube" style="color:{col}">'
            f'<div class="fill" style="height:calc({fill:.0f}% - 6px)"></div><div class="mark" style="bottom:50%"></div></div></div>'
            f'<div class="tn-big" style="color:{col}">{trc:.2f}</div></div>')


def _radar(state=None, rec=None) -> str:
    """SECTOR SCAN: the frame becomes a 120-degree forward sector; lower in the image = closer to the sensor."""
    cx, cy, R = 150, 150, 132
    rings = "".join(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{BORDER}" stroke-width="1"/>' for r in (33, 66, 99, 132))
    spokes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{cx + R * math.sin(math.radians(a)):.1f}" y2="{cy - R * math.cos(math.radians(a)):.1f}" '
        f'stroke="{BORDER}" stroke-width="1" stroke-dasharray="{"3 4" if a % 90 else "0"}"/>' for a in range(0, 360, 30))
    sector = (f'<path d="M{cx},{cy} L{cx + R * math.sin(math.radians(-60)):.1f},{cy - R * math.cos(math.radians(-60)):.1f} '
              f'A{R},{R} 0 0,1 {cx + R * math.sin(math.radians(60)):.1f},{cy - R * math.cos(math.radians(60)):.1f} Z" '
              f'fill="rgba(168,201,127,.06)" stroke="{ACCENT}" stroke-width="1"/>')
    sweep = (f'<g class="tn-sweep"><path d="M{cx},{cy} L{cx},{cy - R} A{R},{R} 0 0,1 {cx + R * math.sin(math.radians(28)):.1f},'
             f'{cy - R * math.cos(math.radians(28)):.1f} Z" fill="url(#sw)"/>'
             f'<line x1="{cx}" y1="{cy}" x2="{cx + R * math.sin(math.radians(28)):.1f}" y2="{cy - R * math.cos(math.radians(28)):.1f}" '
             f'stroke="{BRIGHT}" stroke-width="1.5"/></g>')
    blips = ""
    if rec:
        w, h = state.get("size", (1024, 768))
        lvl = int((rec.get("threat") or {}).get("level", 1))
        for tid, x1, y1, x2, y2, cls, conf in rec["tracks"]:
            ang = math.radians(((x1 + x2) / 2 / w - 0.5) * 120)
            rad = 18 + (1 - min(1.0, y2 / h)) * (R - 26)
            bx, by = cx + rad * math.sin(ang), cy - rad * math.cos(ang)
            col = RED if lvl >= 4 else AMBER if lvl == 3 else BRIGHT
            if int(cls) == 1:
                blips += (f'<polygon points="{bx:.1f},{by - 8:.1f} {bx - 7:.1f},{by + 5:.1f} {bx + 7:.1f},{by + 5:.1f}" '
                          f'fill="none" stroke="{col}" stroke-width="1.6"/><text x="{bx:.1f}" y="{by + 3:.1f}" font-size="8" '
                          f'text-anchor="middle" fill="{col}">!</text>')
            else:
                blips += f'<rect x="{bx - 6:.1f}" y="{by - 5:.1f}" width="12" height="10" fill="none" stroke="{col}" stroke-width="1.6"/>'
            blips += f'<text x="{bx + 9:.1f}" y="{by - 6:.1f}" font-size="8" fill="{MUTED}">{int(tid)}</text>'
    svg = (f'<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" font-family="Share Tech Mono, monospace">'
           f'<defs><linearGradient id="sw" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{ACCENT}" stop-opacity="0"/>'
           f'<stop offset="1" stop-color="{ACCENT}" stop-opacity=".35"/></linearGradient></defs>'
           f'{rings}{spokes}{sector}{sweep}<circle cx="{cx}" cy="{cy}" r="3" fill="{BRIGHT}"/>{blips}</svg>')
    n = len(rec["tracks"]) if rec else 0
    return (f'<div class="mil tn-radar"><p class="mil-h">SECTOR SCAN <small>{n:02d} CONTACTS · FWD 120°</small></p>{svg}</div>')


def _threat_board(rec=None) -> str:
    if not rec or not rec.get("threat"):
        msg = "NO DATA — LOAD EXAMPLE OR EXECUTE A CLIP" if not rec else "RUN RECORDED BEFORE THREAT SCORING — RE-RUN THE CLIP"
        return f'<div class="mil"><p class="mil-h">THREAT <small>1–5</small></p><div class="tn-caption">{msg}</div></div>'
    t = rec["threat"]
    lvl = int(t["level"])
    col = LEVEL_HEX[lvl]
    pips = "".join(f'<span style="background:{LEVEL_HEX[k] if k <= lvl else "transparent"}"></span>' for k in range(1, 6))
    why = "".join(f"<li>{_esc(r)}</li>" for r in t["reasons"])
    flash = '<div class="tn-flash">ALERT RAISED ON THIS FRAME</div>' if rec.get("alerts") else ""
    return (f'<div class="mil{" tn-pulse" if lvl >= 4 else ""}" style="border-color:{col}"><p class="mil-h">THREAT '
            f'<small style="color:{col}">{LEVEL_NAMES[lvl].upper()}</small></p>'
            f'<div class="tn-lvl" style="color:{col}">{lvl}<span style="font-size:24px;color:{MUTED}">/5</span></div>'
            f'<div class="tn-pips">{pips}</div><ul class="tn-why">{why}</ul>{flash}</div>')


def _log(state=None, upto_frame=None) -> str:
    alerts = [a for a in (state or {}).get("alerts", []) if upto_frame is None or a["frame"] <= upto_frame]
    if not alerts:
        body = '<div class="tn-caption">NO ALERTS LOGGED</div>'
    else:
        body = "".join(
            f'<div><div class="t" style="color:{LEVEL_HEX.get(a["level"], BRIGHT)}">#{a["alert_id"]:03d} · F{a["frame"]:04d} · '
            f'{_esc(a["title"]).upper()}</div><div class="m">{a["kind"].upper()} · LVL {a["level"]} · TRC {a.get("trc", 0):.2f}</div>'
            f'<div class="r">{_esc("; ".join(a["reasons"]))}</div></div>' for a in alerts[::-1])
    return f'<div class="mil"><p class="mil-h">LOG <small>ALERTS</small></p><div class="tn-log">{body}</div></div>'


def _signal(state=None, i=0) -> str:
    vals = [0] * 12
    if state:
        recs = state["records"][max(0, i - 11):i + 1]
        vals = ([0] * (12 - len(recs))) + [r["n_dets"] for r in recs]
    top = max(vals) or 1
    bars = "".join(f'<span style="height:{max(4, v / top * 100):.0f}%;opacity:{0.45 + 0.55 * (k == 11):.2f}"></span>'
                   for k, v in enumerate(vals))
    return (f'<div class="mil"><p class="mil-h">SIGNAL <small>DETECTIONS · LAST 12 FR</small></p>'
            f'<div class="tn-sig">{bars}</div></div>')


def _targets(rec=None) -> str:
    counts = {n: 0 for n in ("person", "car", "bus", "truck", "motorcycle", "lamp")}
    if rec:
        for _, *_box, cls, _conf in rec["tracks"]:
            name = CLASS_NAMES.get(int(cls), "other")
            counts[name] = counts.get(name, 0) + 1
    hidden = (rec.get("threat") or {}).get("thermal_only", 0) if rec else 0
    short = {"person": "PERSON", "car": "CAR", "bus": "BUS", "truck": "TRUCK", "motorcycle": "MOTO"}
    cells = "".join(f'<div><b>{counts.get(k, 0):02d}</b><em>{v}</em></div>' for k, v in short.items())
    cells += f'<div><b style="color:{AMBER if hidden else BRIGHT}">{hidden:02d}</b><em>IR-only</em></div>'
    return f'<div class="mil"><p class="mil-h">TARGETS <small>BY CLASS</small></p><div class="tn-tg">{cells}</div></div>'


def _caption_html(state, rec, i) -> str:
    dmg = ""
    if state.get("stress") and state["stress"][0] <= rec["frame"] < state["stress"][1]:
        dmg = ' · <span class="tn-dmg">THERMAL DAMAGE ACTIVE</span>'
    return (f'<div class="tn-caption"><b>FRAME {rec["frame"]:04d}</b> ({i + 1}/{len(state["records"])}) · '
            f'LEFT: OPTICAL (RGB) WITH TRACKS · RIGHT: THERMAL (IR){dmg}</div>')


def _status(state, extra="") -> str:
    n = len(state["records"])
    ids = {t["track_id"] for t in state["tracks"]}
    levels = [r["threat"]["level"] for r in state["records"] if r.get("threat")]
    peak = f" · PEAK THREAT {max(levels)}/5" if levels else ""
    return (f'<div class="tn-status">MISSION COMPLETE<br>{_esc(state["label"])}<br>{n} FRAMES · {len(ids)} TRACKS{peak} · '
            f'{len(state.get("alerts", []))} ALERTS<br><span style="color:{MUTED}">{_esc(extra)}</span></div>')


# ── plots (olive, to match the console) ──────────────────────────────────────
def _dark(fig, *axes):
    fig.patch.set_facecolor(PANEL)
    for ax in axes:
        ax.set_facecolor("#18241b")
        for s in ax.spines.values():
            s.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.yaxis.label.set_color(MUTED); ax.xaxis.label.set_color(MUTED)
        ax.title.set_color(BRIGHT)
        ax.grid(color=BORDER, lw=0.5, alpha=0.5)


def _threat_timeline(state):
    fig, ax = plt.subplots(figsize=(8, 2.6))
    _dark(fig, ax)
    recs = [r for r in state["records"] if r.get("threat")]
    if not recs:
        ax.text(0.5, 0.5, "NO THREAT SCORES IN THIS RUN", ha="center", va="center", color=MUTED); ax.axis("off")
        return fig
    fr = [r["frame"] for r in recs]
    lv = [r["threat"]["level"] for r in recs]
    ax.fill_between(fr, lv, step="mid", color=AMBER, alpha=0.15)
    ax.step(fr, lv, where="mid", color=AMBER, lw=2)
    for a in state.get("alerts", []):
        ax.axvline(a["frame"], color=LEVEL_HEX[a["level"]], lw=1.3, ls="--")
        ax.text(a["frame"], 5.3, f"#{a['alert_id']}", fontsize=7, ha="center", color=LEVEL_HEX[a["level"]])
    ax.set_ylim(0.5, 5.7); ax.set_yticks([1, 2, 3, 4, 5]); ax.set_ylabel("THREAT LEVEL"); ax.set_xlabel("FRAME")
    ax.set_title("THREAT LEVEL PER FRAME  (DASHED = ALERT)", fontsize=9)
    fig.tight_layout()
    return fig


def _summary_df(state) -> pd.DataFrame:
    rows = [[t["track_id"], t.get("cls_name") or CLASS_NAMES.get(t["cls"], t["cls"]), t["first_frame"],
             t["last_frame"], t["n_frames"], t["mean_trc"], ", ".join(t["flags"]) or "-"] for t in state["tracks"]]
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


def _timeline(state):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax2 = ax.twinx()
    _dark(fig, ax, ax2)
    ax2.grid(False)
    fr = [r["frame"] for r in state["records"]]
    if state.get("stress"):
        ax.axvspan(state["stress"][0], state["stress"][1], color=AMBER, alpha=0.15, label="thermal damage")
    ax.axhline(0.5, color=RED, lw=0.8, ls=":")
    ax.plot(fr, [r["trc"] for r in state["records"]], color=ACCENT, lw=2, label="frame TRC")
    ax.set_ylim(0, 1.02); ax.set_ylabel("TRC"); ax.set_xlabel("FRAME")
    ax2.step(fr, [len(r["tracks"]) for r in state["records"]], color=BRIGHT, lw=1.2, where="mid", label="tracks")
    ax2.set_ylabel("ACTIVE TRACKS")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left", facecolor=PANEL2, edgecolor=BORDER)
    for t in leg.get_texts():
        t.set_color(TEXT)
    ax.set_title("THERMAL RELIABILITY (TRC) AND ACTIVE TRACKS  (DOTTED = SENSOR-ALERT THRESHOLD)", fontsize=9)
    fig.tight_layout()
    return fig


def _trajectories(state):
    bg = _read_frame(state["vis_video"], state["start"])
    fig, ax = plt.subplots(figsize=(7, 5.6))
    fig.patch.set_facecolor(PANEL)
    if bg is not None:
        g = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        ax.imshow(np.dstack([g * 0.35, g * 0.5, g * 0.3]).astype(np.uint8))
    paths = {}
    for r in state["records"]:
        for tid, x1, y1, x2, y2, cls, conf in r["tracks"]:
            paths.setdefault(int(tid), []).append(((x1 + x2) / 2, y2))
    for tid, pts in paths.items():
        b, gg, rr = _color_for(tid)
        c = (rr / 255, gg / 255, b / 255)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=c, lw=2.2)
        ax.scatter([xs[-1]], [ys[-1]], color=c, s=24, zorder=3, edgecolors="white", linewidths=0.6)
        ax.text(xs[-1] + 6, ys[-1], f"#{tid}", color="white", fontsize=7, bbox=dict(facecolor=c, pad=1, lw=0))
    ax.axis("off")
    ax.set_title(f"TRACK PATHS (FEET POSITION) OVER {len(state['records'])} FRAMES", fontsize=9, color=BRIGHT)
    fig.tight_layout()
    return fig


def _safe_progress(progress, done, total):
    try:
        progress(done / total, desc=f"Analysing frame {done}/{total}")
    except Exception:       # a progress-bar hiccup must never abort a run
        pass


# ── callbacks ────────────────────────────────────────────────────────────────
def show_example():
    if not (EXAMPLE.with_suffix(".mp4").exists()):
        return _empty("Pre-computed example not found — use EXECUTE.")
    vis, _ = clip_paths(7)
    state = _load_state(EXAMPLE, 0, vis, label="Pre-computed example (older run, TRC gate on)")
    return _outputs(state, _status(state))


def _empty(msg):
    return (None, gr.Slider(), f'<div class="tn-status err">{_esc(msg)}</div>', pd.DataFrame(columns=SUMMARY_COLS),
            None, None, None, pd.DataFrame(columns=ALERT_COLS), [], None)


def _outputs(state, status_html):
    return (state, gr.Slider(maximum=max(1, len(state["records"]) - 1), value=0), status_html,
            _summary_df(state), _timeline(state), _trajectories(state), _threat_timeline(state),
            _alerts_df(state), _gallery(state), state.get("alerts_path"))


def run_tracking(source, seq_name, up_vis, up_thr, start, n_frames, trc_gate, stress, stress_off, stress_len,
                 device, progress=gr.Progress()):
    try:
        if source == "Upload a pair":
            if not up_vis or not up_thr:
                raise ValueError("Upload both an RGB and a thermal video.")
            vis, thr, label = str(up_vis), str(up_thr), "Uploaded pair"
        else:
            seq_no = int(seq_name.split()[1])
            vis, thr = (str(p) for p in clip_paths(seq_no))
            label = seq_name
        start, n_frames = int(start), int(n_frames)
        stress_window = None
        if stress != "none":
            stress_window = (start + int(stress_off), start + int(stress_off) + int(stress_len))
        out_dir = RUNS / time.strftime("%Y%m%d_%H%M%S")
        prefix = out_dir / "tracking"

        res = process_video(
            vis, thr, str(prefix.with_suffix(".mp4")), checkpoint=str(DEFAULT_CHECKPOINT),
            config_path=str(DEFAULT_CONFIG), device=None if device == "auto" else device,
            start=start, max_frames=n_frames, layout="both", out_scale=0.5, trc_gate=bool(trc_gate),
            stress=stress, stress_frames=stress_window, log_every=10_000,
            progress_cb=lambda done, total: _safe_progress(progress, done, total))
        label += f" · frames {start}-{start + n_frames} · TRC gate {'on' if trc_gate else 'off'}"
        if stress != "none":
            label += f" · thermal {stress}"
        state = _load_state(prefix, start, vis, stress_window, label)
        extra = f"Processed {res['frames']} frames at {res['fps']:.1f} fps ({res['elapsed_s']:.0f} s)."
        return _outputs(state, _status(state, extra))
    except Exception as e:  # surface the reason in the UI instead of a blank error
        return _empty(f"Failed: {e}")


def render(state, idx):
    if not state:
        return (None, "", pd.DataFrame(columns=TABLE_COLS), _status_board(), _command(), _gauge(), _radar(),
                _threat_board(), _log(), _signal(), _targets())
    i = int(np.clip(idx, 0, len(state["records"]) - 1))
    rec = state["records"][i]
    frame = _read_frame(state["video"], i)
    flags = {t["track_id"]: t["flags"] for t in state["tracks"]}
    rows = [[int(tid), CLASS_NAMES.get(int(cls), cls), round(conf, 2), int((x1 + x2) / 2), int(y2),
             ", ".join(f for f in flags.get(int(tid), []) if f != "low_trc_track") or "-"]
            for tid, x1, y1, x2, y2, cls, conf in rec["tracks"]]
    return (frame, _caption_html(state, rec, i), pd.DataFrame(rows, columns=TABLE_COLS),
            _status_board(state, rec, i), _command(state, rec), _gauge(rec), _radar(state, rec),
            _threat_board(rec), _log(state, upto_frame=rec["frame"]), _signal(state, i), _targets(rec))


def step(state, idx):
    if not state:
        return 0
    n = len(state["records"])
    return 0 if idx + 1 >= n else idx + 1


# ── UI ───────────────────────────────────────────────────────────────────────
def build_app(default_device: str):
    with gr.Blocks(title="TriNetra — Tactical Interface") as demo:
        gr.HTML(TOPBAR)
        state = gr.State(None)

        # mission input: one compact bar under the header
        with gr.Column(elem_classes="tn-side"):
            gr.HTML('<p class="mil-h">MISSION INPUT <small>SELECT CLIP · OPTIONAL THERMAL DAMAGE · EXECUTE</small></p>')
            with gr.Row(equal_height=True):
                source = gr.Radio(["Dataset clip", "Upload a pair"], value="Dataset clip", label="Source",
                                  scale=3, min_width=200)
                seq = gr.Dropdown(SEQUENCES, value=SEQUENCES[5], label="Dataset clip", scale=3, min_width=190)
                start = gr.Number(0, label="Start", precision=0, minimum=0, scale=1, min_width=80)
                n_frames = gr.Number(40, label="Frames", precision=0, minimum=1, maximum=400, scale=1, min_width=80)
                stress = gr.Radio(["none", "blur", "saturate", "flat"], value="none", label="Thermal damage",
                                  scale=4, min_width=280)
                stress_off = gr.Number(5, label="Damage at", precision=0, minimum=0, scale=1, min_width=80)
                stress_len = gr.Number(10, label="Length", precision=0, minimum=1, scale=1, min_width=80)
                with gr.Column(scale=2, min_width=170):
                    run_btn = gr.Button("▶ EXECUTE", variant="primary", size="lg")
                    ex_btn = gr.Button("LOAD EXAMPLE", variant="secondary", size="sm")
            with gr.Group(visible=False) as upload_box:
                with gr.Row():
                    up_vis = gr.File(label="RGB video", file_types=["video"])
                    up_thr = gr.File(label="Thermal video", file_types=["video"])
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    trc_gate = gr.Checkbox(True, label="TRC gate on (off = stock ByteTrack)")
                    device = gr.Radio(["auto", "cuda", "cpu"], value=default_device, label="Device")

        with gr.Row(equal_height=False):
            # left: status, command terminal, mission result
            with gr.Column(scale=5, min_width=300):
                status_board = gr.HTML(_status_board())
                command = gr.HTML(_command())
                status = gr.HTML('<div class="tn-status">SELECT CLIP AND EXECUTE<br>'
                                 '<span style="color:#7d966c">≈1.5 S PER FRAME ON CPU</span></div>')

            # centre: SYS gauge + feed, then log / signal / targets
            with gr.Column(scale=12):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=120):
                        gauge = gr.HTML(_gauge())
                    with gr.Column(scale=7, elem_classes="tn-feedwrap"):
                        gr.HTML('<p class="mil-h">OPTICAL / THERMAL FEED <small>RGB | IR</small></p>')
                        image = gr.Image(show_label=False, type="numpy", interactive=False, height=360,
                                         elem_classes="tn-frame")
                        caption = gr.HTML("")
                        with gr.Row():
                            frame_slider = gr.Slider(0, 99, value=0, step=1, label="Frame", scale=6)
                            play_btn = gr.Button("▶ PLAY", scale=1, min_width=90)
                            pause_btn = gr.Button("❚❚ HOLD", scale=1, min_width=90)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        log = gr.HTML(_log())
                    with gr.Column(scale=2):
                        signal = gr.HTML(_signal())
                        targets = gr.HTML(_targets())

            # right: radar + threat board
            with gr.Column(scale=6, min_width=300):
                radar = gr.HTML(_radar())
                threat = gr.HTML(_threat_board())

        with gr.Tabs():
            with gr.Tab("Tracks in frame"):
                table = gr.Dataframe(value=pd.DataFrame(columns=TABLE_COLS), interactive=False)
            with gr.Tab("Track summary"):
                summary = gr.Dataframe(value=pd.DataFrame(columns=SUMMARY_COLS), interactive=False)
            with gr.Tab("Timelines"):
                timeline = gr.Plot(show_label=False)
                threat_plot = gr.Plot(show_label=False)
            with gr.Tab("Trajectories"):
                traj = gr.Plot(show_label=False)
            with gr.Tab("Alert log"):
                alerts_all = gr.Dataframe(value=pd.DataFrame(columns=ALERT_COLS), interactive=False, wrap=True)
                gallery = gr.Gallery(label="Alert snapshots", columns=3, height=320)
                alert_file = gr.File(label="Download alert log (JSON)")

        timer = gr.Timer(0.15, active=False)
        outs = [state, frame_slider, status, summary, timeline, traj, threat_plot, alerts_all, gallery, alert_file]
        view = [image, caption, table, status_board, command, gauge, radar, threat, log, signal, targets]
        source.change(lambda s: (gr.Group(visible=s == "Upload a pair"), gr.Dropdown(visible=s != "Upload a pair")),
                      source, [upload_box, seq])
        run_btn.click(run_tracking, [source, seq, up_vis, up_thr, start, n_frames, trc_gate, stress,
                                     stress_off, stress_len, device], outs
                      ).then(render, [state, frame_slider], view)
        ex_btn.click(show_example, None, outs).then(render, [state, frame_slider], view)
        frame_slider.change(render, [state, frame_slider], view, show_progress="hidden")
        play_btn.click(lambda: gr.Timer(active=True), None, timer)
        pause_btn.click(lambda: gr.Timer(active=False), None, timer)
        timer.tick(step, [state, frame_slider], frame_slider, show_progress="hidden")
    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description="TriNetra tactical threat intelligence console")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    build_app(args.device).queue().launch(server_name="127.0.0.1", server_port=args.port, share=False,
                                        allowed_paths=[str(RUNS)], theme=THEME, css=CSS, head=HEAD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
