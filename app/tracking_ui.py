#!/usr/bin/env python3
"""
TriNetra-AMRF — Tracking Demo UI (Phase 6)
==========================================

Run the fusion detector + ByteTrack (with the TRC gate) on a synchronized
RGB + thermal clip and *see* what the tracker does: scrub or play through the
result frame by frame, with persistent IDs, trailing paths, per-track TRC and
behaviour flags, plus a TRC/track-count timeline and trajectory plot.

Two ways to use it:
  * "Show pre-computed example"   instant (sequence 7, 100 frames saved earlier)
  * "Run tracking"                pick a dataset clip or upload your own pair;
                                  optionally damage the thermal stream to watch
                                  TRC fall and how the tracker copes

Left half of each frame = RGB with tracks; right half = thermal with the same
boxes. Playback is a frame scrubber with a Play button (browsers cannot play the
OpenCV-written MP4s, and no ffmpeg is installed).

Usage:
    python app/tracking_ui.py                 # http://127.0.0.1:7862
    python app/tracking_ui.py --device cpu
"""

from __future__ import annotations

import argparse
import json
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

_cfg = yaml.safe_load(open(DEFAULT_CONFIG, encoding="utf-8"))
CLASS_NAMES = {c["id"]: c["name"] for c in _cfg["datasets"]["categories"]}
SEQUENCES = [f"Sequence {i:02d}" for i in range(1, 15)]


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
    return {"video": str(prefix) + ".mp4", "records": records, "tracks": tracks, "start": int(start),
            "vis_video": str(vis_video), "stress": list(stress) if stress else None, "label": label}


def _summary_df(state) -> pd.DataFrame:
    rows = [[t["track_id"], t.get("cls_name") or CLASS_NAMES.get(t["cls"], t["cls"]), t["first_frame"],
             t["last_frame"], t["n_frames"], t["mean_trc"], ", ".join(t["flags"]) or "-"] for t in state["tracks"]]
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


def _timeline(state):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fr = [r["frame"] for r in state["records"]]
    ax.plot(fr, [r["trc"] for r in state["records"]], color="#c0392b", lw=1.6, label="frame TRC")
    ax.set_ylim(0, 1.02); ax.set_ylabel("TRC"); ax.set_xlabel("video frame")
    if state.get("stress"):
        ax.axvspan(state["stress"][0], state["stress"][1], color="#f5b041", alpha=0.25, label="thermal damage")
    ax2 = ax.twinx()
    ax2.step(fr, [len(r["tracks"]) for r in state["records"]], color="#2b6cb0", lw=1.2, where="mid", label="tracks")
    ax2.set_ylabel("active tracks")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    ax.set_title("Thermal Reliability Confidence and active tracks", fontsize=9)
    fig.tight_layout()
    return fig


def _trajectories(state):
    bg = _read_frame(state["vis_video"], state["start"])
    fig, ax = plt.subplots(figsize=(7, 5.6))
    if bg is not None:
        ax.imshow(bg)
    paths = {}
    for r in state["records"]:
        for tid, x1, y1, x2, y2, cls, conf in r["tracks"]:
            paths.setdefault(int(tid), []).append(((x1 + x2) / 2, y2))
    for tid, pts in paths.items():
        b, g, rr = _color_for(tid)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=(rr / 255, g / 255, b / 255), lw=2)
        ax.scatter([xs[-1]], [ys[-1]], color=(rr / 255, g / 255, b / 255), s=18, zorder=3)
        ax.text(xs[-1] + 6, ys[-1], f"#{tid}", color="white", fontsize=7,
                bbox=dict(facecolor=(rr / 255, g / 255, b / 255), pad=1, lw=0))
    ax.axis("off")
    ax.set_title(f"Track paths (feet position) over {len(state['records'])} frames", fontsize=9)
    fig.tight_layout()
    return fig


def _status(state, extra="") -> str:
    n = len(state["records"])
    ids = {t["track_id"] for t in state["tracks"]}
    flagged = [t for t in state["tracks"] if t["flags"] and t["flags"] != ["low_trc_track"]]
    return (f"**{state['label']}** — {n} frames, {len(ids)} confirmed tracks"
            f"{', ' + str(len(flagged)) + ' with behaviour flags' if flagged else ', no behaviour flags'}. {extra}")


def _safe_progress(progress, done, total):
    try:
        progress(done / total, desc=f"Tracking frame {done}/{total}")
    except Exception:       # a progress-bar hiccup must never abort a run
        pass


# ── callbacks ────────────────────────────────────────────────────────────────
def show_example():
    if not (EXAMPLE.with_suffix(".mp4").exists()):
        return None, gr.Slider(), "Pre-computed example not found — use *Run tracking*.", pd.DataFrame(columns=SUMMARY_COLS), None, None
    vis, _ = clip_paths(7)
    state = _load_state(EXAMPLE, 0, vis, label="Sequence 07 (pre-computed, TRC gate on)")
    return (state, gr.Slider(maximum=len(state["records"]) - 1, value=0), _status(state),
            _summary_df(state), _timeline(state), _trajectories(state))


def run_tracking(source, seq_name, up_vis, up_thr, start, n_frames, trc_gate, stress, stress_off, stress_len,
                 device, progress=gr.Progress()):
    try:
        if source == "Upload a pair":
            if not up_vis or not up_thr:
                raise ValueError("Upload both an RGB and a thermal video.")
            vis, thr, label = str(up_vis), str(up_thr), "Uploaded pair"
        else:
            vis, thr = (str(p) for p in clip_paths(int(seq_name.split()[-1])))
            label = seq_name
        start, n_frames = int(start), int(n_frames)
        stress_window = None
        if stress != "none":
            stress_window = (start + int(stress_off), start + int(stress_off) + int(stress_len))
        out_dir = RUNS / time.strftime("%Y%m%d_%H%M%S")
        prefix = out_dir / "tracking"

        t0 = time.time()
        res = process_video(
            vis, thr, str(prefix.with_suffix(".mp4")), checkpoint=str(DEFAULT_CHECKPOINT),
            config_path=str(DEFAULT_CONFIG), device=None if device == "auto" else device,
            start=start, max_frames=n_frames, layout="both", out_scale=0.5, trc_gate=bool(trc_gate),
            stress=stress, stress_frames=stress_window, log_every=10_000,
            progress_cb=lambda done, total: _safe_progress(progress, done, total))
        label += f", frames {start}-{start + n_frames}, TRC gate {'on' if trc_gate else 'off'}"
        if stress != "none":
            label += f", thermal {stress}"
        state = _load_state(prefix, start, vis, stress_window, label)
        extra = f"Processed at {res['fps']:.1f} fps ({res['elapsed_s']:.0f} s)."
        return (state, gr.Slider(maximum=len(state["records"]) - 1, value=0), _status(state, extra),
                _summary_df(state), _timeline(state), _trajectories(state))
    except Exception as e:  # surface the reason in the UI instead of a blank error
        return None, gr.Slider(), f"**Failed:** {e}", pd.DataFrame(columns=SUMMARY_COLS), None, None


def render(state, idx):
    if not state:
        return None, "Run tracking or load the example to begin.", pd.DataFrame(columns=TABLE_COLS)
    i = int(np.clip(idx, 0, len(state["records"]) - 1))
    rec = state["records"][i]
    frame = _read_frame(state["video"], i)
    flags = {t["track_id"]: t["flags"] for t in state["tracks"]}
    rows = [[int(tid), CLASS_NAMES.get(int(cls), cls), round(conf, 2), int((x1 + x2) / 2), int(y2),
             ", ".join(f for f in flags.get(int(tid), []) if f != "low_trc_track") or "-"]
            for tid, x1, y1, x2, y2, cls, conf in rec["tracks"]]
    dmg = ""
    if state.get("stress") and state["stress"][0] <= rec["frame"] < state["stress"][1]:
        dmg = " · **thermal damage active**"
    info = (f"**Frame {rec['frame']}** ({i + 1}/{len(state['records'])}) · frame TRC **{rec['trc']:.2f}** · "
            f"{len(rows)} tracks · {rec['n_dets']} raw detections{dmg}")
    return frame, info, pd.DataFrame(rows, columns=TABLE_COLS)


def step(state, idx):
    if not state:
        return 0
    n = len(state["records"])
    return 0 if idx + 1 >= n else idx + 1


# ── UI ───────────────────────────────────────────────────────────────────────
def build_app(default_device: str):
    with gr.Blocks(title="TriNetra-AMRF — Tracking Demo") as demo:
        gr.Markdown(
            "# TriNetra-AMRF — Tracking Demo\n"
            "Fusion detector → **ByteTrack** with the **TRC gate** → behaviour analysis. "
            "Each frame shows RGB (left, tracks drawn with ID, class, confidence, trail and per-track TRC) and "
            "thermal (right, same boxes). Damage the thermal stream to watch TRC fall.")
        state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Run")
                source = gr.Radio(["Dataset clip", "Upload a pair"], value="Dataset clip", label="Source")
                seq = gr.Dropdown(SEQUENCES, value="Sequence 07", label="Dataset clip")
                up_vis = gr.File(label="RGB video (upload mode)", file_types=["video"])
                up_thr = gr.File(label="Thermal video (upload mode)", file_types=["video"])
                start = gr.Slider(0, 1400, value=0, step=10, label="Start frame (25 fps)")
                n_frames = gr.Slider(30, 300, value=100, step=10, label="Frames to process (≈3.5 fps)")
                trc_gate = gr.Checkbox(True, label="TRC gate on (off = stock ByteTrack)")
                stress = gr.Dropdown(["none", "blur", "saturate", "flat"], value="none",
                                     label="Damage the thermal stream")
                stress_off = gr.Slider(0, 290, value=30, step=10, label="Damage starts (frames after start)")
                stress_len = gr.Slider(10, 200, value=50, step=10, label="Damage length (frames)")
                device = gr.Dropdown(["auto", "cuda", "cpu"], value=default_device, label="Device")
                run_btn = gr.Button("Run tracking", variant="primary")
                ex_btn = gr.Button("Show pre-computed example")
                status = gr.Markdown("Load the example for an instant view, or run tracking on a clip.")

            with gr.Column(scale=3):
                image = gr.Image(label="RGB (left) | thermal (right)", type="numpy", interactive=False)
                info = gr.Markdown()
                with gr.Row():
                    frame_slider = gr.Slider(0, 99, value=0, step=1, label="Frame", scale=5)
                    play_btn = gr.Button("▶ Play", scale=1)
                    pause_btn = gr.Button("⏸ Pause", scale=1)
                table = gr.Dataframe(value=pd.DataFrame(columns=TABLE_COLS), label="Tracks in this frame",
                                     interactive=False)
                with gr.Tabs():
                    with gr.Tab("Track summary"):
                        summary = gr.Dataframe(value=pd.DataFrame(columns=SUMMARY_COLS), interactive=False)
                    with gr.Tab("TRC timeline"):
                        timeline = gr.Plot()
                    with gr.Tab("Trajectories"):
                        traj = gr.Plot()

        timer = gr.Timer(0.12, active=False)
        outs = [state, frame_slider, status, summary, timeline, traj]
        run_btn.click(run_tracking, [source, seq, up_vis, up_thr, start, n_frames, trc_gate, stress,
                                     stress_off, stress_len, device], outs
                      ).then(render, [state, frame_slider], [image, info, table])
        ex_btn.click(show_example, None, outs).then(render, [state, frame_slider], [image, info, table])
        frame_slider.change(render, [state, frame_slider], [image, info, table], show_progress="hidden")
        play_btn.click(lambda: gr.Timer(active=True), None, timer)
        pause_btn.click(lambda: gr.Timer(active=False), None, timer)
        timer.tick(step, [state, frame_slider], frame_slider, show_progress="hidden")
    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description="TriNetra tracking demo UI")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    build_app(args.device).queue().launch(server_name="127.0.0.1", server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
