#!/usr/bin/env python3
"""
TriNetra-AMRF — Tracking Results Report Builder (Phase 6)
=========================================================

Collects the result files of the tracking validation suite and writes
paper-ready material to ``runs/tracking_eval/report/``:

    tracking_tables.md      all tables + auto-generated findings
    tracking_tables.tex     LaTeX (booktabs) versions of the main tables
    fig_*.png / fig_*.pdf   figures
    TrackingResults.pdf     one PDF with protocol, tables and figures

Inputs (any missing section is skipped):
    runs/tracking_eval/synthetic/sev*/results.json     utils.eval_synthetic
    runs/tracking_eval/real_clips_results.json         utils.eval_real_clips
    runs/tracking_eval/stress_results.json             utils.stress_eval

Every verdict sentence is produced from the statistics (paired bootstrap 95%
CI), so the text cannot claim more than the numbers show.

Usage:
    python -m utils.build_tracking_report
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL = PROJECT_ROOT / "runs" / "tracking_eval"
OUT = EVAL / "report"
OURS = "ByteTrack + TRC gate (ours)"
TRACKERS = ["SORT-style", "DeepSORT (motion)", "ByteTrack", OURS]
COLORS = {"SORT-style": "#8c8c8c", "DeepSORT (motion)": "#c9a227", "ByteTrack": "#2b6cb0", OURS: "#c0392b"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ms(d, digits=3):
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def verdict(stat, higher_better=True):
    """Plain-language reading of a paired difference (ours - baseline)."""
    lo, hi = stat["ci95"]
    d = stat["mean_diff"]
    if lo > 0 or hi < 0:
        better = (d > 0) == higher_better
        return f"significantly {'better' if better else 'worse'}"
    return "no significant difference"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tex_table(header, rows, caption, label):
    cols = "l" + "c" * (len(header) - 1)
    esc = lambda s: str(s).replace("±", r"$\pm$").replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


class Report:
    def __init__(self):
        self.md, self.tex, self.pages, self.tables = [], [], [], []

    def table(self, title, header, rows, note=None, tex_caption=None, tex_label=None):
        self.md.append(f"### {title}\n\n{md_table(header, rows)}\n" + (f"\n*{note}*\n" if note else ""))
        if tex_caption:
            self.tex.append(tex_table(header, rows, tex_caption, tex_label))
        self.tables.append((title, header, rows, note))

    def text(self, s):
        self.md.append(s + "\n")


def save_fig(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")


# ── sections ─────────────────────────────────────────────────────────────────
def section_validation(rep):
    counts = {}
    for label, mod in (("tracker / gate / behaviour self-test", "utils.check_bytetrack"),
                       ("metric implementation self-test", "utils.check_mot_metrics")):
        try:
            r = subprocess.run([sys.executable, "-m", mod], capture_output=True, text=True, cwd=PROJECT_ROOT,
                               env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=300)
            m = re.search(r"(\d+)/(\d+) checks passed", r.stdout)
            counts[label] = f"{m.group(1)}/{m.group(2)}" if m else "n/a"
        except Exception:
            counts[label] = "n/a"
    rep.table("Implementation validation", ["Suite", "Checks passed"], [[k, v] for k, v in counts.items()],
              note="Metric tests compare against values derived by hand (e.g. a two-track ID swap gives IDF1 = 0.5, "
                   "HOTA = 0.577); tracker tests cover lifecycle, ID stability, low-confidence recovery, the TRC gate "
                   "and behaviour flags.")


def section_synthetic(rep):
    findings = []
    for sev_dir in sorted((EVAL / "synthetic").glob("sev*")):
        res = load(sev_dir / "results.json")
        if not res:
            continue
        sev = sev_dir.name.replace("sev", "")
        n = res["n_seeds"]
        cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN"]
        if "coupled" in res:
            rows = [[t] + [ms(res["coupled"]["summary"][t][c], 3 if c in ("HOTA", "DetA", "AssA", "MOTA", "IDF1") else 1)
                            for c in cols] for t in TRACKERS]
            rep.table(f"Synthetic benchmark, severity {sev} — TRC-coupled degradation ({n} sequences, exact ground truth)",
                      ["Tracker"] + cols, rows,
                      note="mean ± std over seeded sequences; identical detections and TRC for every tracker.",
                      tex_caption=f"Tracking on synthetic sequences with exact ground truth (degradation severity {sev}, "
                                  f"{n} sequences, mean $\\pm$ std).",
                      tex_label=f"tab:synth_sev{sev.replace('.', '')}")
            prow = []
            for base, key in (("ByteTrack", "paired_ours_vs_bytetrack"), ("SORT-style", "paired_ours_vs_sort"),
                              ("DeepSORT (motion)", "paired_ours_vs_deepsort")):
                for m, hb in (("HOTA", True), ("IDF1", True), ("MOTA", True), ("IDSW", False)):
                    s = res["coupled"][key][m]
                    prow.append([base, m, f"{s['mean_diff']:+.3f}", f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]",
                                 f"{s['wins']}/{s['losses']}", f"{s['wilcoxon_p']:.3g}", verdict(s, hb)])
                    if base == "ByteTrack":
                        findings.append(f"Severity {sev}: TRC gate vs ByteTrack on {m}: {verdict(s, hb)} "
                                        f"(Δ {s['mean_diff']:+.3f}, 95% CI [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]).")
            rep.table(f"Paired comparison, ours minus baseline (severity {sev}, coupled)",
                      ["Baseline", "Metric", "Mean Δ", "95% CI (bootstrap)", "Wins/Losses", "Wilcoxon p", "Reading"], prow,
                      note="Δ = ours − baseline over the same sequences; for IDSW lower is better.")
        for cond in ("uninformative", "no_degradation"):
            if cond in res:
                rows = [[t] + [ms(res["cond" if False else cond]["summary"][t][c], 3 if c != "IDSW" else 1)
                                for c in ("HOTA", "IDF1", "MOTA", "IDSW")] for t in TRACKERS]
                rep.table(f"Control: {cond} (severity {sev})", ["Tracker", "HOTA", "IDF1", "MOTA", "IDSW"], rows,
                          note="uninformative: detector degrades but the tracker receives a constant TRC. "
                               "no_degradation: TRC dips but the detector is unaffected.")
                s = res[cond]["paired_ours_vs_bytetrack"]["HOTA"]
                findings.append(f"Control '{cond}' (severity {sev}): gate vs ByteTrack HOTA {verdict(s)} "
                                f"(Δ {s['mean_diff']:+.3f}).")
        if "beta_sweep" in res:
            rows = [[b, ms(v["HOTA"]), ms(v["IDF1"]), ms(v["MOTA"]), f"{v['IDSW']['mean']:.1f}", f"{v['FN']['mean']:.0f}"]
                    for b, v in res["beta_sweep"].items()]
            rep.table(f"Gate strength β sweep (severity {sev}, coupled)", ["β", "HOTA", "IDF1", "MOTA", "IDSW", "FN"], rows,
                      note="β = 0 disables the gate (plain ByteTrack).")
            fig, ax = plt.subplots(1, 3, figsize=(9, 2.6))
            bs = [float(b) for b in res["beta_sweep"]]
            for a, m in zip(ax, ("HOTA", "IDF1", "IDSW")):
                mu = [res["beta_sweep"][b][m]["mean"] for b in res["beta_sweep"]]
                sd = [res["beta_sweep"][b][m]["std"] / np.sqrt(n) for b in res["beta_sweep"]]
                a.errorbar(range(len(bs)), mu, yerr=sd, marker="o", color=COLORS[OURS], capsize=3)
                a.set_xticks(range(len(bs))); a.set_xticklabels([f"{b:g}" for b in bs]); a.set_xlabel("gate strength β")
                a.set_title(m)
            fig.suptitle(f"β sensitivity (severity {sev}; error bars = s.e.m.)", y=1.03)
            fig.tight_layout(); save_fig(fig, f"fig_beta_sev{sev}"); rep.pages.append(fig)
        if "severity_sweep" in res:
            sev_levels = list(res["severity_sweep"])
            rows = []
            for s_ in sev_levels:
                e = res["severity_sweep"][s_]["summary"]
                rows.append([s_, f"{res['severity_sweep'][s_].get('frac_degraded_frames', float('nan')):.2f}"]
                            + [f"{e[t]['HOTA']['mean']:.3f}" for t in TRACKERS]
                            + [f"{e['ByteTrack']['IDSW']['mean']:.1f} → {e[OURS]['IDSW']['mean']:.1f}"])
            rep.table("Degradation-severity sweep (coupled; HOTA by tracker)",
                      ["Severity", "Frames with TRC<0.6"] + TRACKERS + ["IDSW ByteTrack → ours"], rows)
            fig, ax = plt.subplots(1, 2, figsize=(9, 3))
            x = [float(s_) for s_ in sev_levels]
            for t in TRACKERS:
                ax[0].errorbar(x, [res["severity_sweep"][s_]["summary"][t]["HOTA"]["mean"] for s_ in sev_levels],
                               yerr=[res["severity_sweep"][s_]["summary"][t]["HOTA"]["std"] / np.sqrt(n) for s_ in sev_levels],
                               marker="o", label=t, color=COLORS[t], capsize=2)
                ax[1].errorbar(x, [res["severity_sweep"][s_]["summary"][t]["IDSW"]["mean"] for s_ in sev_levels],
                               yerr=[res["severity_sweep"][s_]["summary"][t]["IDSW"]["std"] / np.sqrt(n) for s_ in sev_levels],
                               marker="o", label=t, color=COLORS[t], capsize=2)
            ax[0].set_ylabel("HOTA"); ax[1].set_ylabel("ID switches / sequence")
            for a in ax:
                a.set_xlabel("degradation severity")
            ax[0].legend(fontsize=7)
            fig.suptitle("Synthetic benchmark: robustness to detector degradation (error bars = s.e.m.)", y=1.03)
            fig.tight_layout(); save_fig(fig, f"fig_severity_sev{sev}"); rep.pages.append(fig)
    return findings


def section_real(rep):
    res = load(EVAL / "real_clips_results.json")
    if not res:
        return []
    clips = res["clips"]
    rows = []
    for c in clips:
        o = c["trackers"][OURS]
        fl = c["flags"].get("tracks_with_flags", {})
        rows.append([c["clip"].replace("sequence_", "seq "), c["n_frames"], f"{c['trc']['mean']:.2f}", f"{c['trc']['min']:.2f}",
                     o["n_tracks"], f"{o['len_median']:.0f}", f"{o['short_ratio']:.2f}", f"{o['coverage']:.2f}",
                     f"{o['frag_rate']:.2f}", f"{o['jump_rate']:.1f}", o["recoveries"],
                     ", ".join(f"{k}:{v}" for k, v in sorted(fl.items()) if k != "low_trc_track") or "—"])
    rep.table("Real footage: per-clip tracking quality (TriNetra tracker; label-free metrics)",
              ["Clip", "Frames", "TRC mean", "TRC min", "Tracks", "Median len", "Short (<1 s)", "Coverage",
               "Frag. rate", "Jumps/1k", "Recoveries", "Behaviour flags (tracks)"], rows,
              note="Label-free proxies (no ground truth exists for these clips). Coverage = share of confident "
                   "detections explained by a track; Frag. rate = suspected identity fragmentations per track.",
              tex_caption="Label-free tracking quality of the TriNetra tracker on the 14 real RGB-thermal clips.",
              tex_label="tab:real_clips")
    agg = res["aggregate"]
    rows = []
    for t in TRACKERS:
        a = agg[t]
        g = lambda k, d=2: f"{a[k]['mean']:.{d}f} ± {a[k]['std']:.{d}f}" if a.get(k) else "n/a"
        rows.append([t, f"{a['n_tracks']['sum']:.0f}", g("len_median", 1), g("short_ratio"), g("coverage"),
                     g("frag_rate"), g("jump_rate", 1),
                     f"{a['recoveries']['sum']:.0f}" if a.get("recoveries") else "n/a",
                     f"{a['tracker_ms_per_frame']['mean']:.2f}" if a.get("tracker_ms_per_frame") else "n/a"])
    rep.table("Real footage: tracker comparison across all clips (mean ± std over clips)",
              ["Tracker", "Total tracks", "Median len", "Short ratio", "Coverage", "Frag. rate", "Jumps/1k",
               "Recoveries", "ms/frame"], rows,
              note="Same cached detections for every tracker. Fewer tracks / shorter-track ratio / fragmentation "
                   "at equal coverage indicate more consolidated identities; none of these prove correctness.",
              tex_caption="Tracker comparison on real footage (label-free metrics, mean $\\pm$ std over 14 clips).",
              tex_label="tab:real_compare")
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    names = [c["clip"].replace("sequence_", "") for c in clips]
    for a, (k, ttl) in zip(ax, (("n_tracks", "confirmed tracks"), ("len_median", "median track length (frames)"),
                                ("coverage", "coverage of confident detections"))):
        w = 0.2
        for i, t in enumerate(TRACKERS):
            a.bar(np.arange(len(clips)) + (i - 1.5) * w, [c["trackers"][t][k] for c in clips], w,
                  label=t, color=COLORS[t])
        a.set_xticks(range(len(clips))); a.set_xticklabels(names, fontsize=6); a.set_title(ttl, fontsize=8)
    ax[0].legend(fontsize=6)
    fig.tight_layout(); save_fig(fig, "fig_real_clips"); rep.pages.append(fig)

    o = agg[OURS]
    b = agg["ByteTrack"]
    return [f"Real clips ({len(clips)}): the TriNetra tracker produced {o['n_tracks']['sum']:.0f} tracks "
            f"(median length {o['len_median']['mean']:.0f} frames, coverage {o['coverage']['mean']:.2f}); "
            f"ByteTrack without the gate produced {b['n_tracks']['sum']:.0f}."]


def section_stress(rep):
    res = load(EVAL / "stress_results.json")
    if not res:
        return []
    runs = res["runs"]
    bins = {}
    for r in runs:
        for k, v in r["coupling"].items():
            bins.setdefault(k, []).append(v)
    rows = []
    for k in sorted(bins):
        vs = bins[k]
        g = lambda key: np.mean([v[key] for v in vs if v.get(key) is not None]) if any(v.get(key) is not None for v in vs) else float("nan")
        rows.append([k, sum(v["frames"] for v in vs), f"{g('recall_vs_clean'):.2f}", f"{g('mean_iou_of_matched'):.2f}",
                     f"{g('score_ratio_of_matched'):.2f}", f"{g('dets_per_frame_ratio'):.2f}"])
    rep.table("Real-footage coupling: detector behaviour vs frame TRC (thermal damaged, RGB intact)",
              ["TRC bin", "Frames", "Recall vs clean", "Mean IoU (matched)", "Score ratio", "Detections/frame ratio"], rows,
              note="Reference = confident detections (score ≥ 0.6) of the clean run. This measures whether low TRC "
                   "actually goes with worse detections — the assumption behind the gate.",
              tex_caption="Detector behaviour on real footage as a function of frame TRC under thermal degradation.",
              tex_label="tab:coupling")

    S = res["summary"]
    rows = [[t] + [ms(S[t][m], 3 if m != "IDSW" else 1) for m in ("IDF1", "HOTA", "AssA", "IDSW")] for t in TRACKERS]
    rep.table(f"Identity stability under thermal degradation ({len(runs)} runs, degraded segment only)",
              ["Tracker", "IDF1", "HOTA", "AssA", "IDSW"], rows,
              note="Each tracker's degraded-run tracks are scored against its own clean-run tracks (self-consistency; "
                   "the reference is a tracker output, not human annotation).",
              tex_caption="Identity stability of each tracker when the thermal stream is degraded (real footage).",
              tex_label="tab:stress")
    findings = []
    prow = []
    for m, hb in (("IDF1", True), ("HOTA", True), ("AssA", True), ("IDSW", False)):
        s = res["paired_ours_vs_bytetrack"][m]
        prow.append([m, f"{s['mean_diff']:+.3f}", f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]",
                     f"{s['wins']}/{s['losses']}", verdict(s, hb)])
        findings.append(f"Real stress test: gate vs ByteTrack identity stability on {m}: {verdict(s, hb)} "
                        f"(Δ {s['mean_diff']:+.3f}, n = {s['n']}).")
    rep.table("Stress test: ours minus ByteTrack (paired over runs)",
              ["Metric", "Mean Δ", "95% CI (bootstrap)", "Wins/Losses", "Reading"], prow)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    mids = sorted(bins)
    for key, ttl, a in (("recall_vs_clean", "recall vs clean detections", ax[0]),
                        ("score_ratio_of_matched", "score ratio (stressed / clean)", ax[0])):
        pass
    for key, lab in (("recall_vs_clean", "recall vs clean"), ("score_ratio_of_matched", "score ratio"),
                     ("mean_iou_of_matched", "mean IoU")):
        ax[0].plot(range(len(mids)), [np.nanmean([v[key] if v.get(key) is not None else np.nan for v in bins[k]]) for k in mids],
                   marker="o", label=lab)
    ax[0].set_xticks(range(len(mids))); ax[0].set_xticklabels(mids, fontsize=7)
    ax[0].set_xlabel("frame TRC bin"); ax[0].set_title("detector vs TRC (real footage)"); ax[0].legend(fontsize=7)
    w = 0.2
    for i, t in enumerate(TRACKERS):
        ax[1].bar(np.arange(2) + (i - 1.5) * w, [S[t]["IDF1"]["mean"], S[t]["HOTA"]["mean"]], w,
                  yerr=[S[t]["IDF1"]["std"] / np.sqrt(len(runs)), S[t]["HOTA"]["std"] / np.sqrt(len(runs))],
                  color=COLORS[t], label=t, capsize=2)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["IDF1", "HOTA"]); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("identity stability vs own clean run"); ax[1].legend(fontsize=6)
    fig.tight_layout(); save_fig(fig, "fig_stress"); rep.pages.append(fig)
    return findings


# ── PDF ──────────────────────────────────────────────────────────────────────
def table_page(title, header, rows, note):
    fig = plt.figure(figsize=(11, 3.2 + 0.28 * len(rows)))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.78]); ax.axis("off")
    fig.text(0.02, 0.93, title, fontsize=10, fontweight="bold", va="top", wrap=True)
    tb = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=header, loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.35)
    tb.auto_set_column_width(list(range(len(header))))
    for (r, c), cell in tb.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#e8e8e8")
    if note:
        fig.text(0.02, 0.04, note, fontsize=7, style="italic", wrap=True)
    return fig


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = Report()
    rep.text("# TriNetra Phase 6 — Tracking Validation Results\n")
    rep.text("**Protocol.** Four layers of evidence, each labelled for what it can and cannot show: (1) unit-tested "
             "metric and tracker implementations; (2) a synthetic benchmark with exact ground truth, whose detector "
             "simulation couples quality to TRC (with controls where the gate has no reason to help); (3) label-free "
             "quality metrics on the 14 real RGB-thermal clips (no track annotations exist); (4) a thermal-degradation "
             "stress test on real footage, measuring both the TRC–detector coupling and identity stability. "
             "No ground-truth accuracy (MOTA/IDF1/HOTA against human labels) is reported for the real clips.\n")
    section_validation(rep)
    findings = section_synthetic(rep) + section_real(rep) + section_stress(rep)
    if findings:
        rep.md.insert(2, "## Findings (generated from the statistics)\n\n" + "\n".join(f"- {f}" for f in findings) + "\n")

    (OUT / "tracking_tables.md").write_text("\n".join(rep.md), encoding="utf-8")
    if rep.tex:
        (OUT / "tracking_tables.tex").write_text("\n".join(rep.tex), encoding="utf-8")

    with PdfPages(OUT / "TrackingResults.pdf") as pdf:
        fig = plt.figure(figsize=(11, 4)); fig.patch.set_facecolor("white")
        fig.text(0.04, 0.9, "TriNetra-AMRF — Phase 6: Tracking Validation Results", fontsize=15, fontweight="bold", va="top")
        body = re.sub(r"\*\*|`", "", rep.md[1]) + "\n\nFindings:\n" + "\n".join(f"  • {f}" for f in findings)
        import textwrap
        fig.text(0.04, 0.8, "\n".join(textwrap.fill(p, 150) if p.strip() else "" for p in body.split("\n")),
                 fontsize=8, va="top")
        pdf.savefig(fig); plt.close(fig)
        for title, header, rows, note in rep.tables:
            f = table_page(title, header, rows, note); pdf.savefig(f); plt.close(f)
        for f in rep.pages:
            pdf.savefig(f); plt.close(f)
    print("report written to", OUT)
    for f in findings:
        print(" -", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
