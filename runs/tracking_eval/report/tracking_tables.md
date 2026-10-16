# TriNetra Phase 6 — Tracking Validation Results


**Protocol.** Four layers of evidence, each labelled for what it can and cannot show: (1) unit-tested metric and tracker implementations; (2) a synthetic benchmark with exact ground truth, whose detector simulation couples quality to TRC (with controls where the gate has no reason to help); (3) label-free quality metrics on the 14 real RGB-thermal clips (no track annotations exist); (4) a thermal-degradation stress test on real footage, measuring both the TRC–detector coupling and identity stability. No ground-truth accuracy (MOTA/IDF1/HOTA against human labels) is reported for the real clips.


## Findings (generated from the statistics)

- Severity 0.5: TRC gate vs ByteTrack on HOTA: no significant difference (Δ -0.000, 95% CI [-0.005, +0.005]).
- Severity 0.5: TRC gate vs ByteTrack on IDF1: no significant difference (Δ -0.002, 95% CI [-0.011, +0.006]).
- Severity 0.5: TRC gate vs ByteTrack on MOTA: significantly worse (Δ -0.003, 95% CI [-0.004, -0.002]).
- Severity 0.5: TRC gate vs ByteTrack on IDSW: no significant difference (Δ +0.167, 95% CI [-0.433, +0.733]).
- Control 'uninformative' (severity 0.5): gate vs ByteTrack HOTA significantly worse (Δ -0.001).
- Control 'no_degradation' (severity 0.5): gate vs ByteTrack HOTA significantly worse (Δ -0.000).
- Severity 1.0: TRC gate vs ByteTrack on HOTA: significantly worse (Δ -0.003, 95% CI [-0.005, -0.000]).
- Severity 1.0: TRC gate vs ByteTrack on IDF1: no significant difference (Δ -0.004, 95% CI [-0.009, +0.000]).
- Severity 1.0: TRC gate vs ByteTrack on MOTA: significantly worse (Δ -0.003, 95% CI [-0.005, -0.002]).
- Severity 1.0: TRC gate vs ByteTrack on IDSW: no significant difference (Δ +0.033, 95% CI [-0.433, +0.467]).
- Control 'uninformative' (severity 1.0): gate vs ByteTrack HOTA significantly worse (Δ -0.001).
- Control 'no_degradation' (severity 1.0): gate vs ByteTrack HOTA no significant difference (Δ -0.001).
- Real clips (8): the TriNetra tracker produced 296 tracks (median length 61 frames, coverage 0.97); ByteTrack without the gate produced 299.
- Real stress test: gate vs ByteTrack identity stability on IDF1: no significant difference (Δ +0.003, n = 6).
- Real stress test: gate vs ByteTrack identity stability on HOTA: no significant difference (Δ +0.007, n = 6).
- Real stress test: gate vs ByteTrack identity stability on AssA: no significant difference (Δ +0.013, n = 6).
- Real stress test: gate vs ByteTrack identity stability on IDSW: no significant difference (Δ +0.000, n = 6).

### Implementation validation

| Suite | Checks passed |
|---|---|
| tracker / gate / behaviour self-test | 28/28 |
| metric implementation self-test | 17/17 |

*Metric tests compare against values derived by hand (e.g. a two-track ID swap gives IDF1 = 0.5, HOTA = 0.577); tracker tests cover lifecycle, ID stability, low-confidence recovery, the TRC gate and behaviour flags.*

### Synthetic benchmark, severity 0.5 — TRC-coupled degradation (30 sequences, exact ground truth)

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---|---|---|---|---|---|---|
| SORT-style | 0.556 ± 0.056 | 0.602 ± 0.036 | 0.515 ± 0.079 | 0.745 ± 0.041 | 0.752 ± 0.075 | 12.8 ± 9.5 | 17.2 ± 10.1 | 769.2 ± 202.6 |
| DeepSORT (motion) | 0.514 ± 0.052 | 0.576 ± 0.032 | 0.461 ± 0.075 | 0.702 ± 0.042 | 0.697 ± 0.072 | 15.9 ± 10.4 | 155.3 ± 53.0 | 755.4 ± 184.9 |
| ByteTrack | 0.559 ± 0.057 | 0.608 ± 0.034 | 0.516 ± 0.084 | 0.754 ± 0.038 | 0.754 ± 0.078 | 12.4 ± 9.3 | 17.6 ± 10.5 | 739.3 ± 190.6 |
| ByteTrack + TRC gate (ours) | 0.559 ± 0.056 | 0.607 ± 0.035 | 0.516 ± 0.080 | 0.751 ± 0.039 | 0.752 ± 0.076 | 12.5 ± 9.4 | 17.0 ± 10.1 | 748.9 ± 195.5 |

*mean ± std over seeded sequences; identical detections and TRC for every tracker.*

### Paired comparison, ours minus baseline (severity 0.5, coupled)

| Baseline | Metric | Mean Δ | 95% CI (bootstrap) | Wins/Losses | Wilcoxon p | Reading |
|---|---|---|---|---|---|---|
| ByteTrack | HOTA | -0.000 | [-0.005, +0.005] | 4/25 | 0.00446 | no significant difference |
| ByteTrack | IDF1 | -0.002 | [-0.011, +0.006] | 5/24 | 0.00583 | no significant difference |
| ByteTrack | MOTA | -0.003 | [-0.004, -0.002] | 1/28 | 5.9e-06 | significantly worse |
| ByteTrack | IDSW | +0.167 | [-0.433, +0.733] | 5/3 | 0.574 | no significant difference |
| SORT-style | HOTA | +0.003 | [-0.000, +0.004] | 29/1 | 3.79e-06 | no significant difference |
| SORT-style | IDF1 | +0.000 | [-0.005, +0.003] | 29/1 | 3.79e-06 | no significant difference |
| SORT-style | MOTA | +0.007 | [+0.006, +0.008] | 30/0 | 1.86e-09 | significantly better |
| SORT-style | IDSW | -0.233 | [-0.600, +0.100] | 2/5 | 0.23 | no significant difference |
| DeepSORT (motion) | HOTA | +0.044 | [+0.029, +0.059] | 26/4 | 1.82e-05 | significantly better |
| DeepSORT (motion) | IDF1 | +0.055 | [+0.030, +0.078] | 25/5 | 0.000153 | significantly better |
| DeepSORT (motion) | MOTA | +0.049 | [+0.042, +0.056] | 30/0 | 1.86e-09 | significantly better |
| DeepSORT (motion) | IDSW | -3.400 | [-7.267, +0.767] | 7/23 | 0.0248 | no significant difference |

*Δ = ours − baseline over the same sequences; for IDSW lower is better.*

### Control: uninformative (severity 0.5)

| Tracker | HOTA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|
| SORT-style | 0.553 ± 0.059 | 0.743 ± 0.081 | 0.747 ± 0.041 | 15.4 ± 8.4 |
| DeepSORT (motion) | 0.526 ± 0.050 | 0.713 ± 0.080 | 0.709 ± 0.044 | 17.0 ± 9.2 |
| ByteTrack | 0.560 ± 0.055 | 0.751 ± 0.075 | 0.757 ± 0.039 | 15.3 ± 8.2 |
| ByteTrack + TRC gate (ours) | 0.559 ± 0.055 | 0.750 ± 0.075 | 0.755 ± 0.039 | 15.3 ± 8.3 |

*uninformative: detector degrades but the tracker receives a constant TRC. no_degradation: TRC dips but the detector is unaffected.*

### Control: no_degradation (severity 0.5)

| Tracker | HOTA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|
| SORT-style | 0.650 ± 0.046 | 0.833 ± 0.067 | 0.826 ± 0.026 | 6.9 ± 5.0 |
| DeepSORT (motion) | 0.601 ± 0.048 | 0.774 ± 0.066 | 0.789 ± 0.031 | 11.8 ± 12.8 |
| ByteTrack | 0.652 ± 0.045 | 0.834 ± 0.067 | 0.828 ± 0.026 | 6.9 ± 5.0 |
| ByteTrack + TRC gate (ours) | 0.651 ± 0.045 | 0.834 ± 0.067 | 0.828 ± 0.026 | 6.9 ± 5.0 |

*uninformative: detector degrades but the tracker receives a constant TRC. no_degradation: TRC dips but the detector is unaffected.*

### Gate strength β sweep (severity 0.5, coupled)

| β | HOTA | IDF1 | MOTA | IDSW | FN |
|---|---|---|---|---|---|
| 0.0 | 0.559 ± 0.057 | 0.754 ± 0.078 | 0.754 ± 0.038 | 12.4 | 739 |
| 0.25 | 0.559 ± 0.059 | 0.753 ± 0.080 | 0.753 ± 0.039 | 12.6 | 744 |
| 0.5 | 0.559 ± 0.056 | 0.752 ± 0.076 | 0.751 ± 0.039 | 12.5 | 749 |
| 1.0 | 0.559 ± 0.056 | 0.754 ± 0.076 | 0.749 ± 0.040 | 12.7 | 755 |
| 2.0 | 0.558 ± 0.056 | 0.753 ± 0.076 | 0.748 ± 0.041 | 12.7 | 760 |
| 4.0 | 0.557 ± 0.056 | 0.752 ± 0.076 | 0.746 ± 0.041 | 12.7 | 765 |

*β = 0 disables the gate (plain ByteTrack).*

### Degradation-severity sweep (coupled; HOTA by tracker)

| Severity | Frames with TRC<0.6 | SORT-style | DeepSORT (motion) | ByteTrack | ByteTrack + TRC gate (ours) | IDSW ByteTrack → ours |
|---|---|---|---|---|---|---|
| 0.0 | 0.30 | 0.758 | 0.703 | 0.758 | 0.758 | 4.3 → 4.3 |
| 0.25 | 0.30 | 0.650 | 0.606 | 0.651 | 0.651 | 9.8 → 9.5 |
| 0.5 | 0.30 | 0.556 | 0.514 | 0.559 | 0.559 | 12.4 → 12.5 |
| 0.75 | 0.30 | 0.407 | 0.428 | 0.418 | 0.413 | 20.5 → 20.8 |
| 1.0 | 0.30 | 0.319 | 0.359 | 0.326 | 0.323 | 17.6 → 17.7 |
| 1.5 | 0.30 | 0.230 | 0.232 | 0.236 | 0.235 | 21.7 → 21.8 |

### Synthetic benchmark, severity 1.0 — TRC-coupled degradation (30 sequences, exact ground truth)

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---|---|---|---|---|---|---|
| SORT-style | 0.319 ± 0.082 | 0.380 ± 0.077 | 0.272 ± 0.092 | 0.471 ± 0.101 | 0.467 ± 0.119 | 17.8 ± 11.6 | 25.3 ± 13.2 | 1611.8 ± 446.3 |
| DeepSORT (motion) | 0.359 ± 0.059 | 0.412 ± 0.056 | 0.317 ± 0.075 | 0.456 ± 0.095 | 0.525 ± 0.092 | 27.0 ± 18.7 | 357.5 ± 94.1 | 1309.6 ± 334.1 |
| ByteTrack | 0.326 ± 0.083 | 0.388 ± 0.076 | 0.278 ± 0.094 | 0.480 ± 0.101 | 0.476 ± 0.121 | 17.6 ± 11.5 | 32.5 ± 18.7 | 1576.4 ± 434.4 |
| ByteTrack + TRC gate (ours) | 0.323 ± 0.082 | 0.385 ± 0.077 | 0.275 ± 0.092 | 0.476 ± 0.102 | 0.471 ± 0.119 | 17.7 ± 11.5 | 30.1 ± 17.8 | 1590.0 ± 440.6 |

*mean ± std over seeded sequences; identical detections and TRC for every tracker.*

### Paired comparison, ours minus baseline (severity 1.0, coupled)

| Baseline | Metric | Mean Δ | 95% CI (bootstrap) | Wins/Losses | Wilcoxon p | Reading |
|---|---|---|---|---|---|---|
| ByteTrack | HOTA | -0.003 | [-0.005, -0.000] | 4/26 | 0.00104 | significantly worse |
| ByteTrack | IDF1 | -0.004 | [-0.009, +0.000] | 8/22 | 0.0106 | no significant difference |
| ByteTrack | MOTA | -0.003 | [-0.005, -0.002] | 3/27 | 1.42e-06 | significantly worse |
| ByteTrack | IDSW | +0.033 | [-0.433, +0.467] | 9/9 | 0.771 | no significant difference |
| SORT-style | HOTA | +0.004 | [+0.002, +0.006] | 28/2 | 1.6e-05 | significantly better |
| SORT-style | IDF1 | +0.005 | [+0.001, +0.008] | 26/4 | 0.000209 | significantly better |
| SORT-style | MOTA | +0.005 | [+0.003, +0.007] | 27/2 | 0.000114 | significantly better |
| SORT-style | IDSW | -0.167 | [-0.767, +0.400] | 8/8 | 0.548 | no significant difference |
| DeepSORT (motion) | HOTA | -0.037 | [-0.054, -0.018] | 6/24 | 0.00113 | significantly worse |
| DeepSORT (motion) | IDF1 | -0.054 | [-0.087, -0.019] | 7/23 | 0.00501 | significantly worse |
| DeepSORT (motion) | MOTA | +0.021 | [+0.007, +0.034] | 20/10 | 0.00664 | significantly better |
| DeepSORT (motion) | IDSW | -9.367 | [-13.467, -5.600] | 4/22 | 0.000241 | significantly better |

*Δ = ours − baseline over the same sequences; for IDSW lower is better.*

### Control: uninformative (severity 1.0)

| Tracker | HOTA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|
| SORT-style | 0.321 ± 0.068 | 0.475 ± 0.098 | 0.470 ± 0.088 | 18.8 ± 7.9 |
| DeepSORT (motion) | 0.352 ± 0.048 | 0.514 ± 0.067 | 0.452 ± 0.091 | 30.3 ± 16.1 |
| ByteTrack | 0.328 ± 0.068 | 0.481 ± 0.099 | 0.478 ± 0.088 | 19.2 ± 8.5 |
| ByteTrack + TRC gate (ours) | 0.327 ± 0.067 | 0.479 ± 0.098 | 0.477 ± 0.088 | 19.0 ± 8.6 |

*uninformative: detector degrades but the tracker receives a constant TRC. no_degradation: TRC dips but the detector is unaffected.*

### Control: no_degradation (severity 1.0)

| Tracker | HOTA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|
| SORT-style | 0.548 ± 0.036 | 0.761 ± 0.059 | 0.744 ± 0.028 | 10.2 ± 9.0 |
| DeepSORT (motion) | 0.506 ± 0.036 | 0.707 ± 0.066 | 0.695 ± 0.030 | 14.4 ± 12.7 |
| ByteTrack | 0.552 ± 0.037 | 0.765 ± 0.060 | 0.752 ± 0.026 | 10.3 ± 10.0 |
| ByteTrack + TRC gate (ours) | 0.552 ± 0.036 | 0.765 ± 0.059 | 0.751 ± 0.026 | 10.2 ± 9.1 |

*uninformative: detector degrades but the tracker receives a constant TRC. no_degradation: TRC dips but the detector is unaffected.*

### Gate strength β sweep (severity 1.0, coupled)

| β | HOTA | IDF1 | MOTA | IDSW | FN |
|---|---|---|---|---|---|
| 0.0 | 0.326 ± 0.083 | 0.476 ± 0.121 | 0.480 ± 0.101 | 17.6 | 1576 |
| 0.25 | 0.324 ± 0.082 | 0.472 ± 0.120 | 0.478 ± 0.101 | 17.6 | 1585 |
| 0.5 | 0.323 ± 0.082 | 0.471 ± 0.119 | 0.476 ± 0.102 | 17.7 | 1590 |
| 1.0 | 0.321 ± 0.082 | 0.470 ± 0.119 | 0.474 ± 0.102 | 17.8 | 1596 |
| 2.0 | 0.321 ± 0.081 | 0.468 ± 0.118 | 0.473 ± 0.102 | 17.8 | 1600 |
| 4.0 | 0.320 ± 0.082 | 0.467 ± 0.119 | 0.472 ± 0.102 | 17.7 | 1605 |

*β = 0 disables the gate (plain ByteTrack).*

### Real footage: per-clip tracking quality (TriNetra tracker; label-free metrics)

| Clip | Frames | TRC mean | TRC min | Tracks | Median len | Short (<1 s) | Coverage | Frag. rate | Jumps/1k | Recoveries | Behaviour flags (tracks) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seq 01 | 1500 | 0.78 | 0.78 | 50 | 56 | 0.12 | 0.98 | 0.08 | 0.1 | 22 | — |
| seq 02 | 1500 | 0.76 | 0.75 | 29 | 76 | 0.10 | 0.97 | 0.03 | 0.0 | 30 | — |
| seq 03 | 1500 | 0.78 | 0.78 | 15 | 101 | 0.13 | 0.97 | 0.13 | 0.0 | 17 | — |
| seq 04 | 1500 | 0.78 | 0.77 | 12 | 22 | 0.58 | 0.98 | 0.42 | 0.0 | 2 | — |
| seq 05 | 1500 | 0.78 | 0.78 | 28 | 26 | 0.39 | 0.98 | 0.29 | 0.2 | 19 | — |
| seq 06 | 1500 | 0.75 | 0.72 | 56 | 46 | 0.39 | 0.95 | 0.09 | 0.0 | 71 | — |
| seq 07 | 1500 | 0.72 | 0.68 | 66 | 64 | 0.21 | 0.96 | 0.30 | 0.3 | 94 | — |
| seq 08 | 1500 | 0.75 | 0.73 | 40 | 95 | 0.15 | 0.98 | 0.07 | 0.0 | 40 | — |

*Label-free proxies (no ground truth exists for these clips). Coverage = share of confident detections explained by a track; Frag. rate = suspected identity fragmentations per track.*

### Real footage: tracker comparison across all clips (mean ± std over clips)

| Tracker | Total tracks | Median len | Short ratio | Coverage | Frag. rate | Jumps/1k | Recoveries | ms/frame |
|---|---|---|---|---|---|---|---|---|
| SORT-style | 319 | 44.5 ± 19.5 | 0.40 ± 0.10 | 0.97 ± 0.01 | 0.19 ± 0.09 | 0.2 ± 0.2 | 1717 | 0.38 |
| DeepSORT (motion) | 509 | 28.0 ± 11.3 | 0.45 ± 0.13 | 0.98 ± 0.01 | 0.32 ± 0.10 | 0.1 ± 0.2 | n/a | n/a |
| ByteTrack | 299 | 62.1 ± 29.7 | 0.27 ± 0.18 | 0.97 ± 0.01 | 0.19 ± 0.12 | 0.1 ± 0.1 | 232 | 0.43 |
| ByteTrack + TRC gate (ours) | 296 | 60.9 ± 29.2 | 0.26 ± 0.18 | 0.97 ± 0.01 | 0.18 ± 0.14 | 0.1 ± 0.1 | 295 | 0.43 |

*Same cached detections for every tracker. Fewer tracks / shorter-track ratio / fragmentation at equal coverage indicate more consolidated identities; none of these prove correctness.*

### Real-footage coupling: detector behaviour vs frame TRC (thermal damaged, RGB intact)

| TRC bin | Frames | Recall vs clean | Mean IoU (matched) | Score ratio | Detections/frame ratio |
|---|---|---|---|---|---|
| 0.40-0.60 | 300 | 0.89 | 0.89 | 0.80 | 0.74 |
| 0.60-0.75 | 1500 | 0.97 | 0.90 | 0.93 | 0.88 |

*Reference = confident detections (score ≥ 0.6) of the clean run. This measures whether low TRC actually goes with worse detections — the assumption behind the gate.*

### Identity stability under thermal degradation (6 runs, degraded segment only)

| Tracker | IDF1 | HOTA | AssA | IDSW |
|---|---|---|---|---|
| SORT-style | 0.733 ± 0.124 | 0.630 ± 0.122 | 0.654 ± 0.178 | 5.2 ± 2.4 |
| DeepSORT (motion) | 0.658 ± 0.083 | 0.592 ± 0.071 | 0.538 ± 0.120 | 38.0 ± 22.7 |
| ByteTrack | 0.790 ± 0.085 | 0.695 ± 0.090 | 0.696 ± 0.151 | 6.2 ± 4.1 |
| ByteTrack + TRC gate (ours) | 0.793 ± 0.082 | 0.702 ± 0.071 | 0.709 ± 0.131 | 6.2 ± 3.9 |

*Each tracker's degraded-run tracks are scored against its own clean-run tracks (self-consistency; the reference is a tracker output, not human annotation).*

### Stress test: ours minus ByteTrack (paired over runs)

| Metric | Mean Δ | 95% CI (bootstrap) | Wins/Losses | Reading |
|---|---|---|---|---|
| IDF1 | +0.003 | [-0.041, +0.045] | 3/3 | no significant difference |
| HOTA | +0.007 | [-0.020, +0.042] | 3/3 | no significant difference |
| AssA | +0.013 | [-0.033, +0.070] | 2/4 | no significant difference |
| IDSW | +0.000 | [-1.000, +1.000] | 2/2 | no significant difference |
