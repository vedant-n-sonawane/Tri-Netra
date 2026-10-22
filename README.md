# TriNetra-AMRF — Dual-Modality Surveillance Framework

> **Tri**-spectral **Ne**ural **Tra**cking with **A**daptive **M**ulti-modal **R**easoning **F**usion

TriNetra is a dual-modality (RGB + Thermal) surveillance framework built on a 7-layer architecture:

1. **Data Acquisition** — Synchronized capture from visible and thermal sensors
2. **Preprocessing & Reliability (TRC)** — Alignment, denoising, normalization with Thermal Reliability Confidence scoring
3. **Dynamic Cross-Modal Fusion** — Adaptive fusion of RGB and thermal feature maps
4. **Multi-Scale Object Detection** — YOLO-based detection across multiple spatial scales
5. **Tracking & Behaviour Analysis** — ByteTrack/DeepSORT tracking with behavioural anomaly detection
6. **Hierarchical Threat Reasoning** — Multi-level risk assessment and threat classification
7. **Explainable Alert Generation** — Human-interpretable alerts with visual explanations

---

## Project Structure

```
TriNetra/
├── datasets/              # Raw and processed datasets
│   ├── visible/           # RGB imagery
│   ├── thermal/           # Thermal/IR imagery
│   ├── annotations/       # COCO-format labels
│   └── calibration/       # Camera calibration matrices
├── preprocessing/         # Data preprocessing pipeline
│   ├── alignment.py       # Cross-modal spatial alignment
│   ├── denoise.py         # Noise reduction filters
│   └── normalize.py       # Intensity/contrast normalization
├── models/                # Model architectures
│   ├── detector/          # Object detection heads
│   ├── fusion/            # Cross-modal fusion modules
│   ├── tracker/           # Tracking algorithms
│   └── risk_model/        # Threat reasoning networks
├── training/              # Training scripts
│   ├── train_detector.py  # Detection model training
│   ├── train_fusion.py    # Fusion module training
│   └── train_risk.py      # Risk model training
├── inference/             # Inference pipelines
│   ├── realtime.py        # Live camera inference
│   └── video.py           # Offline video processing
├── utils/                 # Utility scripts
│   └── check_gpu.py       # GPU/CUDA diagnostics
├── configs/               # Configuration files
│   └── default.yaml       # Default paths & hyperparameters
├── weights/               # Pretrained & trained weights
├── app/                   # FastAPI serving application
├── environment.yml        # Conda environment spec
└── requirements.txt       # Pip dependencies
```

---

## Quick Start

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate trinetra
```

Or install via pip:

```bash
pip install -r requirements.txt
```

### 2. Verify GPU Availability

```bash
python utils/check_gpu.py
```

### 3. Configure Paths

Edit `configs/default.yaml` to point to your dataset and weight directories.

### 4. Download the Trained Model

The trained fusion + TRC detector is too large for the repository (164 MB), so it is published as a release asset:

**[trinetra_fusion_trc_v1.pt](https://github.com/vedant-n-sonawane/Tri-Netra/releases/download/v1.0/trinetra_fusion_trc_v1.pt)** (release [v1.0](https://github.com/vedant-n-sonawane/Tri-Netra/releases/tag/v1.0)): M3FD test split, 630 pairs, mAP@50 **0.700**, mAP@50:95 **0.478**.

SHA-256: `e7fac200784a5518c5b859effdadac8764484cc1bd70823ee9489f4d9a6bb124`

```bash
python -m training.evaluate --checkpoint weights/fusion/trinetra_fusion_trc_v1.pt --split test
```

---

## Phase Roadmap

| Phase | Description                              | Status       |
|-------|------------------------------------------|--------------|
| 1     | Project scaffold & environment setup     | ✅ Complete   |
| 2     | Data acquisition & dataset preparation   | ✅ Complete   |
| 3     | Preprocessing pipeline (TRC)             | ✅ Complete   |
| 4     | Cross-modal fusion module                | ✅ Complete   |
| 5     | Detection integration & training         | ✅ Complete (trained model: release v1.0) |
| 6     | Tracking & behaviour analysis            | ✅ Complete (ByteTrack + TRC gate, ReID, behaviour flags) |
| 7     | Hierarchical threat reasoning            | ✅ Complete (rule-based levels 1–5 with reasons: `models/risk_model/threat_scorer.py`) |
| 8     | Explainable alert generation             | ✅ Complete (threat, sensor and behaviour alerts: `models/risk_model/alerts.py`; tactical web console: `app/tracking_ui.py`) |
| 9     | Real-time inference & deployment         | 🔲 Planned   |

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA ≥ 11.8 (RTX 3060+ recommended)
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 50 GB+ for datasets and weights

---

## License

Proprietary: all rights reserved by the TriNetra project team. The trained model published as release v1.0 is provided for research and educational use only.
