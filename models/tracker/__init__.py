"""
TriNetra-AMRF — Object Tracking Models
=========================================

Multi-object tracking for persistent identity association (Phase 6, Layer 5).

  - ByteTrack + TRC gate  -> byte_tracker.ByteTracker            [v1 core, default]
  - DeepSORT wrapper      -> deepsort_tracker.FusionTracker      [legacy baseline]
  - Dual-modal Re-ID / BoT-SORT hybrid                           [v2, planned]

Select with ``tracking.algorithm`` and build via ``build_tracker(cfg)``.

Self-test: python -m utils.check_bytetrack
"""

from .byte_tracker import ByteTracker  # noqa: F401
from .deepsort_tracker import FusionTracker  # noqa: F401
from .track import Track  # noqa: F401
from .trc_gate import trc_adjusted_cost  # noqa: F401


def build_tracker(cfg: dict, **kwargs):
    """Construct the tracker named by ``tracking.algorithm`` (default bytetrack)."""
    algo = str(cfg.get("tracking", {}).get("algorithm", "bytetrack")).lower()
    if algo == "bytetrack":
        return ByteTracker(cfg, **kwargs)
    if algo == "deepsort":
        return FusionTracker(cfg)
    raise NotImplementedError(
        f"tracking.algorithm={algo!r} is not implemented (bytetrack | deepsort; "
        "botsort is the planned v2)."
    )
