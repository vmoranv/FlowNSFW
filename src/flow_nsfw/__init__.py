"""FlowNSFW — Optical-flow-guided video NSFW detection.

Architecture:
    frames (B,T,3,H,W)
      → UNetEncoder → bottleneck + 3 skips
      → FlowNet/RaftFlowNet → flow_fwd, flow_bwd  (optimized correlation)
      → SparseGlobalTemporal → feat_t (temporally aggregated)
          backend="attention"  : standard Transformer (O(N²))
          backend="mamba"      : SSM via mamba-ssm CUDA kernels (O(N))
          backend="hybrid"     : attention(local) + SSM(global)
      → DetectionHead → multi-scale boxes + cls scores
          sparse=True : foreground-gated sparse window detection
      → decode_boxes → decoded [cx,cy,w,h, obj, cls] per scale
      → VideoClassifier → video-level NSFW score (with optional motion gate)

4K support: MotionRouter (motion_router.py) does cheap frame-diff salience →
motion bbox → RGB patch crops → resized to model resolution. Global downsample
fallback catches static NSFW. All patches stay RGB; motion only selects WHERE.

Key innovation: optical flow captures motion patterns that static detectors miss.
Mamba SSM provides O(N) temporal aggregation for long video sequences.
"""
from .model import FlowNSFW
from .ssm_backend import SSM_BACKEND
from .motion_router import MotionRouter, frame_diff_input, motion_salience

__all__ = ["FlowNSFW", "SSM_BACKEND", "MotionRouter",
           "frame_diff_input", "motion_salience"]
