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
      → TemporalClassifier → video-level NSFW score

Key innovation: optical flow captures motion patterns that static detectors miss.
Mamba SSM provides O(N) temporal aggregation for long video sequences.
"""
from .model import FlowNSFW
from .ssm_backend import SSM_BACKEND

__all__ = ["FlowNSFW", "SSM_BACKEND"]
