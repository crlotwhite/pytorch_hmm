"""
Advanced alignment algorithms for HMM-based sequence modeling.

This module provides various alignment techniques commonly used in
speech processing and sequence modeling:

- DTW: Dynamic Time Warping for flexible temporal alignment
- CTC: Connectionist Temporal Classification alignment
- Attention: Neural attention-based alignment
- Monotonic: Monotonic attention alignment
"""

from .dtw import (
    DTWAligner,
    dtw_alignment,
    compute_dtw_path,
    dtw_distance
)

__all__ = [
    "DTWAligner",
    "dtw_alignment", 
    "compute_dtw_path",
    "dtw_distance"
]
