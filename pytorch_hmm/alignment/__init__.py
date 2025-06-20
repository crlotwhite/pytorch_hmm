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
    dtw_distance,
    ConstrainedDTWAligner,
    phoneme_audio_alignment,
    extract_phoneme_durations
)

from .ctc import (
    CTCAligner,
    CTCSegmentationAligner,
    ctc_alignment_path,
    expand_targets_with_blank,
    ctc_forward_algorithm,
    ctc_backward_algorithm,
    remove_ctc_blanks,
    collapse_repeated_tokens,
    ctc_decode_sequence
)

__all__ = [
    # DTW alignment
    "DTWAligner",
    "dtw_alignment", 
    "compute_dtw_path",
    "dtw_distance",
    "ConstrainedDTWAligner",
    "phoneme_audio_alignment",
    "extract_phoneme_durations",
    
    # CTC alignment
    "CTCAligner",
    "CTCSegmentationAligner", 
    "ctc_alignment_path",
    "expand_targets_with_blank",
    "ctc_forward_algorithm",
    "ctc_backward_algorithm",
    "remove_ctc_blanks",
    "collapse_repeated_tokens",
    "ctc_decode_sequence"
]
