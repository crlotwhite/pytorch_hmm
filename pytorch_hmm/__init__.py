"""PyTorch implementation of Hidden Markov Model.

This library provides efficient implementations of HMM algorithms
optimized for speech synthesis and other sequence modeling tasks.

Main components:
- HMM: Base class with model parameters
- HMMPyTorch: PyTorch implementation with autograd support
- HMMLayer: nn.Module wrapper for easy integration
- GaussianHMMLayer: HMM with Gaussian observation model
- NeuralHMM: Neural network-based HMM with contextual modeling
- SemiMarkovHMM: Hidden Semi-Markov Model with duration modeling
- DTWAligner: Dynamic Time Warping alignment
- CTCAligner: Connectionist Temporal Classification alignment
- Speech quality evaluation metrics: MCD, F0 RMSE, alignment accuracy
"""

from .hmm import HMM, HMMPyTorch
from .hmm_layer import HMMLayer, GaussianHMMLayer
from .neural import (
    NeuralTransitionModel,
    NeuralObservationModel, 
    NeuralHMM,
    ContextualNeuralHMM
)
from .semi_markov import (
    DurationModel,
    SemiMarkovHMM,
    AdaptiveDurationHSMM
)
from .alignment import (
    DTWAligner,
    CTCAligner
)
from .metrics import (
    mel_cepstral_distortion,
    f0_root_mean_square_error,
    log_f0_rmse,
    alignment_accuracy,
    boundary_accuracy,
    duration_accuracy,
    spectral_distortion,
    perceptual_evaluation_speech_quality,
    comprehensive_speech_evaluation,
    print_evaluation_summary,
    save_evaluation_results
)
from .utils import (
    create_transition_matrix,
    create_left_to_right_matrix,
    create_duration_constrained_matrix,
    create_gaussian_observation_model,
    gaussian_log_likelihood,
    align_sequences,
    compute_state_durations,
    interpolate_features
)

__version__ = "0.2.0"
__author__ = "Speech Synthesis Engineer"

__all__ = [
    # Core HMM
    "HMM",
    "HMMPyTorch", 
    "HMMLayer",
    "GaussianHMMLayer",
    
    # Neural HMM
    "NeuralTransitionModel",
    "NeuralObservationModel",
    "NeuralHMM",
    "ContextualNeuralHMM",
    
    # Semi-Markov HMM
    "DurationModel",
    "SemiMarkovHMM", 
    "AdaptiveDurationHSMM",
    
    # Alignment algorithms
    "DTWAligner",
    "CTCAligner",
    
    # Evaluation metrics
    "mel_cepstral_distortion",
    "f0_root_mean_square_error",
    "log_f0_rmse",
    "alignment_accuracy",
    "boundary_accuracy", 
    "duration_accuracy",
    "spectral_distortion",
    "perceptual_evaluation_speech_quality",
    "comprehensive_speech_evaluation",
    "print_evaluation_summary",
    "save_evaluation_results",
    
    # Utilities
    "create_transition_matrix",
    "create_left_to_right_matrix",
    "create_duration_constrained_matrix",
    "create_gaussian_observation_model",
    "gaussian_log_likelihood",
    "align_sequences",
    "compute_state_durations",
    "interpolate_features"
]
