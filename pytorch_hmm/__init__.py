"""PyTorch implementation of Hidden Markov Model.

This library provides efficient implementations of HMM algorithms
optimized for speech synthesis and other sequence modeling tasks.

Main components:
- HMM: Base class with model parameters
- HMMPyTorch: PyTorch implementation with autograd support
- HMMLayer: nn.Module wrapper for easy integration
- GaussianHMMLayer: HMM with Gaussian observation model
"""

from .hmm import HMM, HMMPyTorch
from .hmm_layer import HMMLayer, GaussianHMMLayer
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

__version__ = "0.1.0"
__author__ = "Speech Synthesis Engineer"

__all__ = [
    "HMM",
    "HMMPyTorch", 
    "HMMLayer",
    "GaussianHMMLayer",
    "create_transition_matrix",
    "create_left_to_right_matrix",
    "create_duration_constrained_matrix",
    "create_gaussian_observation_model",
    "gaussian_log_likelihood",
    "align_sequences",
    "compute_state_durations",
    "interpolate_features"
]