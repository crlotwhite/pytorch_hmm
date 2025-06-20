"""PyTorch implementation of Hidden Markov Model.

This library provides efficient implementations of HMM algorithms
optimized for speech synthesis and other sequence modeling tasks.

Main components:
- HMM: Base class with model parameters
- HMMPyTorch: PyTorch implementation with autograd support
- HMMLayer: nn.Module wrapper for easy integration
- GaussianHMMLayer: HMM with Gaussian observation model
- MixtureGaussianHMMLayer: HMM with Mixture of Gaussians observation model
- HSMMLayer: Hidden Semi-Markov Model with explicit duration modeling
- StreamingHMMProcessor: Real-time streaming HMM processor
- AdaptiveTransitionMatrix: Context-dependent transition matrices

Version 0.2.0 Features:
- Mixture Gaussian HMM for complex acoustic modeling
- Semi-Markov models with explicit duration distributions
- Real-time streaming processing with adaptive latency control
- Advanced transition matrix utilities (skip-state, hierarchical, attention-based)
- Memory-efficient implementations for large-scale processing
- Comprehensive benchmarking and validation tools
"""

from .hmm import HMM, HMMPyTorch
from .hmm_layer import HMMLayer, GaussianHMMLayer
from .mixture_gaussian import MixtureGaussianHMMLayer
from .hsmm import HSMMLayer, DurationConstrainedHMM
from .streaming import StreamingHMMProcessor, AdaptiveLatencyController, StreamingResult
from .utils import (
    # Basic transition matrix functions
    create_transition_matrix,
    create_left_to_right_matrix,
    create_duration_constrained_matrix,
    
    # Advanced transition matrix functions  
    create_skip_state_matrix,
    create_phoneme_aware_transitions,
    create_hierarchical_transitions,
    create_attention_based_transitions,
    create_prosody_aware_transitions,
    AdaptiveTransitionMatrix,
    
    # Observation model functions
    create_gaussian_observation_model,
    gaussian_log_likelihood,
    
    # Sequence processing functions
    align_sequences,
    compute_state_durations,
    interpolate_features,
    
    # Optimization and validation
    optimize_transition_matrix,
    validate_transition_matrix,
    benchmark_transition_operations,
    analyze_transition_patterns
)

__version__ = "0.2.0"
__author__ = "Speech Synthesis Engineer"

__all__ = [
    # Core HMM classes
    "HMM",
    "HMMPyTorch", 
    "HMMLayer",
    "GaussianHMMLayer",
    
    # Advanced HMM variants
    "MixtureGaussianHMMLayer",
    "HSMMLayer", 
    "DurationConstrainedHMM",
    
    # Streaming and real-time processing
    "StreamingHMMProcessor",
    "AdaptiveLatencyController", 
    "StreamingResult",
    
    # Basic transition matrix utilities
    "create_transition_matrix",
    "create_left_to_right_matrix",
    "create_duration_constrained_matrix",
    
    # Advanced transition matrix utilities
    "create_skip_state_matrix",
    "create_phoneme_aware_transitions", 
    "create_hierarchical_transitions",
    "create_attention_based_transitions",
    "create_prosody_aware_transitions",
    "AdaptiveTransitionMatrix",
    
    # Observation models
    "create_gaussian_observation_model",
    "gaussian_log_likelihood",
    
    # Sequence processing
    "align_sequences",
    "compute_state_durations", 
    "interpolate_features",
    
    # Optimization and validation
    "optimize_transition_matrix",
    "validate_transition_matrix",
    "benchmark_transition_operations",
    "analyze_transition_patterns"
]

# Version compatibility check
import torch

TORCH_MIN_VERSION = "1.12.0"
if torch.__version__ < TORCH_MIN_VERSION:
    import warnings
    warnings.warn(f"PyTorch {TORCH_MIN_VERSION}+ is recommended. "
                  f"Current version: {torch.__version__}")

# Configuration and settings
class Config:
    """Global configuration for PyTorch HMM"""
    
    # Numerical stability
    EPS = 1e-8
    LOG_EPS = -18.42  # log(1e-8)
    
    # Memory optimization
    DEFAULT_CHUNK_SIZE = 1000
    MAX_SEQUENCE_LENGTH = 10000
    
    # Performance settings
    USE_MIXED_PRECISION = True
    USE_CHECKPOINTING = True
    
    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def set_device(cls, device: str):
        """Set default device"""
        cls.DEVICE = device
    
    @classmethod
    def set_precision(cls, use_mixed_precision: bool):
        """Set mixed precision usage"""
        cls.USE_MIXED_PRECISION = use_mixed_precision
    
    @classmethod
    def get_info(cls):
        """Get current configuration info"""
        return {
            'device': cls.DEVICE,
            'mixed_precision': cls.USE_MIXED_PRECISION, 
            'checkpointing': cls.USE_CHECKPOINTING,
            'chunk_size': cls.DEFAULT_CHUNK_SIZE,
            'max_sequence_length': cls.MAX_SEQUENCE_LENGTH,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }

# Convenience functions for quick model creation
def create_speech_hmm(num_states: int, 
                     feature_dim: int,
                     model_type: str = "mixture_gaussian",
                     **kwargs):
    """
    Create a speech-optimized HMM model
    
    Args:
        num_states: Number of HMM states (typically number of phonemes)
        feature_dim: Dimension of acoustic features (e.g., 80 for mel-spectrogram)
        model_type: Type of model ("mixture_gaussian", "hsmm", "streaming")
        **kwargs: Additional model-specific arguments
        
    Returns:
        Configured HMM model
    """
    if model_type == "mixture_gaussian":
        return MixtureGaussianHMMLayer(
            num_states=num_states,
            feature_dim=feature_dim,
            num_components=kwargs.get('num_components', 3),
            covariance_type=kwargs.get('covariance_type', 'diag'),
            **kwargs
        )
    
    elif model_type == "hsmm":
        return HSMMLayer(
            num_states=num_states,
            feature_dim=feature_dim,
            duration_distribution=kwargs.get('duration_distribution', 'gamma'),
            max_duration=kwargs.get('max_duration', 50),
            **kwargs
        )
    
    elif model_type == "streaming":
        return StreamingHMMProcessor(
            num_states=num_states,
            feature_dim=feature_dim,
            chunk_size=kwargs.get('chunk_size', 160),
            use_beam_search=kwargs.get('use_beam_search', True),
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Choose from: 'mixture_gaussian', 'hsmm', 'streaming'")

def create_korean_tts_hmm(phoneme_list: list = None, **kwargs):
    """
    Create HMM optimized for Korean TTS
    
    Args:
        phoneme_list: List of Korean phonemes (uses default if None)
        **kwargs: Additional arguments for model creation
        
    Returns:
        Korean TTS optimized HMM
    """
    # Default Korean phoneme set
    if phoneme_list is None:
        phoneme_list = [
            'sil', 'a', 'e', 'i', 'o', 'u', 'eo', 'eu', 'ui',  # 모음
            'k', 'n', 't', 'r', 'm', 'p', 's', 'ng', 'j', 'ch', 
            'kh', 'th', 'ph', 'h', 'kk', 'tt', 'pp', 'ss', 'jj'  # 자음
        ]
    
    return create_speech_hmm(
        num_states=len(phoneme_list),
        feature_dim=kwargs.get('feature_dim', 80),
        model_type=kwargs.get('model_type', 'mixture_gaussian'),
        **kwargs
    )

# Quick access to commonly used transition matrices
def get_speech_transitions(num_states: int, speech_type: str = "normal"):
    """
    Get pre-configured transition matrices for speech processing
    
    Args:
        num_states: Number of states
        speech_type: Type of speech ("normal", "fast", "slow", "emotional")
    
    Returns:
        Transition matrix
    """
    if speech_type == "normal":
        return create_left_to_right_matrix(num_states, self_loop_prob=0.7)
    
    elif speech_type == "fast":
        return create_skip_state_matrix(
            num_states, 
            self_loop_prob=0.5, 
            forward_prob=0.4, 
            skip_prob=0.1
        )
    
    elif speech_type == "slow":
        return create_left_to_right_matrix(num_states, self_loop_prob=0.85)
    
    elif speech_type == "emotional":
        # More variable transitions for emotional speech
        return create_transition_matrix(
            num_states, 
            "left_to_right_skip",
            self_loop_prob=0.6,
            forward_prob=0.3,
            skip_prob=0.1
        )
    
    else:
        raise ValueError(f"Unknown speech_type: {speech_type}")

# Model factory for common use cases
class ModelFactory:
    """Factory for creating commonly used HMM configurations"""
    
    @staticmethod
    def create_asr_model(vocabulary_size: int, acoustic_dim: int = 80):
        """Create model optimized for Automatic Speech Recognition"""
        return MixtureGaussianHMMLayer(
            num_states=vocabulary_size,
            feature_dim=acoustic_dim,
            num_components=4,
            covariance_type='diag',
            learnable_transitions=True
        )
    
    @staticmethod  
    def create_tts_model(num_phonemes: int, mel_dim: int = 80):
        """Create model optimized for Text-to-Speech synthesis"""
        return HSMMLayer(
            num_states=num_phonemes,
            feature_dim=mel_dim,
            duration_distribution='gamma',
            max_duration=30,
            learnable_duration_params=True
        )
    
    @staticmethod
    def create_realtime_model(num_states: int, feature_dim: int = 80):
        """Create model optimized for real-time processing"""
        return StreamingHMMProcessor(
            num_states=num_states,
            feature_dim=feature_dim,
            chunk_size=160,  # 10ms chunks
            use_beam_search=False,  # Faster greedy decoding
            lookahead_frames=3      # Minimal lookahead
        )

# Package-level utilities
def get_version():
    """Get package version"""
    return __version__

def get_device_info():
    """Get device and capability information"""
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
    
    return info

def run_quick_test():
    """Run a quick functionality test"""
    print("🧪 Running PyTorch HMM quick test...")
    
    try:
        # Test basic HMM
        hmm = HMMPyTorch(create_left_to_right_matrix(5))
        print("✅ Basic HMM: OK")
        
        # Test Mixture Gaussian HMM
        model = MixtureGaussianHMMLayer(5, 40, 2)
        test_data = torch.randn(2, 20, 40)
        states, _ = model(test_data)
        print("✅ Mixture Gaussian HMM: OK")
        
        # Test HSMM
        hsmm = HSMMLayer(3, 40)
        states, obs = hsmm.generate_sequence(30)
        print("✅ HSMM: OK")
        
        # Test Streaming
        stream = StreamingHMMProcessor(5, 40, chunk_size=10)
        chunk = torch.randn(10, 40)
        result = stream.process_chunk(chunk)
        print("✅ Streaming HMM: OK")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# Optional integrations check
def check_optional_dependencies():
    """Check for optional dependencies"""
    optional_deps = {}
    
    try:
        import librosa
        optional_deps['librosa'] = librosa.__version__
    except ImportError:
        optional_deps['librosa'] = None
    
    try:
        import matplotlib
        optional_deps['matplotlib'] = matplotlib.__version__
    except ImportError:
        optional_deps['matplotlib'] = None
    
    try:
        import torchaudio
        optional_deps['torchaudio'] = torchaudio.__version__
    except ImportError:
        optional_deps['torchaudio'] = None
    
    return optional_deps

# Auto-configuration based on environment
def auto_configure():
    """Automatically configure based on available hardware"""
    device_info = get_device_info()
    
    if device_info['cuda_available']:
        Config.set_device('cuda')
        if device_info['gpu_memory'] >= 8:  # 8GB+ GPU
            Config.MAX_SEQUENCE_LENGTH = 20000
            Config.DEFAULT_CHUNK_SIZE = 2000
        print(f"🚀 Configured for GPU: {device_info['gpu_name']}")
    else:
        Config.set_device('cpu')
        Config.MAX_SEQUENCE_LENGTH = 5000
        Config.DEFAULT_CHUNK_SIZE = 500
        Config.set_precision(False)  # Disable mixed precision on CPU
        print("💻 Configured for CPU processing")
    
    return Config.get_info()

# Initialize configuration on import
_config_info = auto_configure()
