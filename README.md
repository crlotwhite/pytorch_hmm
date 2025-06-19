# PyTorch HMM

PyTorch implementation of Hidden Markov Model with forward-backward and Viterbi algorithms, optimized for speech synthesis applications.

## Features

- **PyTorch Native**: Full autograd support and GPU acceleration
- **Efficient Algorithms**: Optimized forward-backward and Viterbi implementations
- **Batch Processing**: Support for batched operations
- **Speech Synthesis Optimized**: Designed for TTS and voice synthesis tasks
- **Numerical Stability**: Log-space computations for numerical stability

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from pytorch_hmm import HMMPyTorch, create_left_to_right_matrix

# Create transition matrix for left-to-right HMM (common in speech)
num_states = 5
P = create_left_to_right_matrix(num_states, self_loop_prob=0.8)

# Initialize HMM
hmm = HMMPyTorch(P)

# Generate some observation probabilities (batch_size=2, seq_len=10, num_states=5)
observations = torch.rand(2, 10, 5)

# Forward-backward algorithm
posterior, forward, backward = hmm.forward_backward(observations)

# Viterbi decoding
states, scores = hmm.viterbi_decode(observations)
```

## Use Cases in Speech Synthesis

- **Phoneme Duration Modeling**: Model phoneme state durations
- **Acoustic Feature Alignment**: Align linguistic features with acoustic features
- **Voice Conversion**: State-based voice transformation
- **Speech Recognition**: Acoustic model components

## License

MIT License
