"""
Mixture Gaussian HMM Layer for complex acoustic modeling
음성 신호의 복잡한 확률 분포를 모델링하기 위한 GMM-HMM

Author: Speech Synthesis Engineer
Version: 0.2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
import math
from typing import Tuple, Optional
import warnings

from .hmm import HMMPyTorch


class MixtureGaussianHMMLayer(nn.Module):
    """
    Mixture of Gaussians HMM Layer for complex acoustic modeling
    음성 신호의 복잡한 확률 분포를 모델링하기 위한 GMM-HMM
    
    Args:
        num_states: Number of HMM states
        feature_dim: Dimension of observation features
        num_components: Number of Gaussian components per state
        covariance_type: Type of covariance matrix ('diag', 'full', 'tied', 'spherical')
        learnable_transitions: Whether transition matrix is learnable
        max_sequence_length: Maximum sequence length for memory optimization
    """
    
    def __init__(
        self, 
        num_states: int, 
        feature_dim: int, 
        num_components: int = 3,
        covariance_type: str = 'diag', 
        learnable_transitions: bool = True,
        max_sequence_length: int = 10000
    ):
        super().__init__()
        
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.num_components = num_components
        self.covariance_type = covariance_type
        self.learnable_transitions = learnable_transitions
        self.max_sequence_length = max_sequence_length
        
        # Numerical stability constants
        self.eps = 1e-8
        self.log_eps = math.log(self.eps)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters with proper scaling"""
        
        # Transition matrix (learnable or fixed)
        if self.learnable_transitions:
            self.transition_logits = nn.Parameter(
                torch.randn(self.num_states, self.num_states) * 0.1
            )
        else:
            self.register_buffer('transition_matrix', 
                               self._create_left_to_right_matrix())
        
        # GMM parameters for each state
        self.mixture_weights_logits = nn.Parameter(
            torch.randn(self.num_states, self.num_components) * 0.1
        )
        
        # Gaussian means with Xavier initialization
        gain = math.sqrt(2.0 / self.feature_dim)
        self.means = nn.Parameter(
            torch.randn(self.num_states, self.num_components, self.feature_dim) * gain
        )
        
        # Covariance parameters
        if self.covariance_type == 'diag':
            # Diagonal covariance matrices
            self.log_vars = nn.Parameter(
                torch.zeros(self.num_states, self.num_components, self.feature_dim)
            )
        elif self.covariance_type == 'full':
            # Full covariance matrices via Cholesky decomposition
            tril_size = self.feature_dim * (self.feature_dim + 1) // 2
            self.cholesky_params = nn.Parameter(
                torch.zeros(self.num_states, self.num_components, tril_size)
            )
            # Initialize diagonal elements to be positive
            self._init_cholesky_diagonal()
        elif self.covariance_type == 'tied':
            # Single covariance matrix for all states and components
            self.log_vars = nn.Parameter(
                torch.zeros(self.feature_dim)
            )
        elif self.covariance_type == 'spherical':
            # Single variance value for all dimensions
            self.log_vars = nn.Parameter(
                torch.zeros(self.num_states, self.num_components)
            )
        else:
            raise ValueError(f"Unknown covariance_type: {self.covariance_type}")
    
    def _init_cholesky_diagonal(self):
        """Initialize diagonal elements of Cholesky factors"""
        if self.covariance_type == 'full':
            with torch.no_grad():
                for s in range(self.num_states):
                    for c in range(self.num_components):
                        # Set diagonal elements to small positive values
                        diag_indices = [i * (i + 1) // 2 + i for i in range(self.feature_dim)]
                        self.cholesky_params.data[s, c, diag_indices] = 0.1
    
    def _create_left_to_right_matrix(self):
        """Create left-to-right transition matrix for speech"""
        P = torch.zeros(self.num_states, self.num_states)
        for i in range(self.num_states):
            if i < self.num_states - 1:
                P[i, i] = 0.8      # self-loop
                P[i, i + 1] = 0.2  # forward transition
            else:
                P[i, i] = 1.0      # final state
        return P
    
    def get_transition_matrix(self) -> torch.Tensor:
        """Get current transition matrix"""
        if self.learnable_transitions:
            return F.softmax(self.transition_logits, dim=-1)
        else:
            return self.transition_matrix
    
    def _safe_log(self, x: torch.Tensor) -> torch.Tensor:
        """Numerically stable logarithm"""
        return torch.log(torch.clamp(x, min=self.eps))
    
    def _log_sum_exp(self, x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
        """Numerically stable log-sum-exp"""
        max_x = torch.max(x, dim=dim, keepdim=True)[0]
        max_x = torch.where(torch.isinf(max_x), torch.zeros_like(max_x), max_x)
        
        exp_x = torch.exp(x - max_x)
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=keepdim)
        
        result = self._safe_log(sum_exp)
        if keepdim:
            result = result + max_x
        else:
            result = result + max_x.squeeze(dim)
        
        return result
    
    def get_observation_log_probs(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of observations under GMM
        
        Args:
            observations: (batch_size, seq_len, feature_dim)
        
        Returns:
            log_probs: (batch_size, seq_len, num_states)
        """
        batch_size, seq_len, _ = observations.shape
        
        # Check sequence length
        if seq_len > self.max_sequence_length:
            warnings.warn(f"Sequence length {seq_len} exceeds recommended maximum "
                         f"{self.max_sequence_length}. Consider chunked processing.")
        
        # Expand observations for broadcasting
        obs_expanded = observations.unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1, D)
        
        # Get mixture weights
        mixture_weights = F.softmax(self.mixture_weights_logits, dim=-1)  # (S, C)
        log_mixture_weights = self._safe_log(mixture_weights)
        
        # Compute Gaussian log probabilities based on covariance type
        if self.covariance_type == 'diag':
            log_probs_components = self._diagonal_gaussian_log_probs(obs_expanded)
        elif self.covariance_type == 'full':
            log_probs_components = self._full_gaussian_log_probs(obs_expanded)
        elif self.covariance_type == 'tied':
            log_probs_components = self._tied_gaussian_log_probs(obs_expanded)
        elif self.covariance_type == 'spherical':
            log_probs_components = self._spherical_gaussian_log_probs(obs_expanded)
        
        # Apply mixture weights
        log_probs_weighted = (log_probs_components + 
                             log_mixture_weights.unsqueeze(0).unsqueeze(0))
        
        # Log-sum-exp over components
        log_probs = self._log_sum_exp(log_probs_weighted, dim=-1)  # (B, T, S)
        
        return log_probs
    
    def _diagonal_gaussian_log_probs(self, obs_expanded: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for diagonal covariance"""
        vars = torch.exp(self.log_vars)  # (S, C, D)
        means_expanded = self.means.unsqueeze(0).unsqueeze(0)  # (1, 1, S, C, D)
        vars_expanded = vars.unsqueeze(0).unsqueeze(0)  # (1, 1, S, C, D)
        
        # Compute log probabilities
        diff = obs_expanded - means_expanded  # (B, T, S, C, D)
        log_probs = -0.5 * (
            torch.sum(diff ** 2 / vars_expanded, dim=-1) +
            torch.sum(self.log_vars, dim=-1).unsqueeze(0).unsqueeze(0) +
            self.feature_dim * math.log(2 * math.pi)
        )
        
        return log_probs
    
    def _full_gaussian_log_probs(self, obs_expanded: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for full covariance"""
        batch_size, seq_len = obs_expanded.shape[:2]
        
        # Reconstruct Cholesky factors
        L = self._get_cholesky_factors()  # (S, C, D, D)
        
        # Compute Mahalanobis distance using Cholesky decomposition
        means_expanded = self.means.unsqueeze(0).unsqueeze(0)  # (1, 1, S, C, D)
        diff = obs_expanded - means_expanded  # (B, T, S, C, D)
        
        # Solve L @ x = diff for x
        L_expanded = L.unsqueeze(0).unsqueeze(0)  # (1, 1, S, C, D, D)
        x = torch.linalg.solve_triangular(L_expanded, diff.unsqueeze(-1), upper=False)
        mahalanobis_sq = torch.sum(x.squeeze(-1) ** 2, dim=-1)
        
        # Log determinant from Cholesky factors
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + self.eps), dim=-1)
        log_det = log_det.unsqueeze(0).unsqueeze(0)  # (1, 1, S, C)
        
        log_probs = -0.5 * (
            mahalanobis_sq + log_det + self.feature_dim * math.log(2 * math.pi)
        )
        
        return log_probs
    
    def _tied_gaussian_log_probs(self, obs_expanded: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for tied covariance"""
        vars = torch.exp(self.log_vars)  # (D,)
        means_expanded = self.means.unsqueeze(0).unsqueeze(0)  # (1, 1, S, C, D)
        
        diff = obs_expanded - means_expanded  # (B, T, S, C, D)
        log_probs = -0.5 * (
            torch.sum(diff ** 2 / vars.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0), dim=-1) +
            torch.sum(self.log_vars) +
            self.feature_dim * math.log(2 * math.pi)
        )
        
        return log_probs
    
    def _spherical_gaussian_log_probs(self, obs_expanded: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for spherical covariance"""
        vars = torch.exp(self.log_vars)  # (S, C)
        means_expanded = self.means.unsqueeze(0).unsqueeze(0)  # (1, 1, S, C, D)
        vars_expanded = vars.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, S, C, 1)
        
        diff = obs_expanded - means_expanded  # (B, T, S, C, D)
        log_probs = -0.5 * (
            torch.sum(diff ** 2, dim=-1) / vars_expanded.squeeze(-1) +
            self.feature_dim * self.log_vars.unsqueeze(0).unsqueeze(0) +
            self.feature_dim * math.log(2 * math.pi)
        )
        
        return log_probs
    
    def _get_cholesky_factors(self) -> torch.Tensor:
        """Reconstruct Cholesky factors from parameters"""
        batch_size = self.num_states * self.num_components
        
        # Initialize lower triangular matrices
        L = torch.zeros(batch_size, self.feature_dim, self.feature_dim, 
                       device=self.cholesky_params.device,
                       dtype=self.cholesky_params.dtype)
        
        # Fill lower triangular part
        tril_indices = torch.tril_indices(self.feature_dim, self.feature_dim)
        L[:, tril_indices[0], tril_indices[1]] = self.cholesky_params.view(-1, self.cholesky_params.size(-1))
        
        # Ensure positive diagonal elements
        diag_indices = torch.arange(self.feature_dim)
        L[:, diag_indices, diag_indices] = torch.exp(L[:, diag_indices, diag_indices])
        
        return L.view(self.num_states, self.num_components, self.feature_dim, self.feature_dim)
    
    def _viterbi_decode(self, obs_log_probs: torch.Tensor, 
                       log_transitions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Viterbi decoding algorithm optimized with TorchScript
        
        Args:
            obs_log_probs: (batch_size, seq_len, num_states)
            log_transitions: (num_states, num_states)
        
        Returns:
            decoded_states: (batch_size, seq_len)
            log_probs: (batch_size,) - sequence log probabilities
        """
        batch_size, seq_len, num_states = obs_log_probs.shape
        
        # Initialize dynamic programming tables
        delta = torch.full((batch_size, seq_len, num_states), float('-inf'), 
                          device=obs_log_probs.device, dtype=obs_log_probs.dtype)
        psi = torch.zeros((batch_size, seq_len, num_states), dtype=torch.long,
                         device=obs_log_probs.device)
        
        # Initialization (uniform prior)
        delta[:, 0, :] = obs_log_probs[:, 0, :] - math.log(num_states)
        
        # Forward pass
        for t in range(1, seq_len):
            # Transition scores: delta[t-1] + log_transitions
            transition_scores = delta[:, t-1, :].unsqueeze(-1) + log_transitions.unsqueeze(0)
            
            # Find best previous states
            best_scores, best_prev_states = torch.max(transition_scores, dim=-2)
            
            # Update tables
            delta[:, t, :] = best_scores + obs_log_probs[:, t, :]
            psi[:, t, :] = best_prev_states
        
        # Find best final states
        final_scores, final_states = torch.max(delta[:, -1, :], dim=-1)
        
        # Backward pass - reconstruct optimal paths
        decoded_states = torch.zeros((batch_size, seq_len), dtype=torch.long, 
                                   device=obs_log_probs.device)
        decoded_states[:, -1] = final_states
        
        for t in range(seq_len - 2, -1, -1):
            prev_states = psi[torch.arange(batch_size), t + 1, decoded_states[:, t + 1]]
            decoded_states[:, t] = prev_states
        
        return decoded_states, final_scores
    
    def forward(self, observations: torch.Tensor, 
                return_log_probs: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the Mixture Gaussian HMM
        
        Args:
            observations: (batch_size, seq_len, feature_dim)
            return_log_probs: Whether to return sequence log probabilities
        
        Returns:
            decoded_states: (batch_size, seq_len)
            log_probs: (batch_size,) if return_log_probs else None
        """
        # Compute observation log probabilities
        obs_log_probs = self.get_observation_log_probs(observations)
        
        # Get transition probabilities
        log_transitions = self._safe_log(self.get_transition_matrix())
        
        # Viterbi decoding
        decoded_states, sequence_log_probs = self._viterbi_decode(obs_log_probs, log_transitions)
        
        if return_log_probs:
            return decoded_states, sequence_log_probs
        else:
            return decoded_states, None
    
    def get_model_info(self) -> dict:
        """Get model configuration and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'num_states': self.num_states,
            'feature_dim': self.feature_dim,
            'num_components': self.num_components,
            'covariance_type': self.covariance_type,
            'learnable_transitions': self.learnable_transitions,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_efficient': True,
            'max_sequence_length': self.max_sequence_length
        }
