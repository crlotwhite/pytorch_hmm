"""
Hidden Semi-Markov Model (HSMM) Layer
명시적 지속시간 모델링으로 자연스러운 음성 합성 지원

Author: Speech Synthesis Engineer
Version: 0.2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma, Poisson, Weibull
import math
from typing import Tuple, Optional, List, Union
import warnings

from .hmm import HMMPyTorch


class HSMMLayer(nn.Module):
    """
    Hidden Semi-Markov Model Layer
    명시적 지속시간 모델링으로 자연스러운 음성 합성 지원
    
    Args:
        num_states: Number of HMM states
        feature_dim: Dimension of observation features
        duration_distribution: Type of duration distribution ('gamma', 'poisson', 'weibull')
        max_duration: Maximum allowed duration for any state
        learnable_duration_params: Whether duration parameters are learnable
        min_duration: Minimum duration for any state
    """
    
    def __init__(
        self, 
        num_states: int, 
        feature_dim: int, 
        duration_distribution: str = 'gamma',
        max_duration: int = 50,
        learnable_duration_params: bool = True,
        min_duration: int = 1
    ):
        super().__init__()
        
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.duration_distribution = duration_distribution
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.learnable_duration_params = learnable_duration_params
        
        # Numerical stability
        self.eps = 1e-8
        
        # Initialize parameters
        self._init_parameters()
        
        # Precompute duration range
        self.register_buffer('duration_range', 
                           torch.arange(self.min_duration, self.max_duration + 1, dtype=torch.float))
    
    def _init_parameters(self):
        """Initialize model parameters"""
        
        # Transition probabilities (between different states only - no self-loops in HSMM)
        self.transition_logits = nn.Parameter(
            torch.randn(self.num_states, self.num_states) * 0.1
        )
        
        # Observation model parameters (Gaussian)
        self.observation_means = nn.Parameter(
            torch.randn(self.num_states, self.feature_dim) * 0.1
        )
        self.observation_log_vars = nn.Parameter(
            torch.zeros(self.num_states, self.feature_dim)
        )
        
        # Duration model parameters
        if self.learnable_duration_params:
            if self.duration_distribution == 'gamma':
                # Gamma distribution: shape and rate parameters
                self.duration_shape = nn.Parameter(
                    torch.ones(self.num_states) * 2.0  # Initial shape=2
                )
                self.duration_rate = nn.Parameter(
                    torch.ones(self.num_states) * 0.2   # Initial rate=0.2 (mean=10 frames)
                )
            elif self.duration_distribution == 'poisson':
                # Poisson distribution: lambda parameter
                self.duration_lambda = nn.Parameter(
                    torch.ones(self.num_states) * 10.0  # Mean 10 frames
                )
            elif self.duration_distribution == 'weibull':
                # Weibull distribution: scale and concentration
                self.duration_scale = nn.Parameter(
                    torch.ones(self.num_states) * 10.0
                )
                self.duration_concentration = nn.Parameter(
                    torch.ones(self.num_states) * 2.0
                )
            else:
                raise ValueError(f"Unknown duration distribution: {self.duration_distribution}")
        else:
            # Fixed duration parameters
            self.register_buffer('duration_means', 
                               torch.ones(self.num_states) * 10.0)
    
    def get_transition_matrix(self) -> torch.Tensor:
        """Get transition matrix (no self-loops in HSMM)"""
        # Set diagonal to -inf to prevent self-transitions
        logits = self.transition_logits.clone()
        logits.fill_diagonal_(float('-inf'))
        return F.softmax(logits, dim=-1)
    
    def get_duration_probabilities(self) -> torch.Tensor:
        """
        Get duration probabilities for each state
        
        Returns:
            duration_probs: (num_states, max_duration) probabilities
        """
        durations = self.duration_range  # (max_duration,)
        
        if self.duration_distribution == 'gamma':
            shape = F.softplus(self.duration_shape).unsqueeze(1)  # (num_states, 1)
            rate = F.softplus(self.duration_rate).unsqueeze(1)    # (num_states, 1)
            
            # Gamma PDF computation
            durations_expanded = durations.unsqueeze(0)  # (1, max_duration)
            
            # Log probabilities for numerical stability
            log_probs = (
                (shape - 1) * torch.log(durations_expanded + self.eps) -
                rate * durations_expanded -
                torch.lgamma(shape) +
                shape * torch.log(rate + self.eps)
            )
            
            # Handle edge cases
            log_probs = torch.where(durations_expanded >= self.min_duration, 
                                   log_probs, torch.full_like(log_probs, float('-inf')))
            
            return torch.exp(log_probs)
            
        elif self.duration_distribution == 'poisson':
            lambda_param = F.softplus(self.duration_lambda).unsqueeze(1)  # (num_states, 1)
            durations_expanded = durations.unsqueeze(0)
            
            # Poisson PMF
            log_probs = (
                durations_expanded * torch.log(lambda_param + self.eps) -
                lambda_param -
                torch.lgamma(durations_expanded + 1)
            )
            
            # Apply minimum duration constraint
            log_probs = torch.where(durations_expanded >= self.min_duration,
                                   log_probs, torch.full_like(log_probs, float('-inf')))
            
            return torch.exp(log_probs)
            
        elif self.duration_distribution == 'weibull':
            scale = F.softplus(self.duration_scale).unsqueeze(1)  # (num_states, 1)
            concentration = F.softplus(self.duration_concentration).unsqueeze(1)  # (num_states, 1)
            durations_expanded = durations.unsqueeze(0)
            
            # Weibull PDF
            log_probs = (
                torch.log(concentration + self.eps) - 
                concentration * torch.log(scale + self.eps) +
                (concentration - 1) * torch.log(durations_expanded + self.eps) -
                (durations_expanded / scale) ** concentration
            )
            
            # Apply minimum duration constraint
            log_probs = torch.where(durations_expanded >= self.min_duration,
                                   log_probs, torch.full_like(log_probs, float('-inf')))
            
            return torch.exp(log_probs)
    
    def get_observation_log_probs(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute observation log probabilities
        
        Args:
            observations: (batch_size, seq_len, feature_dim)
        
        Returns:
            log_probs: (batch_size, seq_len, num_states)
        """
        batch_size, seq_len, _ = observations.shape
        
        # Expand for broadcasting
        obs_expanded = observations.unsqueeze(2)  # (B, T, 1, D)
        means_expanded = self.observation_means.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
        vars_expanded = torch.exp(self.observation_log_vars).unsqueeze(0).unsqueeze(0)
        
        # Gaussian log likelihood
        diff = obs_expanded - means_expanded
        log_probs = -0.5 * (
            torch.sum(diff ** 2 / vars_expanded, dim=-1) +
            torch.sum(self.observation_log_vars, dim=-1).unsqueeze(0).unsqueeze(0) +
            self.feature_dim * math.log(2 * math.pi)
        )
        
        return log_probs
    
    def viterbi_decode_hsmm(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HSMM Viterbi decoding with explicit duration modeling
        
        Args:
            observations: (batch_size, seq_len, feature_dim)
        
        Returns:
            decoded_states: (batch_size, seq_len)
            log_probs: (batch_size,) - sequence log probabilities
        """
        batch_size, seq_len, _ = observations.shape
        
        if seq_len > 1000:
            warnings.warn(f"Long sequence ({seq_len} frames) may cause memory issues in HSMM decoding.")
        
        # Get probabilities
        obs_log_probs = self.get_observation_log_probs(observations)  # (B, T, S)
        duration_probs = self.get_duration_probabilities()  # (S, D)
        duration_log_probs = torch.log(duration_probs + self.eps)
        transition_matrix = self.get_transition_matrix()
        log_transitions = torch.log(transition_matrix + self.eps)
        
        # Process each batch separately for simplicity (can be optimized)
        decoded_batch = []
        scores_batch = []
        
        for b in range(batch_size):
            obs_batch = obs_log_probs[b]  # (T, S)
            decoded_states, score = self._viterbi_decode_single(
                obs_batch, duration_log_probs, log_transitions
            )
            decoded_batch.append(decoded_states)
            scores_batch.append(score)
        
        return torch.stack(decoded_batch), torch.stack(scores_batch)
    
    def _viterbi_decode_single(self, obs_log_probs: torch.Tensor, 
                              duration_log_probs: torch.Tensor,
                              log_transitions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single sequence HSMM Viterbi decoding
        
        Args:
            obs_log_probs: (seq_len, num_states)
            duration_log_probs: (num_states, max_duration)
            log_transitions: (num_states, num_states)
        """
        seq_len, num_states = obs_log_probs.shape
        
        # DP table: delta[t][s][d] = best score ending at time t, state s, duration d
        delta = torch.full((seq_len, num_states, self.max_duration), 
                          float('-inf'), device=obs_log_probs.device)
        
        # Backpointers: (prev_state, prev_duration)
        psi_state = torch.zeros((seq_len, num_states, self.max_duration), 
                               dtype=torch.long, device=obs_log_probs.device)
        psi_duration = torch.zeros((seq_len, num_states, self.max_duration), 
                                  dtype=torch.long, device=obs_log_probs.device)
        
        # Initialize first frame
        for s in range(num_states):
            for d in range(1, min(self.max_duration + 1, seq_len + 1)):
                if d <= seq_len:
                    # Sum of observation log probs for duration d
                    obs_sum = torch.sum(obs_log_probs[:d, s])
                    delta[d-1, s, d-1] = obs_sum + duration_log_probs[s, d-1]
        
        # Forward pass
        for t in range(1, seq_len):
            for s in range(num_states):
                for d in range(1, min(self.max_duration + 1, seq_len - t + 1)):
                    end_time = t + d - 1
                    if end_time >= seq_len:
                        continue
                    
                    # Observation score for this duration
                    obs_sum = torch.sum(obs_log_probs[t:t+d, s])
                    duration_score = duration_log_probs[s, d-1]
                    
                    # Find best previous state transition
                    best_score = float('-inf')
                    best_prev_state = 0
                    best_prev_duration = 1
                    
                    for prev_s in range(num_states):
                        if prev_s == s:  # No self-transitions in HSMM
                            continue
                        
                        for prev_d in range(1, self.max_duration + 1):
                            prev_end_time = t - 1
                            prev_start_time = prev_end_time - prev_d + 1
                            
                            if prev_start_time >= 0 and prev_start_time < seq_len:
                                prev_score = delta[prev_end_time, prev_s, prev_d-1]
                                if prev_score != float('-inf'):
                                    total_score = (prev_score + 
                                                 log_transitions[prev_s, s] + 
                                                 obs_sum + duration_score)
                                    
                                    if total_score > best_score:
                                        best_score = total_score
                                        best_prev_state = prev_s
                                        best_prev_duration = prev_d
                    
                    if best_score != float('-inf'):
                        delta[end_time, s, d-1] = best_score
                        psi_state[end_time, s, d-1] = best_prev_state
                        psi_duration[end_time, s, d-1] = best_prev_duration
        
        # Find best final state and duration
        best_score = float('-inf')
        best_final_state = 0
        best_final_duration = 1
        
        for s in range(num_states):
            for d in range(1, self.max_duration + 1):
                score = delta[seq_len-1, s, d-1]
                if score > best_score:
                    best_score = score
                    best_final_state = s
                    best_final_duration = d
        
        # Backtrack to reconstruct path
        states = torch.zeros(seq_len, dtype=torch.long, device=obs_log_probs.device)
        t = seq_len - 1
        current_state = best_final_state
        current_duration = best_final_duration
        
        while t >= 0:
            # Fill current segment
            start_t = max(0, t - current_duration + 1)
            states[start_t:t+1] = current_state
            
            # Move to previous segment
            if start_t > 0:
                prev_t = start_t - 1
                prev_state = psi_state[t, current_state, current_duration-1]
                prev_duration = psi_duration[t, current_state, current_duration-1]
                
                t = prev_t
                current_state = prev_state
                current_duration = prev_duration
            else:
                break
        
        return states, torch.tensor(best_score, device=obs_log_probs.device)
    
    def generate_sequence(self, length: int, initial_state: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a sequence using the HSMM model
        
        Args:
            length: Length of sequence to generate
            initial_state: Starting state
        
        Returns:
            states: (length,) state sequence
            observations: (length, feature_dim) observation sequence
        """
        states = torch.zeros(length, dtype=torch.long, device=self.observation_means.device)
        observations = torch.zeros(length, self.feature_dim, device=self.observation_means.device)
        
        current_state = initial_state
        t = 0
        
        transition_matrix = self.get_transition_matrix()
        duration_probs = self.get_duration_probabilities()
        
        while t < length:
            # Sample duration for current state
            duration_dist = torch.distributions.Categorical(duration_probs[current_state])
            duration = duration_dist.sample().item() + self.min_duration
            
            # Generate observations for this duration
            end_t = min(t + duration, length)
            
            for i in range(t, end_t):
                states[i] = current_state
                
                # Generate observation from Gaussian
                mean = self.observation_means[current_state]
                var = torch.exp(self.observation_log_vars[current_state])
                observations[i] = torch.normal(mean, torch.sqrt(var))
            
            t = end_t
            
            # Transition to next state
            if t < length:
                # Sample next state (excluding current state)
                valid_transitions = transition_matrix[current_state].clone()
                valid_transitions[current_state] = 0  # No self-transitions
                valid_transitions = valid_transitions / valid_transitions.sum()
                
                next_state_dist = torch.distributions.Categorical(valid_transitions)
                current_state = next_state_dist.sample().item()
        
        return states, observations
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through HSMM
        
        Args:
            observations: (batch_size, seq_len, feature_dim)
        
        Returns:
            decoded_states: (batch_size, seq_len)
            log_probs: (batch_size,)
        """
        return self.viterbi_decode_hsmm(observations)
    
    def get_expected_durations(self) -> torch.Tensor:
        """Get expected duration for each state"""
        if self.duration_distribution == 'gamma':
            shape = F.softplus(self.duration_shape)
            rate = F.softplus(self.duration_rate)
            return shape / rate
        elif self.duration_distribution == 'poisson':
            return F.softplus(self.duration_lambda)
        elif self.duration_distribution == 'weibull':
            scale = F.softplus(self.duration_scale)
            concentration = F.softplus(self.duration_concentration)
            return scale * torch.exp(torch.lgamma(1 + 1/concentration))
        else:
            return self.duration_means
    
    def get_model_info(self) -> dict:
        """Get model configuration and statistics"""
        expected_durations = self.get_expected_durations()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'HSMM',
            'num_states': self.num_states,
            'feature_dim': self.feature_dim,
            'duration_distribution': self.duration_distribution,
            'max_duration': self.max_duration,
            'min_duration': self.min_duration,
            'expected_durations': expected_durations.tolist(),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'learnable_durations': self.learnable_duration_params
        }


class DurationConstrainedHMM(nn.Module):
    """
    Duration-constrained HMM that enforces minimum and maximum durations
    without full HSMM complexity
    """
    
    def __init__(self, num_states: int, feature_dim: int, 
                 min_duration: int = 3, max_duration: int = 30):
        super().__init__()
        
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Standard HMM parameters
        self.transition_logits = nn.Parameter(
            torch.randn(num_states, num_states) * 0.1
        )
        self.emission_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_states),
            nn.LogSoftmax(dim=-1)
        )
        
        # Duration constraints (soft)
        self.duration_penalty_weight = 0.1
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward with duration constraints"""
        batch_size, seq_len, _ = observations.shape
        
        # Get emission probabilities
        emission_log_probs = self.emission_net(observations)  # (B, T, S)
        
        # Get transition probabilities
        transition_matrix = F.softmax(self.transition_logits, dim=-1)
        log_transitions = torch.log(transition_matrix + 1e-8)
        
        # Constrained Viterbi with duration penalties
        decoded_states = self._constrained_viterbi(emission_log_probs, log_transitions)
        
        return decoded_states
    
    def _constrained_viterbi(self, emission_log_probs: torch.Tensor, 
                            log_transitions: torch.Tensor) -> torch.Tensor:
        """Viterbi with duration constraints"""
        batch_size, seq_len, num_states = emission_log_probs.shape
        
        # Standard Viterbi but track duration constraints
        delta = torch.full((batch_size, seq_len, num_states), float('-inf'),
                          device=emission_log_probs.device)
        psi = torch.zeros((batch_size, seq_len, num_states), dtype=torch.long,
                         device=emission_log_probs.device)
        
        # Track state durations
        duration_tracker = torch.zeros((batch_size, seq_len, num_states), dtype=torch.long,
                                      device=emission_log_probs.device)
        
        # Initialize
        delta[:, 0, :] = emission_log_probs[:, 0, :] - math.log(num_states)
        duration_tracker[:, 0, :] = 1
        
        # Forward pass with duration tracking
        for t in range(1, seq_len):
            for s in range(num_states):
                best_score = float('-inf')
                best_prev_state = 0
                best_duration = 1
                
                for prev_s in range(num_states):
                    if prev_s == s:
                        # Same state - increment duration
                        new_duration = duration_tracker[:, t-1, prev_s] + 1
                        
                        # Apply duration penalty if too long
                        duration_penalty = 0
                        if new_duration.max() > self.max_duration:
                            duration_penalty = -self.duration_penalty_weight * (new_duration.max() - self.max_duration)
                        
                        score = delta[:, t-1, prev_s] + emission_log_probs[:, t, s] + duration_penalty
                        
                    else:
                        # State change - check minimum duration
                        prev_duration = duration_tracker[:, t-1, prev_s]
                        
                        # Apply penalty if previous state was too short
                        duration_penalty = 0
                        if prev_duration.min() < self.min_duration:
                            duration_penalty = -self.duration_penalty_weight * (self.min_duration - prev_duration.min())
                        
                        score = (delta[:, t-1, prev_s] + log_transitions[prev_s, s] + 
                                emission_log_probs[:, t, s] + duration_penalty)
                        new_duration = torch.ones_like(duration_tracker[:, t-1, prev_s])
                    
                    # Update if better
                    mask = score > delta[:, t, s]
                    delta[:, t, s] = torch.where(mask, score, delta[:, t, s])
                    psi[:, t, s] = torch.where(mask, prev_s, psi[:, t, s])
                    duration_tracker[:, t, s] = torch.where(mask, new_duration, duration_tracker[:, t, s])
        
        # Backtrack
        decoded_states = torch.zeros((batch_size, seq_len), dtype=torch.long,
                                   device=emission_log_probs.device)
        
        # Find best final states
        final_scores, final_states = torch.max(delta[:, -1, :], dim=-1)
        decoded_states[:, -1] = final_states
        
        # Backtrack
        for t in range(seq_len - 2, -1, -1):
            prev_states = psi[torch.arange(batch_size), t + 1, decoded_states[:, t + 1]]
            decoded_states[:, t] = prev_states
        
        return decoded_states
