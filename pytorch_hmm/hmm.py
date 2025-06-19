import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union


class HMM:
    """
    Base class for Hidden Markov Models.
    
    음성 합성에서 주로 사용되는 HMM의 기본 구조를 정의합니다.
    transition matrix P와 initial probability p0를 관리합니다.
    
    Args:
        P: Transition matrix of shape (K, K) where K is number of states
        p0: Initial state probabilities of shape (K,). If None, uniform distribution
        device: Device to place tensors ('cpu' or 'cuda')
    """
    
    def __init__(self, P: Union[np.ndarray, torch.Tensor], 
                 p0: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 device: str = 'cpu'):
        
        # Convert to torch tensor if needed
        if isinstance(P, np.ndarray):
            P = torch.from_numpy(P).float()
        P = P.to(device)
        
        self.K = P.shape[0]  # Number of states
        self.device = device
        
        # Validation
        if len(P.shape) != 2:
            raise ValueError(f'P shape should have length 2. found {len(P.shape)}')
        if P.shape[0] != P.shape[1]:
            raise ValueError(f'P should be square, found {P.shape}')
        
        # Normalize transition matrix
        P = P / P.sum(dim=1, keepdim=True)
        
        self.P = P
        self.log_P = torch.log(P + 1e-8)  # Add small epsilon for numerical stability
        
        # Initial probabilities
        if p0 is None:
            self.p0 = torch.ones(self.K, device=device) / self.K
        else:
            if isinstance(p0, np.ndarray):
                p0 = torch.from_numpy(p0).float()
            p0 = p0.to(device)
            if len(p0) != self.K:
                raise ValueError(f'dimensions of p0 {p0.shape} must match P[0] {P.shape[0]}')
            self.p0 = p0 / p0.sum()  # Normalize
            
        self.log_p0 = torch.log(self.p0 + 1e-8)


class HMMPyTorch(HMM):
    """
    PyTorch implementation of Hidden Markov Model with autograd support.
    
    음성 합성에 최적화된 forward-backward 및 Viterbi 알고리즘을 제공합니다.
    배치 처리와 GPU 가속을 지원하며, numerical stability를 위해 log-space에서 계산합니다.
    """
    
    def forward_backward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward-backward 알고리즘을 실행합니다.
        
        Args:
            observations: Shape (B, T, K) 또는 (T, K)
                         B: batch size, T: sequence length, K: number of states
                         
        Returns:
            posterior: 각 시점에서의 상태 posterior probability (B, T, K)
            forward: Forward probabilities (B, T, K)  
            backward: Backward probabilities (B, T, K)
        """
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
            
        B, T, K = observations.shape
        assert K == self.K, f"Observation dim {K} must match model states {self.K}"
        
        # Log-space에서 계산하여 numerical stability 확보
        log_obs = torch.log(observations + 1e-8)
        
        # Forward pass
        log_forward = torch.zeros(B, T, K, device=self.device)
        
        # Initial forward probabilities
        log_forward[:, 0] = self.log_p0 + log_obs[:, 0]
        
        # Forward recursion
        for t in range(1, T):
            # log_forward[:, t-1, :, None] + self.log_P[None, :, :]
            # Shape: (B, K, 1) + (1, K, K) = (B, K, K)
            transition_scores = log_forward[:, t-1, :, None] + self.log_P[None, :, :]
            
            # log-sum-exp across previous states
            log_forward[:, t] = torch.logsumexp(transition_scores, dim=1) + log_obs[:, t]
        
        # Backward pass
        log_backward = torch.zeros(B, T, K, device=self.device)
        
        # Initialize backward (uniform in log space)
        log_backward[:, -1] = 0.0  # log(1) = 0
        
        # Backward recursion  
        for t in range(T-2, -1, -1):
            # self.log_P broadcast to batch dimension
            # Shape: (1, K, K) + (B, 1, K) + (B, 1, K) = (B, K, K)
            combined = (self.log_P[None, :, :] +
                       log_obs[:, t+1, None, :] +
                       log_backward[:, t+1, None, :])
            
            log_backward[:, t] = torch.logsumexp(combined, dim=2)
        
        # Compute posterior probabilities
        log_posterior = log_forward + log_backward
        
        # Normalize (convert from log space)
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=-1, keepdim=True)
        
        # Convert back to probability space
        posterior = torch.exp(log_posterior)
        forward = torch.exp(log_forward)
        backward = torch.exp(log_backward)
        
        return posterior, forward, backward
    
    def viterbi_decode(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Viterbi 알고리즘으로 최적 상태 시퀀스를 찾습니다.
        
        Args:
            observations: Shape (B, T, K) 또는 (T, K)
            
        Returns:
            states: 최적 상태 시퀀스 (B, T)
            scores: 각 시점의 상태별 스코어 (B, T, K)
        """
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        B, T, K = observations.shape
        assert K == self.K, f"Observation dim {K} must match model states {self.K}"
        
        log_obs = torch.log(observations + 1e-8)
        
        # Viterbi trellis
        log_delta = torch.zeros(B, T, K, device=self.device)
        psi = torch.zeros(B, T, K, dtype=torch.long, device=self.device)
        
        # Initialize
        log_delta[:, 0] = self.log_p0 + log_obs[:, 0]
        
        # Forward pass
        for t in range(1, T):
            # transition_scores: (B, K, K) - score from each prev state to each current state
            transition_scores = log_delta[:, t-1, :, None] + self.log_P[None, :, :]
            
            # Find best previous state for each current state
            log_delta[:, t], psi[:, t] = torch.max(transition_scores, dim=1)
            log_delta[:, t] += log_obs[:, t]
        
        # Backtrack to find best path
        states = torch.zeros(B, T, dtype=torch.long, device=self.device)
        
        # Start from best final state
        states[:, -1] = torch.argmax(log_delta[:, -1], dim=1)
        
        # Backtrack
        for t in range(T-2, -1, -1):
            states[:, t] = psi[torch.arange(B), t+1, states[:, t+1]]
        
        if squeeze_output:
            states = states.squeeze(0)
            log_delta = log_delta.squeeze(0)
            
        return states, log_delta
    
    def compute_likelihood(self, observations: torch.Tensor) -> torch.Tensor:
        """
        주어진 관측 시퀀스의 log-likelihood를 계산합니다.
        
        Args:
            observations: Shape (B, T, K) 또는 (T, K)
            
        Returns:
            log_likelihood: Shape (B,) 또는 scalar
        """
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Forward algorithm으로 likelihood 계산
        _, forward, _ = self.forward_backward(observations)
        
        # Final time step의 forward probability들을 sum
        log_likelihood = torch.logsumexp(torch.log(forward[:, -1] + 1e-8), dim=-1)
        
        if squeeze_output:
            log_likelihood = log_likelihood.squeeze(0)
            
        return log_likelihood
    
    def sample(self, seq_length: int, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HMM에서 상태 시퀀스와 관측을 샘플링합니다.
        
        Args:
            seq_length: 시퀀스 길이
            batch_size: 배치 크기
            
        Returns:
            states: 상태 시퀀스 (batch_size, seq_length)
            observations: 관측 확률 (batch_size, seq_length, K)
        """
        states = torch.zeros(batch_size, seq_length, dtype=torch.long, device=self.device)
        observations = torch.zeros(batch_size, seq_length, self.K, device=self.device)
        
        # Initial state sampling
        initial_dist = torch.distributions.Categorical(self.p0)
        states[:, 0] = initial_dist.sample((batch_size,))
        
        # Generate sequence
        for t in range(seq_length):
            if t > 0:
                # Transition to next state
                prev_states = states[:, t-1]
                transition_probs = self.P[prev_states]  # (batch_size, K)
                state_dist = torch.distributions.Categorical(transition_probs)
                states[:, t] = state_dist.sample()
            
            # Generate observations (identity observation model for simplicity)
            current_states = states[:, t]
            observations[torch.arange(batch_size), t, current_states] = 1.0
        
        return states, observations
    
    def to(self, device: str):
        """모델을 지정된 device로 이동합니다."""
        self.device = device
        self.P = self.P.to(device)
        self.log_P = self.log_P.to(device)
        self.p0 = self.p0.to(device)
        self.log_p0 = self.log_p0.to(device)
        return self
