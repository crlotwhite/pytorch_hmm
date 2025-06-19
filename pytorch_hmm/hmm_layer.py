import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np

from .hmm import HMMPyTorch
from .utils import create_left_to_right_matrix, create_transition_matrix


class HMMLayer(nn.Module):
    """
    PyTorch nn.Module wrapper for Hidden Markov Model.
    
    신경망 모델에 HMM을 쉽게 통합할 수 있도록 해주는 레이어입니다.
    음성 합성 모델에서 상태 정렬 및 지속시간 모델링에 활용할 수 있습니다.
    
    Args:
        num_states: HMM 상태 수
        learnable_transitions: transition matrix를 학습 가능하게 할지 여부
        transition_type: 초기 transition matrix 타입
        self_loop_prob: 자기 상태 유지 확률
        viterbi_inference: inference 시 Viterbi 디코딩 사용 여부
        apply_sigmoid: 입력에 sigmoid 적용 여부
    """
    
    def __init__(self, 
                 num_states: int,
                 learnable_transitions: bool = True,
                 transition_type: str = "left_to_right", 
                 self_loop_prob: float = 0.7,
                 viterbi_inference: bool = True,
                 apply_sigmoid: bool = True):
        super(HMMLayer, self).__init__()
        
        self.num_states = num_states
        self.viterbi_inference = viterbi_inference
        self.apply_sigmoid = apply_sigmoid
        
        # Initialize transition matrix
        if transition_type == "left_to_right":
            P_init = create_left_to_right_matrix(num_states, self_loop_prob)
        else:
            P_init = create_transition_matrix(num_states, transition_type, self_loop_prob)
        
        if learnable_transitions:
            # Learnable transition matrix (in log space for stability)
            self.log_transition_logits = nn.Parameter(torch.log(P_init + 1e-8))
        else:
            # Fixed transition matrix
            self.register_buffer('transition_matrix', P_init)
            self.log_transition_logits = None
        
        # Initial state probabilities (learnable)
        p0_init = torch.ones(num_states) / num_states
        self.log_initial_logits = nn.Parameter(torch.log(p0_init + 1e-8))
        
        # HMM instance will be created dynamically
        self._hmm = None
        
    def _get_transition_matrix(self) -> torch.Tensor:
        """현재 transition matrix를 반환합니다."""
        if self.log_transition_logits is not None:
            # Learnable transitions: apply softmax to ensure row-wise normalization
            return F.softmax(self.log_transition_logits, dim=1)
        else:
            return self.transition_matrix
    
    def _get_initial_probabilities(self) -> torch.Tensor:
        """현재 initial probabilities를 반환합니다."""
        return F.softmax(self.log_initial_logits, dim=0)
    
    def _get_hmm(self) -> HMMPyTorch:
        """현재 파라미터로 HMM 인스턴스를 생성/업데이트합니다."""
        P = self._get_transition_matrix()
        p0 = self._get_initial_probabilities()
        device = P.device
        
        if self._hmm is None:
            self._hmm = HMMPyTorch(P, p0, device=str(device))
        else:
            # Update existing HMM with current parameters
            self._hmm.P = P
            self._hmm.log_P = torch.log(P + 1e-8)
            self._hmm.p0 = p0
            self._hmm.log_p0 = torch.log(p0 + 1e-8)
            self._hmm.device = str(device)
            
        return self._hmm
    
    def forward(self, x: torch.Tensor, 
                return_alignment: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through HMM layer.
        
        Args:
            x: Input observations (B, T, K) where K should match num_states
            return_alignment: 추가로 Viterbi alignment도 반환할지 여부
            
        Returns:
            posteriors: 상태 posterior probabilities (B, T, K)
            alignment: (optional) Viterbi alignment (B, T) if return_alignment=True
        """
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        
        # Ensure input has correct shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        B, T, K = x.shape
        if K != self.num_states:
            raise ValueError(f"Input feature dim {K} must match num_states {self.num_states}")
        
        # Get current HMM
        hmm = self._get_hmm()
        
        # Training vs Inference
        if self.training:
            # Training: use forward-backward algorithm
            posteriors, _, _ = hmm.forward_backward(x)
        else:
            # Inference: choose based on viterbi_inference flag
            if self.viterbi_inference:
                # Viterbi decoding (hard alignment)
                states, _ = hmm.viterbi_decode(x)
                # Convert to one-hot posteriors
                posteriors = F.one_hot(states, num_classes=self.num_states).float()
            else:
                # Forward-backward (soft alignment)
                posteriors, _, _ = hmm.forward_backward(x)
        
        if return_alignment and not self.training:
            # Return hard alignment for analysis
            if self.viterbi_inference:
                alignment = states
            else:
                # Get hard alignment from posteriors
                alignment = torch.argmax(posteriors, dim=-1)
            return posteriors, alignment
        
        return posteriors
    
    def compute_loss(self, 
                    observations: torch.Tensor,
                    target_alignment: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        HMM training loss를 계산합니다.
        
        Args:
            observations: 관측값 (B, T, K)
            target_alignment: 타겟 정렬 (B, T) - 제공되면 supervised learning
            
        Returns:
            loss: 계산된 loss
        """
        hmm = self._get_hmm()
        
        if target_alignment is not None:
            # Supervised learning: cross-entropy loss
            posteriors = self.forward(observations)
            # Flatten for cross-entropy
            posteriors_flat = posteriors.view(-1, self.num_states)
            target_flat = target_alignment.view(-1)
            loss = F.cross_entropy(posteriors_flat, target_flat)
        else:
            # Unsupervised learning: negative log-likelihood
            if self.apply_sigmoid:
                observations = torch.sigmoid(observations)
            log_likelihood = hmm.compute_likelihood(observations)
            loss = -log_likelihood.mean()
        
        return loss
    
    def align(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        관측값에 대한 최적 상태 정렬을 수행합니다.
        
        Args:
            observations: 관측값 (B, T, K) 또는 (T, K)
            
        Returns:
            states: 최적 상태 시퀀스 (B, T) 또는 (T,)
            scores: 상태별 스코어 (B, T, K) 또는 (T, K)
        """
        hmm = self._get_hmm()
        
        if self.apply_sigmoid:
            observations = torch.sigmoid(observations)
            
        return hmm.viterbi_decode(observations)
    
    def sample(self, seq_length: int, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HMM에서 시퀀스를 샘플링합니다.
        
        Args:
            seq_length: 시퀀스 길이
            batch_size: 배치 크기
            
        Returns:
            states: 상태 시퀀스 (batch_size, seq_length)
            observations: 관측 확률 (batch_size, seq_length, K)
        """
        hmm = self._get_hmm()
        return hmm.sample(seq_length, batch_size)
    
    def get_transition_matrix(self) -> torch.Tensor:
        """현재 transition matrix를 반환합니다."""
        return self._get_transition_matrix()
    
    def get_initial_probabilities(self) -> torch.Tensor:
        """현재 initial probabilities를 반환합니다."""
        return self._get_initial_probabilities()
    
    def extra_repr(self) -> str:
        return f'num_states={self.num_states}, viterbi_inference={self.viterbi_inference}'


class GaussianHMMLayer(nn.Module):
    """
    Gaussian observation model을 가진 HMM Layer.
    
    연속적인 특징 벡터를 위한 가우시안 관측 모델을 사용합니다.
    음성 특징(MFCC, mel-spectrogram 등)에 적합합니다.
    
    Args:
        num_states: HMM 상태 수
        feature_dim: 입력 특징 차원
        covariance_type: 공분산 타입 ('full', 'diag', 'spherical')
        learnable_transitions: transition matrix 학습 여부
        transition_type: transition matrix 타입
    """
    
    def __init__(self,
                 num_states: int,
                 feature_dim: int,
                 covariance_type: str = 'diag',
                 learnable_transitions: bool = True,
                 transition_type: str = "left_to_right"):
        super(GaussianHMMLayer, self).__init__()
        
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.covariance_type = covariance_type
        
        # HMM transition parameters
        self.hmm_layer = HMMLayer(
            num_states=num_states,
            learnable_transitions=learnable_transitions,
            transition_type=transition_type,
            apply_sigmoid=False  # Gaussian model doesn't need sigmoid
        )
        
        # Gaussian observation model parameters
        self.means = nn.Parameter(torch.randn(num_states, feature_dim))
        
        if covariance_type == 'full':
            # Full covariance matrices
            self.log_scales = nn.Parameter(torch.zeros(num_states, feature_dim, feature_dim))
        elif covariance_type == 'diag':
            # Diagonal covariance matrices
            self.log_scales = nn.Parameter(torch.zeros(num_states, feature_dim))
        elif covariance_type == 'spherical':
            # Spherical (isotropic) covariance matrices
            self.log_scales = nn.Parameter(torch.zeros(num_states, 1))
        else:
            raise ValueError(f"Unknown covariance_type: {covariance_type}")
    
    def _compute_gaussian_log_probs(self, observations: torch.Tensor) -> torch.Tensor:
        """
        가우시안 관측 모델의 log-probability를 계산합니다.
        
        Args:
            observations: (B, T, D)
            
        Returns:
            log_probs: (B, T, K)
        """
        B, T, D = observations.shape
        
        # Expand dimensions for broadcasting
        obs_expanded = observations.unsqueeze(-2)  # (B, T, 1, D)
        means_expanded = self.means.unsqueeze(0).unsqueeze(0)  # (1, 1, K, D)
        
        # Compute differences
        diff = obs_expanded - means_expanded  # (B, T, K, D)
        
        if self.covariance_type == 'spherical':
            # Spherical covariance: σ²I
            log_var = 2 * self.log_scales  # (K, 1)
            var = torch.exp(log_var)
            
            # Mahalanobis distance
            mahal_dist = torch.sum(diff ** 2, dim=-1) / var.squeeze(-1)  # (B, T, K)
            
            # Normalization constant
            log_norm = -0.5 * (D * np.log(2 * np.pi) + D * log_var.squeeze(-1))
            
        elif self.covariance_type == 'diag':
            # Diagonal covariance
            log_var = 2 * self.log_scales  # (K, D)
            var = torch.exp(log_var)
            
            # Mahalanobis distance
            mahal_dist = torch.sum(diff ** 2 / var.unsqueeze(0).unsqueeze(0), dim=-1)  # (B, T, K)
            
            # Normalization constant
            log_norm = -0.5 * (D * np.log(2 * np.pi) + torch.sum(log_var, dim=-1))
            
        elif self.covariance_type == 'full':
            # Full covariance (more complex, simplified implementation)
            # For brevity, using diagonal approximation
            scales = torch.diagonal(self.log_scales, dim1=-2, dim2=-1)  # (K, D)
            log_var = 2 * scales
            var = torch.exp(log_var)
            
            mahal_dist = torch.sum(diff ** 2 / var.unsqueeze(0).unsqueeze(0), dim=-1)
            log_norm = -0.5 * (D * np.log(2 * np.pi) + torch.sum(log_var, dim=-1))
        
        log_probs = log_norm.unsqueeze(0).unsqueeze(0) - 0.5 * mahal_dist
        
        return log_probs
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Gaussian observation model.
        
        Args:
            observations: 연속 특징 벡터 (B, T, D)
            
        Returns:
            posteriors: 상태 posterior probabilities (B, T, K)
        """
        # Compute Gaussian log-probabilities
        log_probs = self._compute_gaussian_log_probs(observations)
        observation_probs = torch.exp(log_probs)
        
        # Apply HMM forward-backward
        return self.hmm_layer(observation_probs)
    
    def compute_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """
        가우시안 HMM의 negative log-likelihood를 계산합니다.
        
        Args:
            observations: 연속 특징 벡터 (B, T, D)
            
        Returns:
            loss: negative log-likelihood
        """
        log_probs = self._compute_gaussian_log_probs(observations)
        observation_probs = torch.exp(log_probs)
        
        # Compute log-likelihood using HMM
        hmm = self.hmm_layer._get_hmm()
        log_likelihood = hmm.compute_likelihood(observation_probs)
        
        return -log_likelihood.mean()
    
    def extra_repr(self) -> str:
        return (f'num_states={self.num_states}, feature_dim={self.feature_dim}, '
                f'covariance_type={self.covariance_type}')
