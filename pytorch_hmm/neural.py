import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union, Dict, Any

from .hmm import HMMPyTorch


class NeuralTransitionModel(nn.Module):
    """
    신경망 기반 HMM 전이 확률 모델.

    컨텍스트 정보를 활용하여 동적으로 전이 확률을 계산합니다.
    음성 합성에서 음소 컨텍스트나 운율 정보를 반영할 때 유용합니다.

    Args:
        num_states: HMM 상태 수
        context_dim: 컨텍스트 특징 차원
        hidden_dim: 신경망 은닉층 차원
        model_type: 신경망 타입 ('mlp', 'rnn', 'transformer')
        dropout: 드롭아웃 확률
    """

    def __init__(self,
                 num_states: int,
                 context_dim: int,
                 hidden_dim: int = 256,
                 model_type: str = 'mlp',
                 dropout: float = 0.1):
        super(NeuralTransitionModel, self).__init__()

        self.num_states = num_states
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type

        if model_type == 'mlp':
            self.network = nn.Sequential(
                nn.Linear(context_dim + num_states, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_states * num_states)
            )
        elif model_type == 'rnn':
            self.rnn = nn.LSTM(context_dim, hidden_dim, batch_first=True, dropout=dropout)
            self.output_layer = nn.Linear(hidden_dim + num_states, num_states * num_states)
        elif model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=context_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.output_layer = nn.Linear(context_dim + num_states, num_states * num_states)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self,
                context: torch.Tensor,
                current_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        컨텍스트에 기반한 전이 확률 계산.

        Args:
            context: 컨텍스트 특징 (B, T, context_dim) 또는 (B, context_dim)
            current_state: 현재 상태 (B, T, num_states) 또는 (B, num_states)

        Returns:
            transition_probs: 전이 확률 (B, T, num_states, num_states) 또는 (B, num_states, num_states)
        """
        batch_size = context.shape[0]

        if context.dim() == 2:
            # Single timestep
            context = context.unsqueeze(1)
            single_step = True
        else:
            single_step = False

        seq_len = context.shape[1]

        if current_state is None:
            # 균등 분포로 초기화
            current_state = torch.ones(batch_size, seq_len, self.num_states, device=context.device) / self.num_states
        elif current_state.dim() == 2:
            current_state = current_state.unsqueeze(1)

        if self.model_type == 'mlp':
            # MLP 기반 전이 확률
            combined_input = torch.cat([context, current_state], dim=-1)
            logits = self.network(combined_input)

        elif self.model_type == 'rnn':
            # RNN 기반 전이 확률
            rnn_output, _ = self.rnn(context)
            combined_input = torch.cat([rnn_output, current_state], dim=-1)
            logits = self.output_layer(combined_input)

        elif self.model_type == 'transformer':
            # Transformer 기반 전이 확률
            transformer_output = self.transformer(context)
            combined_input = torch.cat([transformer_output, current_state], dim=-1)
            logits = self.output_layer(combined_input)

        # Reshape to transition matrix format
        logits = logits.view(batch_size, seq_len, self.num_states, self.num_states)

        # Softmax to ensure valid probabilities
        transition_probs = F.softmax(logits, dim=-1)

        if single_step:
            transition_probs = transition_probs.squeeze(1)

        return transition_probs


class NeuralObservationModel(nn.Module):
    """
    신경망 기반 HMM 관측 확률 모델.

    복잡한 관측 분포를 모델링할 수 있습니다.
    연속적인 음향 특징이나 고차원 특징에 적합합니다.

    Args:
        num_states: HMM 상태 수
        observation_dim: 관측 특징 차원
        hidden_dim: 신경망 은닉층 차원
        model_type: 모델 타입 ('gaussian', 'mixture', 'autoregressive')
        num_components: mixture 모델의 컴포넌트 수
    """

    def __init__(self,
                 num_states: int,
                 observation_dim: int,
                 hidden_dim: int = 256,
                 model_type: str = 'gaussian',
                 num_components: int = 3,
                 dropout: float = 0.1):
        super(NeuralObservationModel, self).__init__()

        self.num_states = num_states
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.num_components = num_components

        if model_type == 'gaussian':
            # 각 상태별 가우시안 분포 파라미터
            self.mean_net = nn.Linear(hidden_dim, observation_dim)
            self.logvar_net = nn.Linear(hidden_dim, observation_dim)

        elif model_type == 'mixture':
            # Gaussian Mixture Model
            self.weight_net = nn.Linear(hidden_dim, num_components)
            self.mean_net = nn.Linear(hidden_dim, num_components * observation_dim)
            self.logvar_net = nn.Linear(hidden_dim, num_components * observation_dim)

        elif model_type == 'autoregressive':
            # 자기회귀 모델 (음성 생성용)
            self.ar_net = nn.LSTM(observation_dim, hidden_dim, batch_first=True)
            self.output_net = nn.Linear(hidden_dim, observation_dim)

        # 상태별 특징 임베딩
        self.state_embedding = nn.Embedding(num_states, hidden_dim)

        # 공통 특징 추출기
        self.feature_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self,
                observations: torch.Tensor,
                state_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        관측 확률 계산.

        Args:
            observations: 관측값 (B, T, observation_dim)
            state_indices: 상태 인덱스 (B, T) - None이면 모든 상태에 대해 계산

        Returns:
            log_probs: 관측 로그 확률 (B, T, num_states) 또는 (B, T)
        """
        batch_size, seq_len, obs_dim = observations.shape
        device = observations.device

        if state_indices is None:
            # 모든 상태에 대해 확률 계산
            log_probs = torch.zeros(batch_size, seq_len, self.num_states, device=device)

            for state in range(self.num_states):
                state_tensor = torch.full((batch_size, seq_len), state, device=device)
                log_probs[:, :, state] = self._compute_single_state_prob(observations, state_tensor)
        else:
            # 특정 상태에 대해서만 확률 계산
            log_probs = self._compute_single_state_prob(observations, state_indices)

        return log_probs

    def _compute_single_state_prob(self, observations: torch.Tensor, state_indices: torch.Tensor) -> torch.Tensor:
        """단일 상태에 대한 관측 확률 계산."""
        batch_size, seq_len, obs_dim = observations.shape
        device = observations.device

        # 상태 임베딩
        state_embeddings = self.state_embedding(state_indices)  # (B, T, hidden_dim)

        # 관측 특징 추출
        obs_features = self.feature_net(observations)  # (B, T, hidden_dim)

        # 상태와 관측 특징 결합
        combined_features = state_embeddings + obs_features

        if self.model_type == 'gaussian':
            # 단일 가우시안
            means = self.mean_net(combined_features)
            log_vars = self.logvar_net(combined_features)

            # 가우시안 로그 확률
            log_probs = self._gaussian_log_prob(observations, means, log_vars)

        elif self.model_type == 'mixture':
            # Gaussian Mixture Model
            weights = F.softmax(self.weight_net(combined_features), dim=-1)
            means = self.mean_net(combined_features).view(batch_size, seq_len, self.num_components, self.observation_dim)
            log_vars = self.logvar_net(combined_features).view(batch_size, seq_len, self.num_components, self.observation_dim)

            # 각 컴포넌트의 로그 확률
            component_log_probs = torch.zeros(batch_size, seq_len, self.num_components, device=device)
            for c in range(self.num_components):
                component_log_probs[:, :, c] = self._gaussian_log_prob(
                    observations, means[:, :, c], log_vars[:, :, c])

            # Mixture 로그 확률
            log_probs = torch.logsumexp(torch.log(weights + 1e-8) + component_log_probs, dim=-1)

        elif self.model_type == 'autoregressive':
            # 자기회귀 모델
            ar_output, _ = self.ar_net(observations)
            predicted = self.output_net(ar_output)

            # MSE 기반 로그 확률 (간단화)
            mse = F.mse_loss(predicted, observations, reduction='none').mean(dim=-1)
            log_probs = -mse

        return log_probs

    def _gaussian_log_prob(self, x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """다변량 가우시안 로그 확률 계산."""
        var = torch.exp(log_var)

        # 정규화 상수
        log_norm = -0.5 * (self.observation_dim * math.log(2 * math.pi) + torch.sum(log_var, dim=-1))

        # 마할라노비스 거리
        diff = x - mean
        mahal_dist = torch.sum(diff ** 2 / var, dim=-1)

        return log_norm - 0.5 * mahal_dist

    def sample(self, state_indices: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """상태에서 관측값 샘플링."""
        batch_size, seq_len = state_indices.shape
        device = state_indices.device

        samples = torch.zeros(batch_size, seq_len, self.observation_dim, device=device)

        state_embeddings = self.state_embedding(state_indices)

        if self.model_type == 'gaussian':
            means = self.mean_net(state_embeddings)
            log_vars = self.logvar_net(state_embeddings)
            stds = torch.exp(0.5 * log_vars)

            noise = torch.randn_like(means)
            samples = means + stds * noise

        elif self.model_type == 'mixture':
            # Mixture에서 샘플링 (구현 생략)
            pass

        return samples


class NeuralHMM(nn.Module):
    """
    신경망 기반 Hidden Markov Model.

    전이 확률과 관측 확률 모두를 신경망으로 모델링합니다.
    복잡한 시퀀스 패턴과 고차원 관측을 효과적으로 처리할 수 있습니다.

    음성 합성에서는 음소 컨텍스트, 운율 정보 등을 활용하여
    더 자연스러운 음성 정렬과 지속시간 모델링이 가능합니다.

    Args:
        num_states: HMM 상태 수
        observation_dim: 관측 특징 차원
        context_dim: 컨텍스트 특징 차원
        hidden_dim: 신경망 은닉층 차원
        transition_type: 전이 모델 타입
        observation_type: 관측 모델 타입
    """

    def __init__(self,
                 num_states: int,
                 observation_dim: int,
                 context_dim: int = 0,
                 hidden_dim: int = 256,
                 transition_type: str = 'mlp',
                 observation_type: str = 'gaussian',
                 dropout: float = 0.1):
        super(NeuralHMM, self).__init__()

        self.num_states = num_states
        self.observation_dim = observation_dim
        self.context_dim = context_dim

        # 신경망 기반 전이 모델
        if context_dim > 0:
            self.transition_model = NeuralTransitionModel(
                num_states=num_states,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                model_type=transition_type,
                dropout=dropout
            )
        else:
            # 컨텍스트가 없으면 전통적인 전이 행렬 사용
            self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states))
            self.transition_model = None

        # 신경망 기반 관측 모델
        self.observation_model = NeuralObservationModel(
            num_states=num_states,
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            model_type=observation_type,
            dropout=dropout
        )

        # 초기 상태 확률
        self.initial_logits = nn.Parameter(torch.zeros(num_states))

    def forward(self,
                observations: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Neural HMM forward-backward 알고리즘.

        Args:
            observations: 관측 시퀀스 (B, T, observation_dim)
            context: 컨텍스트 특징 (B, T, context_dim)

        Returns:
            posteriors: 상태 posterior 확률 (B, T, num_states)
            forward: Forward 확률 (B, T, num_states)
            backward: Backward 확률 (B, T, num_states)
        """
        batch_size, seq_len, obs_dim = observations.shape
        device = observations.device

        # 관측 확률 계산
        log_obs_probs = self.observation_model(observations)  # (B, T, num_states)

        # 전이 확률 계산
        if self.transition_model is not None and context is not None:
            # 동적 전이 확률 (시간에 따라 변함)
            transition_probs = self.transition_model(context)  # (B, T, num_states, num_states)
            log_transition_probs = torch.log(transition_probs + 1e-8)
        else:
            # 정적 전이 확률
            transition_probs = F.softmax(self.transition_matrix, dim=1)
            log_transition_probs = torch.log(transition_probs + 1e-8)
            log_transition_probs = log_transition_probs.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)

        # 초기 확률
        initial_probs = F.softmax(self.initial_logits, dim=0)
        log_initial_probs = torch.log(initial_probs + 1e-8)

        # Forward 알고리즘
        log_forward = self._forward_algorithm(log_obs_probs, log_transition_probs, log_initial_probs)

        # Backward 알고리즘
        log_backward = self._backward_algorithm(log_obs_probs, log_transition_probs)

        # Posterior 계산
        log_posterior = log_forward + log_backward
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=-1, keepdim=True)

        return torch.exp(log_posterior), torch.exp(log_forward), torch.exp(log_backward)

    def _forward_algorithm(self,
                          log_obs_probs: torch.Tensor,
                          log_transition_probs: torch.Tensor,
                          log_initial_probs: torch.Tensor) -> torch.Tensor:
        """Forward 알고리즘."""
        batch_size, seq_len, num_states = log_obs_probs.shape
        device = log_obs_probs.device

        log_forward = torch.full((batch_size, seq_len, num_states), float('-inf'), device=device)

        # 초기화
        log_forward[:, 0] = log_initial_probs + log_obs_probs[:, 0]

        # Forward 재귀
        for t in range(1, seq_len):
            if self.transition_model is not None:
                # 시간에 따라 변하는 전이 확률
                transition_t = log_transition_probs[:, t-1]  # (B, num_states, num_states)
            else:
                # 고정 전이 확률
                transition_t = log_transition_probs[:, 0]

            # 이전 시점에서 현재 시점으로의 전이
            prev_forward = log_forward[:, t-1].unsqueeze(-1)  # (B, num_states, 1)
            transition_scores = prev_forward + transition_t  # (B, num_states, num_states)

            log_forward[:, t] = torch.logsumexp(transition_scores, dim=1) + log_obs_probs[:, t]

        return log_forward

    def _backward_algorithm(self,
                           log_obs_probs: torch.Tensor,
                           log_transition_probs: torch.Tensor) -> torch.Tensor:
        """Backward 알고리즘."""
        batch_size, seq_len, num_states = log_obs_probs.shape
        device = log_obs_probs.device

        log_backward = torch.full((batch_size, seq_len, num_states), float('-inf'), device=device)

        # 종료 조건
        log_backward[:, -1] = 0.0

        # Backward 재귀
        for t in range(seq_len - 2, -1, -1):
            if self.transition_model is not None:
                # 시간에 따라 변하는 전이 확률
                transition_t = log_transition_probs[:, t]  # (B, num_states, num_states)
            else:
                # 고정 전이 확률
                transition_t = log_transition_probs[:, 0]

            # 다음 시점의 관측과 backward 확률
            next_obs = log_obs_probs[:, t+1].unsqueeze(1)  # (B, 1, num_states)
            next_backward = log_backward[:, t+1].unsqueeze(1)  # (B, 1, num_states)

            combined = transition_t + next_obs + next_backward  # (B, num_states, num_states)
            log_backward[:, t] = torch.logsumexp(combined, dim=-1)

        return log_backward

    def viterbi_decode(self,
                      observations: torch.Tensor,
                      context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Neural HMM Viterbi 디코딩."""
        batch_size, seq_len, obs_dim = observations.shape
        device = observations.device

        # 관측 및 전이 확률 계산
        log_obs_probs = self.observation_model(observations)

        if self.transition_model is not None and context is not None:
            transition_probs = self.transition_model(context)
            log_transition_probs = torch.log(transition_probs + 1e-8)
        else:
            transition_probs = F.softmax(self.transition_matrix, dim=1)
            log_transition_probs = torch.log(transition_probs + 1e-8)
            log_transition_probs = log_transition_probs.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)

        initial_probs = F.softmax(self.initial_logits, dim=0)
        log_initial_probs = torch.log(initial_probs + 1e-8)

        # Viterbi 테이블
        log_delta = torch.full((batch_size, seq_len, self.num_states), float('-inf'), device=device)
        psi = torch.zeros((batch_size, seq_len, self.num_states), dtype=torch.long, device=device)

        # 초기화
        log_delta[:, 0] = log_initial_probs + log_obs_probs[:, 0]

        # Forward pass
        for t in range(1, seq_len):
            if self.transition_model is not None:
                transition_t = log_transition_probs[:, t-1]
            else:
                transition_t = log_transition_probs[:, 0]

            prev_delta = log_delta[:, t-1].unsqueeze(-1)  # (B, num_states, 1)
            transition_scores = prev_delta + transition_t  # (B, num_states, num_states)

            log_delta[:, t], psi[:, t] = torch.max(transition_scores, dim=1)
            log_delta[:, t] += log_obs_probs[:, t]

        # Backtrack
        states = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        states[:, -1] = torch.argmax(log_delta[:, -1], dim=1)

        for t in range(seq_len - 2, -1, -1):
            states[:, t] = psi[torch.arange(batch_size), t+1, states[:, t+1]]

        return states, log_delta

    def compute_likelihood(self,
                          observations: torch.Tensor,
                          context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Neural HMM log-likelihood 계산."""
        _, forward, _ = self.forward(observations, context)
        log_likelihood = torch.logsumexp(torch.log(forward[:, -1] + 1e-8), dim=-1)
        return log_likelihood


class ContextualNeuralHMM(NeuralHMM):
    """
    컨텍스트 인식 Neural HMM.

    언어적 컨텍스트, 음소 컨텍스트, 운율 정보 등을 활용하여
    더 정교한 음성 모델링을 수행합니다.
    """

    def __init__(self,
                 num_states: int,
                 observation_dim: int,
                 phoneme_vocab_size: int,
                 linguistic_context_dim: int = 64,
                 prosody_dim: int = 16,
                 **kwargs):

        self.phoneme_vocab_size = phoneme_vocab_size
        self.linguistic_context_dim = linguistic_context_dim
        self.prosody_dim = prosody_dim

        # 총 컨텍스트 차원
        total_context_dim = linguistic_context_dim + prosody_dim

        super().__init__(
            num_states=num_states,
            observation_dim=observation_dim,
            context_dim=total_context_dim,
            **kwargs
        )

        # 음소 임베딩
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, linguistic_context_dim)

        # 운율 특징 인코더
        self.prosody_encoder = nn.Linear(prosody_dim, prosody_dim)

    def encode_context(self,
                      phoneme_sequence: torch.Tensor,
                      prosody_features: torch.Tensor) -> torch.Tensor:
        """
        컨텍스트 특징 인코딩.

        Args:
            phoneme_sequence: 음소 시퀀스 (B, T)
            prosody_features: 운율 특징 (B, T, prosody_dim)

        Returns:
            context: 인코딩된 컨텍스트 (B, T, context_dim)
        """
        # 음소 임베딩
        phoneme_emb = self.phoneme_embedding(phoneme_sequence)

        # 운율 특징 인코딩
        prosody_emb = self.prosody_encoder(prosody_features)

        # 컨텍스트 결합
        context = torch.cat([phoneme_emb, prosody_emb], dim=-1)

        return context

    def forward_with_context(self,
                           observations: torch.Tensor,
                           phoneme_sequence: torch.Tensor,
                           prosody_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """컨텍스트를 포함한 forward pass."""
        context = self.encode_context(phoneme_sequence, prosody_features)
        return self.forward(observations, context)
