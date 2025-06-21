import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union, List, Dict
from torch.distributions import Gamma, Poisson, Normal


class DurationModel(nn.Module):
    """
    지속시간 분포 모델.
    
    다양한 확률 분포를 사용하여 상태별 지속시간을 모델링합니다.
    음성 합성에서 음소별 자연스러운 지속시간 패턴을 학습할 수 있습니다.
    
    Args:
        num_states: HMM 상태 수
        max_duration: 최대 지속시간
        distribution_type: 지속시간 분포 ('gamma', 'poisson', 'gaussian', 'neural')
        min_duration: 최소 지속시간
    """
    
    def __init__(self,
                 num_states: int,
                 max_duration: int = 50,
                 distribution_type: str = 'gamma',
                 min_duration: int = 1,
                 hidden_dim: int = 128):
        super(DurationModel, self).__init__()
        
        self.num_states = num_states
        self.max_duration = max_duration
        self.distribution_type = distribution_type
        self.min_duration = min_duration
        self.hidden_dim = hidden_dim
        
        if distribution_type == 'gamma':
            # 각 상태별 감마 분포 파라미터
            self.alpha_params = nn.Parameter(torch.ones(num_states))  # shape parameter
            self.beta_params = nn.Parameter(torch.ones(num_states))   # rate parameter
            
        elif distribution_type == 'poisson':
            # 각 상태별 포아송 분포 파라미터
            self.lambda_params = nn.Parameter(torch.ones(num_states) * 5)
            
        elif distribution_type == 'gaussian':
            # 각 상태별 가우시안 분포 파라미터
            self.mean_params = nn.Parameter(torch.ones(num_states) * 10)
            self.std_params = nn.Parameter(torch.ones(num_states))
            
        elif distribution_type == 'neural':
            # 신경망 기반 지속시간 모델
            self.duration_net = nn.Sequential(
                nn.Embedding(num_states, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max_duration),
                nn.LogSoftmax(dim=-1)
            )
        else:
            raise ValueError(f"Unknown distribution_type: {distribution_type}")
    
    def forward(self, state_indices: torch.Tensor, durations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        지속시간 확률 계산.
        
        Args:
            state_indices: 상태 인덱스 (batch_size, seq_len) 또는 (batch_size,)
            durations: 지속시간 (batch_size, seq_len) 또는 (batch_size,) - None이면 모든 지속시간에 대해 계산
            
        Returns:
            log_probs: 지속시간 로그 확률
        """
        if durations is None:
            # 모든 가능한 지속시간에 대한 확률 계산
            return self._compute_duration_distribution(state_indices)
        else:
            # 특정 지속시간에 대한 확률 계산
            return self._compute_duration_probability(state_indices, durations)
    
    def _compute_duration_distribution(self, state_indices: torch.Tensor) -> torch.Tensor:
        """모든 지속시간에 대한 분포 계산."""
        device = state_indices.device
        batch_size = state_indices.shape[0]
        
        if self.distribution_type == 'neural':
            # 신경망 기반 분포
            log_probs = self.duration_net(state_indices)  # (batch_size, max_duration)
            return log_probs
        else:
            # 파라미터 기반 분포
            durations = torch.arange(1, self.max_duration + 1, device=device).float()
            log_probs = torch.zeros(batch_size, self.max_duration, device=device)
            
            for i, state_idx in enumerate(state_indices):
                log_probs[i] = self._compute_parametric_distribution(state_idx.item(), durations)
            
            return log_probs
    
    def _compute_duration_probability(self, state_indices: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """특정 지속시간에 대한 확률 계산."""
        device = state_indices.device
        
        if self.distribution_type == 'neural':
            # 신경망 기반
            full_log_probs = self.duration_net(state_indices)  # (batch_size, max_duration)
            # 지속시간에 해당하는 확률 선택
            batch_indices = torch.arange(state_indices.shape[0], device=device)
            duration_indices = torch.clamp(durations - 1, 0, self.max_duration - 1)
            log_probs = full_log_probs[batch_indices, duration_indices]
            return log_probs
        else:
            # 파라미터 기반
            log_probs = torch.zeros_like(durations, dtype=torch.float, device=device)
            
            for i, (state_idx, duration) in enumerate(zip(state_indices, durations)):
                log_probs[i] = self._compute_parametric_distribution(
                    state_idx.item(), duration.float().unsqueeze(0)).squeeze()
            
            return log_probs
    
    def _compute_parametric_distribution(self, state_idx: int, durations: torch.Tensor) -> torch.Tensor:
        """파라미터 기반 지속시간 분포 계산."""
        device = durations.device
        
        if self.distribution_type == 'gamma':
            alpha = F.softplus(self.alpha_params[state_idx]) + 1e-6
            beta = F.softplus(self.beta_params[state_idx]) + 1e-6
            
            # 감마 분포 PDF
            log_probs = (alpha - 1) * torch.log(durations + 1e-8) - beta * durations
            log_probs -= torch.lgamma(alpha) - alpha * torch.log(beta)
            
        elif self.distribution_type == 'poisson':
            lambda_param = F.softplus(self.lambda_params[state_idx]) + 1e-6
            
            # 포아송 분포 PMF
            log_probs = durations * torch.log(lambda_param + 1e-8) - lambda_param
            log_probs -= torch.lgamma(durations + 1)
            
        elif self.distribution_type == 'gaussian':
            mean = F.softplus(self.mean_params[state_idx]) + self.min_duration
            std = F.softplus(self.std_params[state_idx]) + 1e-6
            
            # 가우시안 분포 PDF (절단된)
            log_probs = -0.5 * torch.log(2 * math.pi * std**2)
            log_probs = log_probs - 0.5 * ((durations - mean) / std)**2
        
        # 최소 지속시간 제약
        mask = durations >= self.min_duration
        log_probs = torch.where(mask, log_probs, torch.full_like(log_probs, float('-inf')))
        
        return log_probs
    
    def sample(self, state_indices: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """상태별 지속시간 샘플링."""
        device = state_indices.device
        batch_size = len(state_indices)
        
        if self.distribution_type == 'neural':
            # 신경망 기반 샘플링
            log_probs = self.duration_net(state_indices)
            probs = torch.exp(log_probs)
            
            samples = torch.multinomial(probs, num_samples, replacement=True) + 1
            return samples.squeeze(-1) if num_samples == 1 else samples
        else:
            # 파라미터 기반 샘플링
            samples = torch.zeros(batch_size, num_samples, device=device)
            
            for i, state_idx in enumerate(state_indices):
                if self.distribution_type == 'gamma':
                    alpha = F.softplus(self.alpha_params[state_idx]) + 1e-6
                    beta = F.softplus(self.beta_params[state_idx]) + 1e-6
                    gamma_dist = Gamma(alpha, beta)
                    samples[i] = gamma_dist.sample((num_samples,))
                    
                elif self.distribution_type == 'poisson':
                    lambda_param = F.softplus(self.lambda_params[state_idx]) + 1e-6
                    poisson_dist = Poisson(lambda_param)
                    samples[i] = poisson_dist.sample((num_samples,))
                    
                elif self.distribution_type == 'gaussian':
                    mean = F.softplus(self.mean_params[state_idx]) + self.min_duration
                    std = F.softplus(self.std_params[state_idx]) + 1e-6
                    normal_dist = Normal(mean, std)
                    samples[i] = torch.clamp(normal_dist.sample((num_samples,)), 
                                           min=self.min_duration, max=self.max_duration)
            
            # 최소 지속시간 보장
            samples = torch.clamp(samples, min=self.min_duration)
            return samples.squeeze(-1) if num_samples == 1 else samples


class SemiMarkovHMM(nn.Module):
    """
    Hidden Semi-Markov Model (HSMM).
    
    명시적 지속시간 모델링을 가진 HMM입니다.
    각 상태에서의 지속시간을 확률적으로 모델링하여
    더 자연스러운 시퀀스 생성이 가능합니다.
    
    음성 합성에서는:
    - 음소별 자연스러운 지속시간 모델링
    - 운율에 따른 지속시간 조절
    - 감정 표현을 위한 지속시간 변화
    
    Args:
        num_states: 상태 수
        observation_dim: 관측 차원
        max_duration: 최대 지속시간
        duration_distribution: 지속시간 분포 타입
        observation_model: 관측 모델 타입
    """
    
    def __init__(self,
                 num_states: int,
                 observation_dim: int,
                 max_duration: int = 50,
                 duration_distribution: str = 'gamma',
                 observation_model: str = 'gaussian',
                 min_duration: int = 1):
        super(SemiMarkovHMM, self).__init__()
        
        self.num_states = num_states
        self.observation_dim = observation_dim
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        # 지속시간 모델
        self.duration_model = DurationModel(
            num_states=num_states,
            max_duration=max_duration,
            distribution_type=duration_distribution,
            min_duration=min_duration
        )
        
        # 전이 확률 (상태 간 전이만, self-loop 없음)
        self.transition_logits = nn.Parameter(torch.randn(num_states, num_states))
        
        # 초기 상태 확률
        self.initial_logits = nn.Parameter(torch.zeros(num_states))
        
        # 관측 모델
        if observation_model == 'gaussian':
            self.observation_means = nn.Parameter(torch.randn(num_states, observation_dim))
            self.observation_logvars = nn.Parameter(torch.zeros(num_states, observation_dim))
        elif observation_model == 'neural':
            from .neural import NeuralObservationModel
            self.neural_obs_model = NeuralObservationModel(
                num_states=num_states,
                observation_dim=observation_dim,
                model_type='gaussian'
            )
        
        self.observation_model_type = observation_model
    
    def forward(self, 
                observations: torch.Tensor,
                state_sequence: Optional[torch.Tensor] = None,
                duration_sequence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        HSMM forward algorithm.
        
        Args:
            observations: 관측 시퀀스 (batch_size, seq_len, obs_dim)
            state_sequence: 상태 시퀀스 (batch_size, num_segments) - supervised learning
            duration_sequence: 지속시간 시퀀스 (batch_size, num_segments) - supervised learning
            
        Returns:
            results: forward probabilities, posteriors 등
        """
        if state_sequence is not None and duration_sequence is not None:
            # Supervised mode: compute likelihood given alignment
            return self._supervised_forward(observations, state_sequence, duration_sequence)
        else:
            # Unsupervised mode: marginalize over all possible alignments
            return self._unsupervised_forward(observations)
    
    def _supervised_forward(self, 
                           observations: torch.Tensor,
                           state_sequence: torch.Tensor,
                           duration_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Supervised forward (정렬이 주어진 경우)."""
        batch_size, seq_len, obs_dim = observations.shape
        device = observations.device
        
        # 관측 확률 계산
        log_obs_probs = self._compute_observation_logprobs(observations, state_sequence, duration_sequence)
        
        # 지속시간 확률 계산
        log_duration_probs = self.duration_model(state_sequence.flatten(), duration_sequence.flatten())
        log_duration_probs = log_duration_probs.view(batch_size, -1).sum(dim=1)
        
        # 전이 확률 계산
        log_transition_probs = self._compute_transition_logprobs(state_sequence)
        
        # 총 로그 확률
        total_log_prob = log_obs_probs + log_duration_probs + log_transition_probs
        
        return {
            'log_probability': total_log_prob,
            'log_observation': log_obs_probs,
            'log_duration': log_duration_probs,
            'log_transition': log_transition_probs
        }
    
    def _unsupervised_forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Unsupervised forward (정렬을 모르는 경우)."""
        batch_size, seq_len, obs_dim = observations.shape
        device = observations.device
        
        # Forward variable: alpha[t][s][d] = P(o_1:t, q_t=s, d_t=d)
        # t: 현재 시간, s: 현재 상태, d: 현재 상태에서의 지속시간
        log_alpha = torch.full((seq_len, self.num_states, self.max_duration), 
                              float('-inf'), device=device)
        
        # 초기화
        initial_probs = F.softmax(self.initial_logits, dim=0)
        log_initial_probs = torch.log(initial_probs + 1e-8)
        
        for s in range(self.num_states):
            for d in range(1, min(self.max_duration + 1, seq_len + 1)):
                if d <= seq_len:
                    # 처음 d개 프레임에 대한 관측 확률
                    log_obs_prob = self._compute_segment_observation_logprob(
                        observations[0, :d], s, d)
                    # 지속시간 확률
                    log_dur_prob = self.duration_model(
                        torch.tensor([s], device=device), 
                        torch.tensor([d], device=device))[0]
                    
                    log_alpha[d-1, s, d-1] = log_initial_probs[s] + log_obs_prob + log_dur_prob
        
        # Forward recursion
        transition_probs = F.softmax(self.transition_logits, dim=1)
        log_transition_probs = torch.log(transition_probs + 1e-8)
        
        for t in range(seq_len):
            for s in range(self.num_states):
                for d in range(1, min(self.max_duration + 1, t + 2)):
                    if t + 1 - d >= 0:
                        # 이전 시간 t-d에서 다른 상태로부터의 전이
                        log_alpha_sum = float('-inf')
                        
                        for prev_s in range(self.num_states):
                            if prev_s != s:  # Semi-Markov: no self-transitions
                                for prev_d in range(1, min(self.max_duration + 1, t - d + 2)):
                                    if t - d >= 0:
                                        prev_alpha = log_alpha[t - d, prev_s, prev_d - 1]
                                        if prev_alpha > float('-inf'):
                                            transition_prob = log_transition_probs[prev_s, s]
                                            log_alpha_sum = torch.logaddexp(
                                                log_alpha_sum, 
                                                prev_alpha + transition_prob)
                        
                        if log_alpha_sum > float('-inf'):
                            # 현재 세그먼트의 관측 확률
                            segment_start = max(0, t + 1 - d)
                            segment_end = t + 1
                            log_obs_prob = self._compute_segment_observation_logprob(
                                observations[0, segment_start:segment_end], s, d)
                            
                            # 지속시간 확률
                            log_dur_prob = self.duration_model(
                                torch.tensor([s], device=device),
                                torch.tensor([d], device=device))[0]
                            
                            log_alpha[t, s, d-1] = log_alpha_sum + log_obs_prob + log_dur_prob
        
        # 전체 시퀀스 확률 계산
        log_total_prob = float('-inf')
        for s in range(self.num_states):
            for d in range(1, min(self.max_duration + 1, seq_len + 1)):
                if seq_len - 1 < len(log_alpha) and d - 1 < log_alpha.shape[2]:
                    log_total_prob = torch.logaddexp(
                        log_total_prob, 
                        log_alpha[seq_len - 1, s, d - 1])
        
        return {
            'log_probability': log_total_prob,
            'forward_variables': log_alpha
        }
    
    def _compute_observation_logprobs(self, 
                                    observations: torch.Tensor,
                                    state_sequence: torch.Tensor,
                                    duration_sequence: torch.Tensor) -> torch.Tensor:
        """각 세그먼트의 관측 확률 계산."""
        batch_size, seq_len, obs_dim = observations.shape
        device = observations.device
        
        log_probs = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):
            current_time = 0
            for seg_idx in range(state_sequence.shape[1]):
                state = state_sequence[b, seg_idx]
                duration = duration_sequence[b, seg_idx]
                
                if current_time + duration <= seq_len:
                    segment_obs = observations[b, current_time:current_time + duration]
                    log_prob = self._compute_segment_observation_logprob(segment_obs, state.item(), duration.item())
                    log_probs[b] += log_prob
                    current_time += duration
                else:
                    break
        
        return log_probs
    
    def _compute_segment_observation_logprob(self, 
                                           segment_observations: torch.Tensor,
                                           state: int,
                                           duration: int) -> torch.Tensor:
        """단일 세그먼트의 관측 로그 확률 계산."""
        if self.observation_model_type == 'gaussian':
            mean = self.observation_means[state]
            logvar = self.observation_logvars[state]
            var = torch.exp(logvar)
            
            # 가우시안 로그 확률
            log_prob = -0.5 * torch.sum(logvar) - 0.5 * self.observation_dim * math.log(2 * math.pi)
            diff = segment_observations - mean
            log_prob -= 0.5 * torch.sum((diff ** 2) / var, dim=-1).sum()
            
        elif self.observation_model_type == 'neural':
            # 신경망 관측 모델
            state_indices = torch.full((segment_observations.shape[0],), state, 
                                     device=segment_observations.device)
            log_probs = self.neural_obs_model(
                segment_observations.unsqueeze(0), 
                state_indices.unsqueeze(0))
            log_prob = log_probs.sum()
        
        return log_prob
    
    def _compute_transition_logprobs(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """전이 확률 계산."""
        batch_size, num_segments = state_sequence.shape
        device = state_sequence.device
        
        transition_probs = F.softmax(self.transition_logits, dim=1)
        log_transition_probs = torch.log(transition_probs + 1e-8)
        
        log_probs = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):
            for seg_idx in range(1, num_segments):
                prev_state = state_sequence[b, seg_idx - 1]
                curr_state = state_sequence[b, seg_idx]
                log_probs[b] += log_transition_probs[prev_state, curr_state]
        
        return log_probs
    
    def viterbi_decode(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        HSMM Viterbi 디코딩으로 최적 상태-지속시간 시퀀스 찾기.
        
        Args:
            observations: 관측 시퀀스 (seq_len, obs_dim)
            
        Returns:
            states: 최적 상태 시퀀스
            durations: 최적 지속시간 시퀀스  
            log_prob: 최적 경로의 로그 확률
        """
        seq_len, obs_dim = observations.shape
        device = observations.device
        
        # Viterbi variables: delta[t][s][d] = max P(o_1:t, path ending with state s, duration d)
        log_delta = torch.full((seq_len, self.num_states, self.max_duration), 
                              float('-inf'), device=device)
        
        # Backtrack pointers
        psi_state = torch.zeros((seq_len, self.num_states, self.max_duration), 
                               dtype=torch.long, device=device)
        psi_duration = torch.zeros((seq_len, self.num_states, self.max_duration), 
                                  dtype=torch.long, device=device)
        
        # 초기화
        initial_probs = F.softmax(self.initial_logits, dim=0)
        log_initial_probs = torch.log(initial_probs + 1e-8)
        
        for s in range(self.num_states):
            for d in range(1, min(self.max_duration + 1, seq_len + 1)):
                if d <= seq_len:
                    log_obs_prob = self._compute_segment_observation_logprob(
                        observations[:d], s, d)
                    log_dur_prob = self.duration_model(
                        torch.tensor([s], device=device),
                        torch.tensor([d], device=device))[0]
                    
                    log_delta[d-1, s, d-1] = log_initial_probs[s] + log_obs_prob + log_dur_prob
        
        # Forward pass
        transition_probs = F.softmax(self.transition_logits, dim=1)
        log_transition_probs = torch.log(transition_probs + 1e-8)
        
        for t in range(seq_len):
            for s in range(self.num_states):
                for d in range(1, min(self.max_duration + 1, t + 2)):
                    if t + 1 - d >= 0:
                        best_prev_prob = float('-inf')
                        best_prev_state = 0
                        best_prev_duration = 1
                        
                        for prev_s in range(self.num_states):
                            if prev_s != s:
                                for prev_d in range(1, min(self.max_duration + 1, t - d + 2)):
                                    if t - d >= 0 and prev_d - 1 < self.max_duration:
                                        prev_prob = log_delta[t - d, prev_s, prev_d - 1]
                                        transition_prob = log_transition_probs[prev_s, s]
                                        total_prob = prev_prob + transition_prob
                                        
                                        if total_prob > best_prev_prob:
                                            best_prev_prob = total_prob
                                            best_prev_state = prev_s
                                            best_prev_duration = prev_d
                        
                        if best_prev_prob > float('-inf'):
                            segment_start = max(0, t + 1 - d)
                            segment_end = t + 1
                            log_obs_prob = self._compute_segment_observation_logprob(
                                observations[segment_start:segment_end], s, d)
                            log_dur_prob = self.duration_model(
                                torch.tensor([s], device=device),
                                torch.tensor([d], device=device))[0]
                            
                            log_delta[t, s, d-1] = best_prev_prob + log_obs_prob + log_dur_prob
                            psi_state[t, s, d-1] = best_prev_state
                            psi_duration[t, s, d-1] = best_prev_duration
        
        # 최적 경로 찾기
        best_final_prob = float('-inf')
        best_final_state = 0
        best_final_duration = 1
        
        for s in range(self.num_states):
            for d in range(1, min(self.max_duration + 1, seq_len + 1)):
                if seq_len - 1 < len(log_delta) and d - 1 < log_delta.shape[2]:
                    if log_delta[seq_len - 1, s, d - 1] > best_final_prob:
                        best_final_prob = log_delta[seq_len - 1, s, d - 1]
                        best_final_state = s
                        best_final_duration = d
        
        # Backtrack
        states = []
        durations = []
        
        current_t = seq_len - 1
        current_s = best_final_state
        current_d = best_final_duration
        
        while current_t >= 0:
            states.append(current_s)
            durations.append(current_d)
            
            if current_t - current_d >= 0:
                next_s = psi_state[current_t, current_s, current_d - 1].item()
                next_d = psi_duration[current_t, current_s, current_d - 1].item()
                current_t = current_t - current_d
                current_s = next_s
                current_d = next_d
            else:
                break
        
        states.reverse()
        durations.reverse()
        
        return torch.tensor(states, device=device), torch.tensor(durations, device=device), best_final_prob
    
    def sample(self, num_states: int, max_length: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        HSMM에서 시퀀스 샘플링.
        
        Args:
            num_states: 샘플링할 상태 수
            max_length: 최대 시퀀스 길이
            
        Returns:
            state_sequence: 상태 시퀀스
            duration_sequence: 지속시간 시퀀스
            observations: 관측 시퀀스
        """
        device = next(self.parameters()).device
        
        # 초기 상태 샘플링
        initial_probs = F.softmax(self.initial_logits, dim=0)
        current_state = torch.multinomial(initial_probs, 1).item()
        
        states = []
        durations = []
        observations = []
        total_length = 0
        
        transition_probs = F.softmax(self.transition_logits, dim=1)
        
        for _ in range(num_states):
            if total_length >= max_length:
                break
                
            # 현재 상태의 지속시간 샘플링
            duration = self.duration_model.sample(torch.tensor([current_state], device=device)).item()
            duration = min(duration, max_length - total_length)
            
            states.append(current_state)
            durations.append(duration)
            
            # 관측 샘플링
            if self.observation_model_type == 'gaussian':
                mean = self.observation_means[current_state]
                std = torch.exp(0.5 * self.observation_logvars[current_state])
                
                segment_obs = torch.normal(mean.unsqueeze(0).expand(int(duration), -1), 
                                         std.unsqueeze(0).expand(int(duration), -1))
                observations.append(segment_obs)
            
            total_length += duration
            
            # 다음 상태 샘플링
            if total_length < max_length:
                next_state = torch.multinomial(transition_probs[current_state], 1).item()
                current_state = next_state
        
        state_sequence = torch.tensor(states, device=device)
        duration_sequence = torch.tensor(durations, device=device)
        
        if observations:
            observation_sequence = torch.cat(observations, dim=0)
        else:
            observation_sequence = torch.zeros(0, self.observation_dim, device=device)
        
        return state_sequence, duration_sequence, observation_sequence


class AdaptiveDurationHSMM(SemiMarkovHMM):
    """
    적응형 지속시간 HSMM.
    
    컨텍스트에 따라 지속시간 분포가 변하는 HSMM입니다.
    음성 합성에서 화자, 감정, 발화 속도 등에 따른 
    지속시간 변화를 모델링할 때 유용합니다.
    """
    
    def __init__(self,
                 num_states: int,
                 observation_dim: int,
                 context_dim: int,
                 **kwargs):
        super().__init__(num_states, observation_dim, **kwargs)
        
        self.context_dim = context_dim
        
        # 컨텍스트 의존적 지속시간 모델
        self.context_duration_net = nn.Sequential(
            nn.Linear(context_dim + num_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.max_duration),
            nn.LogSoftmax(dim=-1)
        )
        
        # 상태 임베딩
        self.state_embedding = nn.Embedding(num_states, num_states)
    
    def compute_contextual_duration_probs(self, 
                                        state_indices: torch.Tensor,
                                        context: torch.Tensor) -> torch.Tensor:
        """컨텍스트 기반 지속시간 확률 계산."""
        # 상태 임베딩
        state_emb = self.state_embedding(state_indices)
        
        # 컨텍스트와 상태 결합
        combined_input = torch.cat([context, state_emb], dim=-1)
        
        # 지속시간 분포 계산
        log_duration_probs = self.context_duration_net(combined_input)
        
        return log_duration_probs
