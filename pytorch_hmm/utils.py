import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, List, Dict, Tuple
import warnings


def create_transition_matrix(num_states: int, 
                           transition_type: str = "ergodic",
                           self_loop_prob: float = 0.5,
                           forward_prob: float = 0.4,
                           skip_prob: float = 0.1,
                           device: str = 'cpu') -> torch.Tensor:
    """
    다양한 유형의 transition matrix를 생성합니다.
    
    음성 합성에서 자주 사용되는 패턴들을 제공합니다:
    - ergodic: 모든 상태 간 전이 가능 (완전 연결)
    - left_to_right: 순차 진행만 가능 (Bakis model)
    - left_to_right_skip: 상태 건너뛰기 허용
    - circular: 순환 구조
    
    Args:
        num_states: 상태 수
        transition_type: 전이 타입 ('ergodic', 'left_to_right', 'left_to_right_skip', 'circular')
        self_loop_prob: 자기 자신 상태 유지 확률
        forward_prob: 다음 상태로 진행 확률
        skip_prob: 상태 건너뛰기 확률 (left_to_right_skip에서만 사용)
        device: 텐서 device
        
    Returns:
        Transition matrix of shape (num_states, num_states)
    """
    P = torch.zeros(num_states, num_states, device=device)
    
    if transition_type == "ergodic":
        # 모든 상태 간 전이 가능 (균등 분포)
        P = torch.ones(num_states, num_states, device=device)
        # 대각선 성분을 더 높게 설정 (상태 유지 경향)
        P = P + torch.eye(num_states, device=device) * self_loop_prob * num_states
        
    elif transition_type == "left_to_right":
        # Left-to-right model (Bakis model)
        for i in range(num_states):
            if i < num_states - 1:
                P[i, i] = self_loop_prob      # 현재 상태 유지
                P[i, i+1] = forward_prob      # 다음 상태로 진행
            else:
                P[i, i] = 1.0                 # 마지막 상태에서는 유지만 가능
                
    elif transition_type == "left_to_right_skip":
        # Left-to-right with state skipping
        for i in range(num_states):
            if i < num_states - 2:
                P[i, i] = self_loop_prob      # 현재 상태 유지
                P[i, i+1] = forward_prob      # 다음 상태로 진행
                P[i, i+2] = skip_prob         # 한 상태 건너뛰기
            elif i < num_states - 1:
                P[i, i] = self_loop_prob      # 현재 상태 유지
                P[i, i+1] = forward_prob      # 다음 상태로 진행
            else:
                P[i, i] = 1.0                 # 마지막 상태에서는 유지만 가능
                
    elif transition_type == "circular":
        # 순환 구조
        for i in range(num_states):
            P[i, i] = self_loop_prob          # 현재 상태 유지
            P[i, (i+1) % num_states] = forward_prob  # 다음 상태 (순환)
            
    else:
        raise ValueError(f"Unknown transition_type: {transition_type}")
    
    # 행별로 정규화하여 확률 분포로 만들기
    P = P / P.sum(dim=1, keepdim=True)
    
    return P


def create_left_to_right_matrix(num_states: int,
                               self_loop_prob: float = 0.7,
                               device: str = 'cpu') -> torch.Tensor:
    """
    Left-to-right HMM transition matrix를 생성합니다.
    
    음성 합성에서 가장 일반적으로 사용되는 패턴입니다.
    각 상태에서 자기 자신을 유지하거나 다음 상태로만 진행할 수 있습니다.
    
    Args:
        num_states: 상태 수
        self_loop_prob: 자기 상태 유지 확률 (0.0 ~ 1.0)
        device: 텐서 device
        
    Returns:
        Transition matrix of shape (num_states, num_states)
    """
    return create_transition_matrix(
        num_states=num_states,
        transition_type="left_to_right",
        self_loop_prob=self_loop_prob,
        forward_prob=1.0 - self_loop_prob,
        device=device
    )


def create_skip_state_matrix(num_states: int, 
                           self_loop_prob: float = 0.6, 
                           forward_prob: float = 0.3, 
                           skip_prob: float = 0.1,
                           max_skip: int = 2) -> torch.Tensor:
    """
    음성 인식을 위한 Skip-state transition matrix 생성
    빠른 발화나 음소 생략 현상을 모델링
    
    Args:
        num_states: 상태 개수
        self_loop_prob: 같은 상태 유지 확률
        forward_prob: 다음 상태로 이동 확률  
        skip_prob: 상태 건너뛰기 확률
        max_skip: 최대 건너뛸 수 있는 상태 수
    
    Returns:
        transition_matrix: (num_states, num_states) tensor
    """
    P = torch.zeros(num_states, num_states)
    
    for i in range(num_states):
        if i < num_states - 1:
            # Self-loop (같은 상태 유지)
            P[i, i] = self_loop_prob
            
            # Forward transition (순차 이동)
            P[i, i + 1] = forward_prob
            
            # Skip transitions (상태 건너뛰기)
            remaining_prob = skip_prob
            skip_weights = [1.0 / skip for skip in range(2, max_skip + 1)]
            total_weight = sum(skip_weights)
            
            for skip in range(2, min(max_skip + 1, num_states - i)):
                if i + skip < num_states:
                    skip_weight = skip_weights[skip - 2] / total_weight
                    P[i, i + skip] = remaining_prob * skip_weight
        else:
            # 마지막 상태는 자기 자신으로만 전이
            P[i, i] = 1.0
    
    # 확률 정규화
    row_sums = P.sum(dim=1, keepdim=True)
    P = P / (row_sums + 1e-8)
    
    return P


def create_phoneme_aware_transitions(phoneme_durations: List[float], 
                                   duration_variance: float = 0.2) -> torch.Tensor:
    """
    음소별 평균 지속시간을 고려한 전이 행렬
    
    Args:
        phoneme_durations: 각 음소의 평균 지속시간 (프레임 단위)
        duration_variance: 지속시간 분산 계수
    
    Returns:
        transition_matrix: (num_phonemes, num_phonemes) tensor
    """
    num_phonemes = len(phoneme_durations)
    P = torch.zeros(num_phonemes, num_phonemes)
    
    for i, duration in enumerate(phoneme_durations):
        if i < num_phonemes - 1:
            # 지속시간 기반 self-loop 확률 계산
            # 긴 음소일수록 높은 self-loop 확률
            base_self_prob = min(0.9, max(0.5, 1.0 - 1.0/duration))
            
            # 분산을 고려한 확률 조정
            variance = duration_variance * duration
            self_prob = np.clip(base_self_prob + np.random.normal(0, variance), 
                              0.3, 0.95)
            
            P[i, i] = self_prob
            P[i, i + 1] = 1.0 - self_prob
        else:
            P[i, i] = 1.0
    
    return P


def create_hierarchical_transitions(word_boundaries: List[int], 
                                  syllable_boundaries: List[int],
                                  phoneme_level_prob: float = 0.8,
                                  syllable_level_prob: float = 0.15,
                                  word_level_prob: float = 0.05) -> torch.Tensor:
    """
    계층적 언어 구조를 반영한 전이 행렬
    음소 -> 음절 -> 단어 레벨의 전이 확률
    
    Args:
        word_boundaries: 단어 경계 인덱스 리스트
        syllable_boundaries: 음절 경계 인덱스 리스트  
        phoneme_level_prob: 음소 레벨 전이 확률
        syllable_level_prob: 음절 레벨 전이 확률
        word_level_prob: 단어 레벨 전이 확률
    """
    total_phonemes = max(max(word_boundaries), max(syllable_boundaries)) + 1
    P = torch.zeros(total_phonemes, total_phonemes)
    
    for i in range(total_phonemes):
        if i < total_phonemes - 1:
            # 기본 음소 레벨 전이
            P[i, i] = 0.7  # self-loop
            P[i, i + 1] = phoneme_level_prob
            
            # 음절 경계에서의 특별한 전이
            if i in syllable_boundaries and i + 1 < total_phonemes:
                P[i, i + 1] += syllable_level_prob
            
            # 단어 경계에서의 특별한 전이  
            if i in word_boundaries and i + 1 < total_phonemes:
                P[i, i + 1] += word_level_prob
        else:
            P[i, i] = 1.0
    
    # 정규화
    row_sums = P.sum(dim=1, keepdim=True) 
    P = P / (row_sums + 1e-8)
    
    return P


class AdaptiveTransitionMatrix(nn.Module):
    """
    학습 가능한 적응형 전이 행렬
    컨텍스트에 따라 동적으로 전이 확률 조정
    """
    
    def __init__(self, num_states: int, context_dim: int = 128):
        super().__init__()
        self.num_states = num_states
        self.context_dim = context_dim
        
        # Base transition matrix (learnable)
        self.base_transition_logits = nn.Parameter(
            torch.randn(num_states, num_states) * 0.1
        )
        
        # Context-dependent modulation network
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_states * num_states),
            nn.Tanh()  # 제한된 범위의 조정
        )
        
    def forward(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        컨텍스트 기반 전이 행렬 생성
        
        Args:
            context: (batch_size, context_dim) 컨텍스트 벡터
                    예: speaker embedding, linguistic features 등
        
        Returns:
            transition_matrix: (batch_size, num_states, num_states)
        """
        base_transitions = F.softmax(self.base_transition_logits, dim=-1)
        
        if context is None:
            # 컨텍스트가 없으면 기본 전이 행렬 반환
            return base_transitions.unsqueeze(0)
        
        batch_size = context.shape[0]
        
        # 컨텍스트 기반 조정 계수 계산
        modulation = self.context_net(context)  # (batch_size, num_states^2)
        modulation = modulation.view(batch_size, self.num_states, self.num_states)
        
        # 기본 전이 행렬에 조정 적용
        # Log space에서 연산하여 numerical stability 확보
        log_base = torch.log(base_transitions + 1e-8)
        adjusted_logits = log_base.unsqueeze(0) + 0.1 * modulation
        
        # Softmax로 정규화
        adjusted_transitions = F.softmax(adjusted_logits, dim=-1)
        
        return adjusted_transitions


def create_duration_constrained_matrix(num_states: int,
                                     min_duration: int = 1,
                                     max_duration: Optional[int] = None,
                                     device: str = 'cpu') -> torch.Tensor:
    """
    지속시간 제약이 있는 transition matrix를 생성합니다.
    
    각 상태에서의 최소/최대 머무르는 시간을 제어할 수 있습니다.
    음성 합성에서 음소 지속시간 모델링에 유용합니다.
    
    Args:
        num_states: 상태 수
        min_duration: 각 상태에서의 최소 지속시간
        max_duration: 각 상태에서의 최대 지속시간 (None이면 제한 없음)
        device: 텐서 device
        
    Returns:
        Transition matrix of shape (num_states * duration_states, num_states * duration_states)
    """
    if max_duration is None:
        max_duration = min_duration * 3  # 기본값
    
    duration_states = max_duration
    total_states = num_states * duration_states
    
    P = torch.zeros(total_states, total_states, device=device)
    
    for state in range(num_states):
        for duration in range(duration_states):
            current_idx = state * duration_states + duration
            
            if duration < min_duration - 1:
                # 최소 지속시간 미달 시 강제로 다음 duration 상태로
                if duration < duration_states - 1:
                    P[current_idx, current_idx + 1] = 1.0
            elif duration < duration_states - 1:
                # 상태 유지 또는 다음 duration 상태로
                P[current_idx, current_idx + 1] = 0.7  # 상태 내 duration 증가
                
                # 다음 음소 상태로 전이 (duration 리셋)
                if state < num_states - 1:
                    next_state_idx = (state + 1) * duration_states
                    P[current_idx, next_state_idx] = 0.3
            else:
                # 최대 지속시간 도달 시 다음 상태로 강제 전이
                if state < num_states - 1:
                    next_state_idx = (state + 1) * duration_states
                    P[current_idx, next_state_idx] = 1.0
                else:
                    # 마지막 상태에서는 유지
                    P[current_idx, current_idx] = 1.0
    
    return P


def create_gaussian_observation_model(num_states: int,
                                    feature_dim: int,
                                    means: Optional[torch.Tensor] = None,
                                    covariances: Optional[torch.Tensor] = None,
                                    device: str = 'cpu') -> tuple:
    """
    가우시안 관측 모델의 파라미터를 생성합니다.
    
    Args:
        num_states: 상태 수
        feature_dim: 특징 차원
        means: 각 상태의 평균 (num_states, feature_dim)
        covariances: 각 상태의 공분산 (num_states, feature_dim, feature_dim)
        device: 텐서 device
        
    Returns:
        (means, covariances): 가우시안 파라미터들
    """
    if means is None:
        # 랜덤하게 초기화된 평균들
        means = torch.randn(num_states, feature_dim, device=device)
    
    if covariances is None:
        # 단위 행렬로 초기화된 공분산들
        covariances = torch.eye(feature_dim, device=device).unsqueeze(0).repeat(num_states, 1, 1)
    
    return means, covariances


def gaussian_log_likelihood(observations: torch.Tensor,
                          means: torch.Tensor,
                          covariances: torch.Tensor) -> torch.Tensor:
    """
    다변량 가우시안 분포의 log-likelihood를 계산합니다.
    
    Args:
        observations: 관측값 (B, T, D) 또는 (T, D)
        means: 각 상태의 평균 (K, D)
        covariances: 각 상태의 공분산 (K, D, D)
        
    Returns:
        Log-likelihood (B, T, K) 또는 (T, K)
    """
    if observations.dim() == 2:
        observations = observations.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, T, D = observations.shape
    K = means.shape[0]
    
    # 각 관측에 대해 각 상태의 log-likelihood 계산
    log_likelihoods = torch.zeros(B, T, K, device=observations.device)
    
    for k in range(K):
        mean_k = means[k]  # (D,)
        cov_k = covariances[k]  # (D, D)
        
        # 역공분산 행렬과 determinant 계산
        cov_inv = torch.inverse(cov_k)
        cov_det = torch.det(cov_k)
        
        # 정규화 상수
        norm_const = -0.5 * (D * np.log(2 * np.pi) + torch.log(cov_det))
        
        # 마할라노비스 거리
        diff = observations - mean_k  # (B, T, D)
        mahal_dist = torch.sum(diff.unsqueeze(-2) @ cov_inv @ diff.unsqueeze(-1), dim=(-2, -1))
        
        log_likelihoods[:, :, k] = norm_const - 0.5 * mahal_dist
    
    if squeeze_output:
        log_likelihoods = log_likelihoods.squeeze(0)
    
    return log_likelihoods


def align_sequences(reference_states: torch.Tensor,
                   observations: torch.Tensor,
                   hmm: 'HMMPyTorch') -> torch.Tensor:
    """
    참조 상태 시퀀스와 관측을 정렬합니다.
    
    음성 합성에서 텍스트와 음성 특징을 정렬할 때 사용합니다.
    
    Args:
        reference_states: 참조 상태 시퀀스 (T_ref,)
        observations: 관측 시퀀스 (T_obs, K)
        hmm: HMM 모델
        
    Returns:
        정렬된 상태 시퀀스 (T_obs,)
    """
    # Viterbi 알고리즘으로 최적 경로 찾기
    optimal_states, _ = hmm.viterbi_decode(observations)
    
    # DTW 유사한 정렬 수행 (간단한 버전)
    # 실제로는 더 정교한 정렬 알고리즘이 필요할 수 있습니다
    
    return optimal_states


def compute_state_durations(state_sequence: torch.Tensor) -> torch.Tensor:
    """
    상태 시퀀스에서 각 상태의 지속시간을 계산합니다.
    
    Args:
        state_sequence: 상태 시퀀스 (T,)
        
    Returns:
        각 고유 상태의 지속시간들
    """
    if len(state_sequence) == 0:
        return torch.tensor([])
    
    durations = []
    current_state = state_sequence[0]
    current_duration = 1
    
    for t in range(1, len(state_sequence)):
        if state_sequence[t] == current_state:
            current_duration += 1
        else:
            durations.append(current_duration)
            current_state = state_sequence[t]
            current_duration = 1
    
    durations.append(current_duration)  # 마지막 상태
    
    return torch.tensor(durations)


def interpolate_features(features: torch.Tensor,
                        source_durations: torch.Tensor,
                        target_durations: torch.Tensor) -> torch.Tensor:
    """
    상태별 지속시간에 따라 특징을 보간합니다.
    
    음성 합성에서 지속시간 조절 시 사용합니다.
    
    Args:
        features: 원본 특징 (T, D)
        source_durations: 원본 상태별 지속시간
        target_durations: 목표 상태별 지속시간
        
    Returns:
        보간된 특징 (T_new, D)
    """
    # 간단한 선형 보간 구현
    # 실제로는 더 정교한 보간 방법이 필요할 수 있습니다
    
    device = features.device
    T, D = features.shape
    
    # 상태별 특징 계산
    state_features = []
    start_idx = 0
    
    for duration in source_durations:
        end_idx = start_idx + duration
        state_feature = features[start_idx:end_idx].mean(dim=0)
        state_features.append(state_feature)
        start_idx = end_idx
    
    # 목표 지속시간에 맞춰 보간
    interpolated_features = []
    
    for i, target_duration in enumerate(target_durations):
        state_feature = state_features[i]
        # 각 상태를 목표 지속시간만큼 복제
        repeated_features = state_feature.unsqueeze(0).repeat(target_duration, 1)
        interpolated_features.append(repeated_features)
    
    return torch.cat(interpolated_features, dim=0)


# 새로운 고급 유틸리티 함수들

def create_attention_based_transitions(num_states: int, 
                                     attention_dim: int = 64) -> nn.Module:
    """
    어텐션 기반 동적 전이 행렬 생성기
    
    Args:
        num_states: HMM 상태 수
        attention_dim: 어텐션 차원
    
    Returns:
        어텐션 기반 전이 모듈
    """
    
    class AttentionTransition(nn.Module):
        def __init__(self, num_states, attention_dim):
            super().__init__()
            self.num_states = num_states
            self.attention_dim = attention_dim
            
            # 쿼리, 키, 값 네트워크
            self.query_net = nn.Linear(attention_dim, attention_dim)
            self.key_net = nn.Linear(attention_dim, attention_dim)
            self.value_net = nn.Linear(attention_dim, num_states * num_states)
            
            # 기본 전이 행렬
            self.base_transitions = nn.Parameter(
                torch.randn(num_states, num_states) * 0.1
            )
        
        def forward(self, context: torch.Tensor) -> torch.Tensor:
            """
            Args:
                context: (batch_size, seq_len, attention_dim)
            Returns:
                transition_matrices: (batch_size, seq_len, num_states, num_states)
            """
            batch_size, seq_len, _ = context.shape
            
            # 어텐션 계산
            queries = self.query_net(context)  # (B, T, D)
            keys = self.key_net(context)       # (B, T, D)
            values = self.value_net(context)   # (B, T, S*S)
            
            # Self-attention
            attention_weights = torch.softmax(
                torch.matmul(queries, keys.transpose(-2, -1)) / (self.attention_dim ** 0.5),
                dim=-1
            )
            
            # 가중 평균된 값
            attended_values = torch.matmul(attention_weights, values)  # (B, T, S*S)
            
            # 전이 행렬로 변형
            dynamic_transitions = attended_values.view(
                batch_size, seq_len, self.num_states, self.num_states
            )
            
            # 기본 전이와 결합
            base_expanded = self.base_transitions.unsqueeze(0).unsqueeze(0)
            combined_transitions = F.softmax(
                F.log_softmax(base_expanded, dim=-1) + 0.1 * dynamic_transitions,
                dim=-1
            )
            
            return combined_transitions
    
    return AttentionTransition(num_states, attention_dim)


def optimize_transition_matrix(transition_matrix: torch.Tensor, 
                             target_durations: Optional[List[float]] = None,
                             smoothness_weight: float = 0.1) -> torch.Tensor:
    """
    전이 행렬을 목표 지속시간과 스무스니스를 고려하여 최적화
    
    Args:
        transition_matrix: 원본 전이 행렬 (num_states, num_states)
        target_durations: 각 상태의 목표 지속시간
        smoothness_weight: 스무스니스 가중치
    
    Returns:
        최적화된 전이 행렬
    """
    num_states = transition_matrix.shape[0]
    optimized_matrix = transition_matrix.clone()
    
    if target_durations is not None:
        # 목표 지속시간에 맞춘 self-loop 확률 조정
        for i, target_duration in enumerate(target_durations):
            # 지속시간이 길수록 높은 self-loop 확률
            target_self_prob = 1.0 - 1.0 / max(target_duration, 1.0)
            target_self_prob = min(0.95, max(0.1, target_self_prob))
            
            # 부드럽게 조정
            current_self_prob = optimized_matrix[i, i].item()
            adjusted_self_prob = (1 - smoothness_weight) * current_self_prob + \
                               smoothness_weight * target_self_prob
            
            # 전이 확률 재정규화
            if i < num_states - 1:
                optimized_matrix[i, i] = adjusted_self_prob
                optimized_matrix[i, i + 1] = 1.0 - adjusted_self_prob
            else:
                optimized_matrix[i, i] = 1.0
    
    # 확률 정규화
    row_sums = optimized_matrix.sum(dim=1, keepdim=True)
    optimized_matrix = optimized_matrix / (row_sums + 1e-8)
    
    return optimized_matrix


def validate_transition_matrix(transition_matrix: torch.Tensor, 
                             tolerance: float = 1e-6) -> Dict[str, bool]:
    """
    전이 행렬의 유효성 검증
    
    Args:
        transition_matrix: 검증할 전이 행렬
        tolerance: 수치 오차 허용치
    
    Returns:
        검증 결과 딕셔너리
    """
    results = {}
    
    # 1. 확률 합 검증 (각 행의 합이 1인지)
    row_sums = transition_matrix.sum(dim=1)
    results['row_sums_valid'] = torch.all(torch.abs(row_sums - 1.0) < tolerance).item()
    
    # 2. 비음수 검증
    results['non_negative'] = torch.all(transition_matrix >= 0).item()
    
    # 3. 유한성 검증
    results['finite'] = torch.all(torch.isfinite(transition_matrix)).item()
    
    # 4. Left-to-right 패턴 검증 (선택적)
    upper_triangle = torch.triu(transition_matrix, diagonal=2)
    results['left_to_right_pattern'] = torch.all(upper_triangle == 0).item()
    
    # 5. 연결성 검증 (모든 상태가 도달 가능한지)
    # 간단한 버전: 전이 행렬의 거듭제곱으로 검증
    n = transition_matrix.shape[0]
    reachability_matrix = transition_matrix.clone()
    
    for _ in range(n - 1):
        reachability_matrix = torch.matmul(reachability_matrix, transition_matrix)
    
    # 마지막 상태를 제외하고 모든 상태가 마지막 상태에 도달 가능한지
    if n > 1:
        results['reachable_to_final'] = torch.all(reachability_matrix[:-1, -1] > tolerance).item()
    else:
        results['reachable_to_final'] = True
    
    return results


def benchmark_transition_operations(num_states_list: List[int], 
                                  num_trials: int = 100) -> Dict[str, Dict[int, float]]:
    """
    다양한 전이 행렬 연산의 성능 벤치마크
    
    Args:
        num_states_list: 테스트할 상태 수 리스트
        num_trials: 각 테스트의 반복 횟수
    
    Returns:
        벤치마크 결과
    """
    import time
    
    results = {
        'matrix_creation': {},
        'matrix_multiplication': {},
        'softmax_normalization': {},
        'validation': {}
    }
    
    for num_states in num_states_list:
        print(f"Benchmarking with {num_states} states...")
        
        # 1. 행렬 생성 벤치마크
        start_time = time.time()
        for _ in range(num_trials):
            create_transition_matrix(num_states, "left_to_right")
        creation_time = (time.time() - start_time) / num_trials
        results['matrix_creation'][num_states] = creation_time * 1000  # ms
        
        # 2. 행렬 곱셈 벤치마크
        P = create_transition_matrix(num_states, "left_to_right")
        start_time = time.time()
        for _ in range(num_trials):
            torch.matmul(P, P)
        mult_time = (time.time() - start_time) / num_trials
        results['matrix_multiplication'][num_states] = mult_time * 1000  # ms
        
        # 3. Softmax 정규화 벤치마크
        logits = torch.randn(num_states, num_states)
        start_time = time.time()
        for _ in range(num_trials):
            F.softmax(logits, dim=-1)
        softmax_time = (time.time() - start_time) / num_trials
        results['softmax_normalization'][num_states] = softmax_time * 1000  # ms
        
        # 4. 검증 벤치마크
        start_time = time.time()
        for _ in range(num_trials):
            validate_transition_matrix(P)
        validation_time = (time.time() - start_time) / num_trials
        results['validation'][num_states] = validation_time * 1000  # ms
    
    return results


# 음성 처리 특화 유틸리티

def create_prosody_aware_transitions(f0_contour: torch.Tensor,
                                   energy_contour: torch.Tensor,
                                   num_states: int) -> torch.Tensor:
    """
    운율 정보를 반영한 전이 행렬 생성
    
    Args:
        f0_contour: F0 윤곽 (time_steps,)
        energy_contour: 에너지 윤곽 (time_steps,)
        num_states: HMM 상태 수
    
    Returns:
        운율 정보가 반영된 전이 행렬
    """
    time_steps = f0_contour.shape[0]
    
    # 운율 특징 정규화
    f0_norm = (f0_contour - f0_contour.mean()) / (f0_contour.std() + 1e-8)
    energy_norm = (energy_contour - energy_contour.mean()) / (energy_contour.std() + 1e-8)
    
    # 운율 변화율 계산
    f0_delta = torch.diff(f0_norm, prepend=f0_norm[0:1])
    energy_delta = torch.diff(energy_norm, prepend=energy_norm[0:1])
    
    # 기본 left-to-right 행렬
    P_base = create_left_to_right_matrix(num_states)
    
    # 시간별 전이 행렬 조정
    P_sequence = []
    
    for t in range(time_steps):
        P_t = P_base.clone()
        
        # F0 상승 시 더 빠른 전이
        if f0_delta[t] > 0.5:
            for i in range(num_states - 1):
                P_t[i, i] *= 0.8      # self-loop 감소
                P_t[i, i + 1] *= 1.2  # forward 증가
        
        # 에너지 강조 시 상태 유지 증가
        if energy_norm[t] > 1.0:
            for i in range(num_states):
                P_t[i, i] *= 1.1
        
        # 정규화
        P_t = P_t / P_t.sum(dim=1, keepdim=True)
        P_sequence.append(P_t)
    
    return torch.stack(P_sequence)


def analyze_transition_patterns(state_sequences: List[torch.Tensor]) -> Dict[str, float]:
    """
    상태 전이 패턴 분석
    
    Args:
        state_sequences: 상태 시퀀스 리스트
    
    Returns:
        전이 패턴 통계
    """
    total_transitions = 0
    self_loops = 0
    forward_transitions = 0
    backward_transitions = 0
    skip_transitions = 0
    
    all_durations = []
    
    for seq in state_sequences:
        if len(seq) < 2:
            continue
        
        # 지속시간 분석
        durations = compute_state_durations(seq)
        all_durations.extend(durations.tolist())
        
        # 전이 분석
        for t in range(len(seq) - 1):
            current_state = seq[t].item()
            next_state = seq[t + 1].item()
            
            total_transitions += 1
            
            if next_state == current_state:
                self_loops += 1
            elif next_state == current_state + 1:
                forward_transitions += 1
            elif next_state < current_state:
                backward_transitions += 1
            elif next_state > current_state + 1:
                skip_transitions += 1
    
    # 통계 계산
    if total_transitions > 0:
        stats = {
            'self_loop_ratio': self_loops / total_transitions,
            'forward_ratio': forward_transitions / total_transitions,
            'backward_ratio': backward_transitions / total_transitions,
            'skip_ratio': skip_transitions / total_transitions,
            'avg_duration': np.mean(all_durations) if all_durations else 0,
            'std_duration': np.std(all_durations) if all_durations else 0,
            'total_transitions': total_transitions,
            'total_sequences': len(state_sequences)
        }
    else:
        stats = {key: 0.0 for key in ['self_loop_ratio', 'forward_ratio', 
                                     'backward_ratio', 'skip_ratio', 
                                     'avg_duration', 'std_duration']}
        stats.update({'total_transitions': 0, 'total_sequences': len(state_sequences)})
    
    return stats
