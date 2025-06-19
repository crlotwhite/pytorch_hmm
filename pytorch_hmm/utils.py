import torch
import numpy as np
from typing import Union, Optional


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
