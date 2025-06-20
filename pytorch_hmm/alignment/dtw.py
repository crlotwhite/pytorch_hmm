import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Callable


def compute_distance_matrix(x: torch.Tensor, 
                          y: torch.Tensor,
                          distance_fn: str = 'euclidean') -> torch.Tensor:
    """
    두 시퀀스 간의 거리 행렬을 계산합니다.
    
    Args:
        x: 첫 번째 시퀀스 (N, D)
        y: 두 번째 시퀀스 (M, D)  
        distance_fn: 거리 함수 ('euclidean', 'cosine', 'manhattan')
        
    Returns:
        distance_matrix: 거리 행렬 (N, M)
    """
    if distance_fn == 'euclidean':
        # Broadcasting을 이용한 효율적인 유클리드 거리 계산
        x_expanded = x.unsqueeze(1)  # (N, 1, D)
        y_expanded = y.unsqueeze(0)  # (1, M, D)
        distance_matrix = torch.norm(x_expanded - y_expanded, dim=2)
        
    elif distance_fn == 'cosine':
        # 코사인 거리 = 1 - 코사인 유사도
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        cosine_sim = torch.mm(x_norm, y_norm.t())
        distance_matrix = 1 - cosine_sim
        
    elif distance_fn == 'manhattan':
        # 맨하탄 거리 (L1 norm)
        x_expanded = x.unsqueeze(1)  # (N, 1, D)
        y_expanded = y.unsqueeze(0)  # (1, M, D)
        distance_matrix = torch.sum(torch.abs(x_expanded - y_expanded), dim=2)
        
    else:
        raise ValueError(f"Unknown distance function: {distance_fn}")
    
    return distance_matrix


def compute_dtw_path(distance_matrix: torch.Tensor,
                    step_pattern: str = 'symmetric') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DTW 최적 경로를 계산합니다.
    
    Args:
        distance_matrix: 거리 행렬 (N, M)
        step_pattern: 스텝 패턴 ('symmetric', 'asymmetric', 'rabiner_juang')
        
    Returns:
        path_i: 최적 경로의 i 인덱스 (path_length,)
        path_j: 최적 경로의 j 인덱스 (path_length,)
        cost_matrix: DTW 비용 행렬 (N, M)
    """
    N, M = distance_matrix.shape
    device = distance_matrix.device
    
    # DTW 비용 행렬 초기화
    cost_matrix = torch.full((N, M), float('inf'), device=device)
    cost_matrix[0, 0] = distance_matrix[0, 0]
    
    # 스텝 패턴에 따른 전이 규칙 정의
    if step_pattern == 'symmetric':
        # (i-1,j-1), (i-1,j), (i,j-1) 모두 가중치 1
        for i in range(N):
            for j in range(M):
                if i == 0 and j == 0:
                    continue
                    
                candidates = []
                if i > 0 and j > 0:
                    candidates.append(cost_matrix[i-1, j-1])  # 대각선
                if i > 0:
                    candidates.append(cost_matrix[i-1, j])    # 세로
                if j > 0:
                    candidates.append(cost_matrix[i, j-1])    # 가로
                
                if candidates:
                    cost_matrix[i, j] = distance_matrix[i, j] + min(candidates)
                    
    elif step_pattern == 'asymmetric':
        # (i-1,j-1) 가중치 2, (i-1,j) 가중치 1, (i,j-1) 가중치 1
        for i in range(N):
            for j in range(M):
                if i == 0 and j == 0:
                    continue
                    
                candidates = []
                if i > 0 and j > 0:
                    candidates.append(cost_matrix[i-1, j-1] + distance_matrix[i, j])  # 대각선
                if i > 0:
                    candidates.append(cost_matrix[i-1, j] + distance_matrix[i, j])    # 세로
                if j > 0:
                    candidates.append(cost_matrix[i, j-1] + distance_matrix[i, j])    # 가로
                
                if candidates:
                    cost_matrix[i, j] = min(candidates)
                    
    elif step_pattern == 'rabiner_juang':
        # Rabiner-Juang 스텝 패턴: 음성 인식에 최적화
        for i in range(N):
            for j in range(M):
                if i == 0 and j == 0:
                    continue
                    
                candidates = []
                if i > 0 and j > 0:
                    candidates.append(cost_matrix[i-1, j-1] + 2 * distance_matrix[i, j])
                if i > 0:
                    candidates.append(cost_matrix[i-1, j] + distance_matrix[i, j])
                if j > 0:
                    candidates.append(cost_matrix[i, j-1] + distance_matrix[i, j])
                
                if candidates:
                    cost_matrix[i, j] = min(candidates)
    
    # 백트래킹으로 최적 경로 찾기
    path_i = []
    path_j = []
    
    i, j = N - 1, M - 1
    while i > 0 or j > 0:
        path_i.append(i)
        path_j.append(j)
        
        # 이전 위치 찾기
        candidates = []
        if i > 0 and j > 0:
            candidates.append((cost_matrix[i-1, j-1], i-1, j-1))
        if i > 0:
            candidates.append((cost_matrix[i-1, j], i-1, j))
        if j > 0:
            candidates.append((cost_matrix[i, j-1], i, j-1))
        
        if candidates:
            _, prev_i, prev_j = min(candidates)
            i, j = prev_i, prev_j
    
    path_i.append(0)
    path_j.append(0)
    
    # 경로를 역순으로 정렬
    path_i = torch.tensor(path_i[::-1], device=device)
    path_j = torch.tensor(path_j[::-1], device=device)
    
    return path_i, path_j, cost_matrix


def dtw_distance(x: torch.Tensor, 
                y: torch.Tensor,
                distance_fn: str = 'euclidean',
                step_pattern: str = 'symmetric') -> torch.Tensor:
    """
    두 시퀀스 간의 DTW 거리를 계산합니다.
    
    Args:
        x: 첫 번째 시퀀스 (N, D)
        y: 두 번째 시퀀스 (M, D)
        distance_fn: 거리 함수
        step_pattern: 스텝 패턴
        
    Returns:
        dtw_dist: DTW 거리
    """
    distance_matrix = compute_distance_matrix(x, y, distance_fn)
    _, _, cost_matrix = compute_dtw_path(distance_matrix, step_pattern)
    
    N, M = distance_matrix.shape
    return cost_matrix[N-1, M-1]


def dtw_alignment(x: torch.Tensor,
                 y: torch.Tensor, 
                 distance_fn: str = 'euclidean',
                 step_pattern: str = 'symmetric') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DTW를 이용해 두 시퀀스를 정렬합니다.
    
    Args:
        x: 첫 번째 시퀀스 (N, D)
        y: 두 번째 시퀀스 (M, D)
        distance_fn: 거리 함수
        step_pattern: 스텝 패턴
        
    Returns:
        path_i: x의 정렬 인덱스 (path_length,)
        path_j: y의 정렬 인덱스 (path_length,)  
        total_cost: 총 DTW 비용
    """
    distance_matrix = compute_distance_matrix(x, y, distance_fn)
    path_i, path_j, cost_matrix = compute_dtw_path(distance_matrix, step_pattern)
    
    N, M = distance_matrix.shape
    total_cost = cost_matrix[N-1, M-1]
    
    return path_i, path_j, total_cost


class DTWAligner(nn.Module):
    """
    PyTorch nn.Module 기반 DTW 정렬기.
    
    음성 합성에서 텍스트-음성 정렬에 활용할 수 있습니다.
    differentiable DTW를 지원하여 end-to-end 학습이 가능합니다.
    
    Args:
        distance_fn: 거리 함수
        step_pattern: 스텝 패턴  
        bandwidth: 제한된 DTW를 위한 대역폭 (None이면 전체 DTW)
        soft_dtw: Soft DTW 사용 여부 (미분 가능)
        gamma: Soft DTW의 smoothing 파라미터
    """
    
    def __init__(self,
                 distance_fn: str = 'euclidean',
                 step_pattern: str = 'symmetric',
                 bandwidth: Optional[int] = None,
                 soft_dtw: bool = False,
                 gamma: float = 0.1):
        super(DTWAligner, self).__init__()
        
        self.distance_fn = distance_fn
        self.step_pattern = step_pattern
        self.bandwidth = bandwidth
        self.soft_dtw = soft_dtw
        self.gamma = gamma
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DTW 정렬 수행.
        
        Args:
            x: 첫 번째 시퀀스 (B, N, D) 또는 (N, D)
            y: 두 번째 시퀀스 (B, M, D) 또는 (M, D)
            
        Returns:
            path_i: x의 정렬 인덱스
            path_j: y의 정렬 인덱스
            total_cost: 총 DTW 비용
        """
        if x.dim() == 3:
            # 배치 처리
            batch_size = x.shape[0]
            all_paths_i = []
            all_paths_j = []
            all_costs = []
            
            for b in range(batch_size):
                path_i, path_j, cost = self._align_single(x[b], y[b])
                all_paths_i.append(path_i)
                all_paths_j.append(path_j)
                all_costs.append(cost)
            
            return all_paths_i, all_paths_j, torch.stack(all_costs)
        else:
            return self._align_single(x, y)
    
    def _align_single(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """단일 시퀀스 쌍에 대한 DTW 정렬."""
        if self.soft_dtw:
            return self._soft_dtw_align(x, y)
        else:
            return dtw_alignment(x, y, self.distance_fn, self.step_pattern)
    
    def _soft_dtw_align(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Soft DTW 정렬 (미분 가능).
        
        참고: "Soft-DTW: a Differentiable Loss Function for Time-Series" (Cuturi & Blondel, 2017)
        """
        distance_matrix = compute_distance_matrix(x, y, self.distance_fn)
        N, M = distance_matrix.shape
        device = distance_matrix.device
        
        # Soft DTW 비용 행렬
        cost_matrix = torch.full((N + 1, M + 1), float('inf'), device=device)
        cost_matrix[0, 0] = 0
        
        # Forward pass with smoothing
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                candidates = torch.stack([
                    cost_matrix[i-1, j-1],  # 대각선
                    cost_matrix[i-1, j],    # 세로
                    cost_matrix[i, j-1]     # 가로
                ])
                
                # Soft minimum 계산
                cost_matrix[i, j] = distance_matrix[i-1, j-1] - self.gamma * torch.logsumexp(-candidates / self.gamma, dim=0)
        
        # Soft DTW에서는 경로를 직접 계산하지 않고 근사값 반환
        # 실제 구현에서는 gradient 기반 백트래킹 필요
        total_cost = cost_matrix[N, M]
        
        # 간단한 근사 경로 (실제로는 더 정교한 방법 필요)
        path_i = torch.arange(N, device=device)
        path_j = torch.round(torch.linspace(0, M-1, N, device=device)).long()
        
        return path_i, path_j, total_cost


class ConstrainedDTWAligner(DTWAligner):
    """
    제약이 있는 DTW 정렬기.
    
    음성 합성에서 monotonic alignment나 bandwidth 제약을 적용할 때 사용합니다.
    """
    
    def __init__(self, 
                 bandwidth: int = 10,
                 monotonic: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.monotonic = monotonic
    
    def _align_single(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """제약이 적용된 DTW 정렬."""
        distance_matrix = compute_distance_matrix(x, y, self.distance_fn)
        N, M = distance_matrix.shape
        device = distance_matrix.device
        
        # 대역폭 제약 적용
        if self.bandwidth is not None:
            mask = torch.ones((N, M), device=device, dtype=torch.bool)
            for i in range(N):
                for j in range(M):
                    # 대각선에서 bandwidth 밖의 셀들을 마스킹
                    if abs(i - j * N / M) > self.bandwidth:
                        mask[i, j] = False
            
            distance_matrix = distance_matrix.masked_fill(~mask, float('inf'))
        
        return dtw_alignment(x, y, self.distance_fn, self.step_pattern)


# 음성 합성 특화 유틸리티 함수들
def phoneme_audio_alignment(phoneme_features: torch.Tensor,
                           audio_features: torch.Tensor,
                           phoneme_durations: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    음소 특징과 음성 특징 간의 정렬을 수행합니다.
    
    Args:
        phoneme_features: 음소 특징 (num_phonemes, feature_dim)
        audio_features: 음성 특징 (num_frames, feature_dim)
        phoneme_durations: 음소별 예상 지속시간 (num_phonemes,)
        
    Returns:
        alignment: 각 프레임에 대응하는 음소 인덱스 (num_frames,)
        boundaries: 음소 경계 프레임 인덱스 (num_phonemes + 1,)
    """
    aligner = DTWAligner(distance_fn='cosine', step_pattern='asymmetric')
    path_i, path_j, _ = aligner(phoneme_features, audio_features)
    
    # 정렬 결과를 음소 경계로 변환
    num_frames = audio_features.shape[0]
    num_phonemes = phoneme_features.shape[0]
    
    alignment = torch.zeros(num_frames, dtype=torch.long, device=audio_features.device)
    boundaries = [0]
    
    current_phoneme = 0
    for frame_idx in range(len(path_j)):
        frame = path_j[frame_idx].item()
        phoneme = path_i[frame_idx].item()
        
        if frame < num_frames:
            alignment[frame] = phoneme
            
        if phoneme > current_phoneme:
            boundaries.append(frame)
            current_phoneme = phoneme
    
    boundaries.append(num_frames)
    boundaries = torch.tensor(boundaries, device=audio_features.device)
    
    return alignment, boundaries


def extract_phoneme_durations(alignment: torch.Tensor, num_phonemes: int) -> torch.Tensor:
    """
    정렬 결과에서 음소별 지속시간을 추출합니다.
    
    Args:
        alignment: 프레임별 음소 인덱스 (num_frames,)
        num_phonemes: 총 음소 수
        
    Returns:
        durations: 음소별 지속시간 (num_phonemes,)
    """
    durations = torch.zeros(num_phonemes, dtype=torch.long, device=alignment.device)
    
    for phoneme_idx in range(num_phonemes):
        durations[phoneme_idx] = (alignment == phoneme_idx).sum()
    
    return durations
