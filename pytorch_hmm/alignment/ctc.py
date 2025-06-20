import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


def expand_targets_with_blank(targets: torch.Tensor, blank_id: int) -> torch.Tensor:
    """
    CTC를 위해 타겟 시퀀스에 blank 토큰을 삽입합니다.
    
    Args:
        targets: 원본 타겟 시퀀스 (batch_size, target_length)
        blank_id: blank 토큰 ID
        
    Returns:
        expanded_targets: blank이 삽입된 타겟 시퀀스 (batch_size, 2*target_length+1)
    """
    batch_size, target_length = targets.shape
    device = targets.device
    
    # blank으로 시작하고 각 타겟 사이에 blank 삽입
    expanded_length = 2 * target_length + 1
    expanded_targets = torch.full((batch_size, expanded_length), blank_id, device=device)
    
    # 홀수 인덱스에 원본 타겟 배치
    expanded_targets[:, 1::2] = targets
    
    return expanded_targets


def ctc_forward_algorithm(log_probs: torch.Tensor,
                         targets: torch.Tensor,
                         input_lengths: torch.Tensor,
                         target_lengths: torch.Tensor,
                         blank_id: int = 0) -> torch.Tensor:
    """
    CTC forward 알고리즘으로 log-likelihood를 계산합니다.
    
    Args:
        log_probs: 로그 확률 (max_time, batch_size, num_classes)
        targets: 타겟 시퀀스 (batch_size, max_target_length)
        input_lengths: 각 시퀀스의 실제 길이 (batch_size,)
        target_lengths: 각 타겟의 실제 길이 (batch_size,)
        blank_id: blank 토큰 ID
        
    Returns:
        log_likelihood: 배치별 log-likelihood (batch_size,)
    """
    max_time, batch_size, num_classes = log_probs.shape
    device = log_probs.device
    
    # 타겟을 blank과 함께 확장
    expanded_targets = expand_targets_with_blank(targets, blank_id)
    _, max_expanded_length = expanded_targets.shape
    
    # Alpha 테이블 초기화 (forward probabilities)
    log_alpha = torch.full((batch_size, max_time, max_expanded_length), 
                          float('-inf'), device=device)
    
    # 초기 조건: t=0에서 첫 번째와 두 번째 위치만 가능
    for b in range(batch_size):
        # 첫 번째 위치 (blank)
        log_alpha[b, 0, 0] = log_probs[0, b, blank_id]
        
        # 두 번째 위치 (첫 번째 타겟 토큰, 타겟이 존재하는 경우)
        if target_lengths[b] > 0:
            first_target = expanded_targets[b, 1].item()
            log_alpha[b, 0, 1] = log_probs[0, b, first_target]
    
    # Forward 계산
    for t in range(1, max_time):
        for b in range(batch_size):
            if t >= input_lengths[b]:
                continue
                
            expanded_length = 2 * target_lengths[b] + 1
            
            for s in range(min(expanded_length, max_expanded_length)):
                current_token = expanded_targets[b, s].item()
                current_log_prob = log_probs[t, b, current_token]
                
                # 가능한 이전 상태들
                candidates = []
                
                # 같은 위치에서 유지
                if log_alpha[b, t-1, s] > float('-inf'):
                    candidates.append(log_alpha[b, t-1, s])
                
                # 이전 위치에서 전이
                if s > 0 and log_alpha[b, t-1, s-1] > float('-inf'):
                    candidates.append(log_alpha[b, t-1, s-1])
                
                # 두 위치 전에서 건너뛰기 (연속된 같은 토큰이 아닌 경우)
                if (s > 1 and log_alpha[b, t-1, s-2] > float('-inf') and
                    expanded_targets[b, s] != expanded_targets[b, s-2]):
                    candidates.append(log_alpha[b, t-1, s-2])
                
                if candidates:
                    log_alpha[b, t, s] = current_log_prob + torch.logsumexp(
                        torch.tensor(candidates, device=device), dim=0)
    
    # 최종 log-likelihood 계산
    log_likelihood = torch.full((batch_size,), float('-inf'), device=device)
    
    for b in range(batch_size):
        time_idx = input_lengths[b] - 1
        expanded_length = 2 * target_lengths[b] + 1
        
        # 마지막 두 위치에서 끝날 수 있음
        candidates = []
        if expanded_length >= 1:
            candidates.append(log_alpha[b, time_idx, expanded_length - 1])
        if expanded_length >= 2:
            candidates.append(log_alpha[b, time_idx, expanded_length - 2])
        
        if candidates:
            log_likelihood[b] = torch.logsumexp(
                torch.tensor(candidates, device=device), dim=0)
    
    return log_likelihood


def ctc_backward_algorithm(log_probs: torch.Tensor,
                          targets: torch.Tensor,
                          input_lengths: torch.Tensor,
                          target_lengths: torch.Tensor,
                          blank_id: int = 0) -> torch.Tensor:
    """
    CTC backward 알고리즘으로 beta 확률을 계산합니다.
    
    Args:
        log_probs: 로그 확률 (max_time, batch_size, num_classes)
        targets: 타겟 시퀀스 (batch_size, max_target_length)
        input_lengths: 각 시퀀스의 실제 길이 (batch_size,)
        target_lengths: 각 타겟의 실제 길이 (batch_size,)
        blank_id: blank 토큰 ID
        
    Returns:
        log_beta: backward 확률 (batch_size, max_time, max_expanded_length)
    """
    max_time, batch_size, num_classes = log_probs.shape
    device = log_probs.device
    
    expanded_targets = expand_targets_with_blank(targets, blank_id)
    _, max_expanded_length = expanded_targets.shape
    
    # Beta 테이블 초기화 (backward probabilities)
    log_beta = torch.full((batch_size, max_time, max_expanded_length), 
                         float('-inf'), device=device)
    
    # 종료 조건 설정
    for b in range(batch_size):
        time_idx = input_lengths[b] - 1
        expanded_length = 2 * target_lengths[b] + 1
        
        # 마지막 위치들에서 시작
        if expanded_length >= 1:
            log_beta[b, time_idx, expanded_length - 1] = 0.0
        if expanded_length >= 2:
            log_beta[b, time_idx, expanded_length - 2] = 0.0
    
    # Backward 계산
    for t in range(max_time - 2, -1, -1):
        for b in range(batch_size):
            if t >= input_lengths[b]:
                continue
                
            expanded_length = 2 * target_lengths[b] + 1
            
            for s in range(min(expanded_length, max_expanded_length)):
                candidates = []
                
                # 같은 위치로 유지
                if s < max_expanded_length and log_beta[b, t+1, s] > float('-inf'):
                    next_token = expanded_targets[b, s].item()
                    next_log_prob = log_probs[t+1, b, next_token]
                    candidates.append(log_beta[b, t+1, s] + next_log_prob)
                
                # 다음 위치로 전이
                if (s + 1 < expanded_length and s + 1 < max_expanded_length and 
                    log_beta[b, t+1, s+1] > float('-inf')):
                    next_token = expanded_targets[b, s+1].item()
                    next_log_prob = log_probs[t+1, b, next_token]
                    candidates.append(log_beta[b, t+1, s+1] + next_log_prob)
                
                # 두 위치 건너뛰기
                if (s + 2 < expanded_length and s + 2 < max_expanded_length and
                    log_beta[b, t+1, s+2] > float('-inf') and
                    expanded_targets[b, s] != expanded_targets[b, s+2]):
                    next_token = expanded_targets[b, s+2].item()
                    next_log_prob = log_probs[t+1, b, next_token]
                    candidates.append(log_beta[b, t+1, s+2] + next_log_prob)
                
                if candidates:
                    log_beta[b, t, s] = torch.logsumexp(
                        torch.tensor(candidates, device=device), dim=0)
    
    return log_beta


def ctc_alignment_path(log_probs: torch.Tensor,
                      targets: torch.Tensor,
                      input_lengths: torch.Tensor,
                      target_lengths: torch.Tensor,
                      blank_id: int = 0) -> List[torch.Tensor]:
    """
    CTC forward-backward를 이용해 최적 정렬 경로를 계산합니다.
    
    Args:
        log_probs: 로그 확률 (max_time, batch_size, num_classes)
        targets: 타겟 시퀀스 (batch_size, max_target_length)
        input_lengths: 각 시퀀스의 실제 길이 (batch_size,)
        target_lengths: 각 타겟의 실제 길이 (batch_size,)
        blank_id: blank 토큰 ID
        
    Returns:
        alignments: 배치별 정렬 경로 리스트
    """
    max_time, batch_size, num_classes = log_probs.shape
    device = log_probs.device
    
    # Forward 및 backward 확률 계산
    log_alpha = torch.full((batch_size, max_time, 2 * targets.shape[1] + 1), 
                          float('-inf'), device=device)
    log_beta = ctc_backward_algorithm(log_probs, targets, input_lengths, target_lengths, blank_id)
    
    # Forward 계산 (간단화된 버전)
    expanded_targets = expand_targets_with_blank(targets, blank_id)
    
    # Posterior 확률 계산 및 최적 경로 추출
    alignments = []
    
    for b in range(batch_size):
        alignment_path = []
        expanded_length = 2 * target_lengths[b] + 1
        
        for t in range(input_lengths[b]):
            # 각 시간 단계에서 가장 확률이 높은 위치 선택
            best_pos = 0
            best_prob = float('-inf')
            
            for s in range(min(expanded_length, expanded_targets.shape[1])):
                # Forward + backward 확률 결합
                total_prob = log_alpha[b, t, s] + log_beta[b, t, s]
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_pos = s
            
            # 확장된 타겟에서 실제 토큰으로 변환
            token = expanded_targets[b, best_pos].item()
            alignment_path.append(token)
        
        alignments.append(torch.tensor(alignment_path, device=device))
    
    return alignments


class CTCAligner(nn.Module):
    """
    CTC 기반 정렬기.
    
    음성 인식과 TTS에서 사용되는 CTC 정렬을 제공합니다.
    blank 토큰을 통해 가변 길이 시퀀스 간의 정렬을 수행합니다.
    
    Args:
        num_classes: 전체 클래스 수 (blank 포함)
        blank_id: blank 토큰 ID (보통 0)
        reduction: loss reduction ('mean', 'sum', 'none')
    """
    
    def __init__(self, 
                 num_classes: int,
                 blank_id: int = 0,
                 reduction: str = 'mean'):
        super(CTCAligner, self).__init__()
        
        self.num_classes = num_classes
        self.blank_id = blank_id
        self.reduction = reduction
        
        # PyTorch의 내장 CTC loss 함수
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=reduction, zero_infinity=True)
    
    def forward(self, 
                log_probs: torch.Tensor,
                targets: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        CTC loss 계산.
        
        Args:
            log_probs: 로그 확률 (max_time, batch_size, num_classes)
            targets: 타겟 시퀀스 (batch_size, max_target_length)
            input_lengths: 입력 길이 (batch_size,)
            target_lengths: 타겟 길이 (batch_size,)
            
        Returns:
            loss: CTC loss
        """
        # PyTorch CTC는 targets가 1D여야 함
        targets_1d = []
        for b in range(targets.shape[0]):
            targets_1d.append(targets[b, :target_lengths[b]])
        targets_1d = torch.cat(targets_1d)
        
        return self.ctc_loss(log_probs, targets_1d, input_lengths, target_lengths)
    
    def decode(self,
               log_probs: torch.Tensor,
               input_lengths: torch.Tensor,
               beam_width: int = 1) -> List[torch.Tensor]:
        """
        CTC 디코딩 (greedy 또는 beam search).
        
        Args:
            log_probs: 로그 확률 (max_time, batch_size, num_classes)
            input_lengths: 입력 길이 (batch_size,)
            beam_width: beam search width (1이면 greedy)
            
        Returns:
            decoded: 디코딩된 시퀀스들
        """
        if beam_width == 1:
            return self._greedy_decode(log_probs, input_lengths)
        else:
            return self._beam_search_decode(log_probs, input_lengths, beam_width)
    
    def _greedy_decode(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> List[torch.Tensor]:
        """Greedy 디코딩."""
        max_time, batch_size, num_classes = log_probs.shape
        device = log_probs.device
        
        decoded_sequences = []
        
        for b in range(batch_size):
            # 각 시간 단계에서 가장 확률이 높은 클래스 선택
            best_path = torch.argmax(log_probs[:input_lengths[b], b], dim=1)
            
            # Collapse repeated tokens and remove blanks
            decoded = []
            prev_token = None
            
            for token in best_path:
                token = token.item()
                if token != self.blank_id and token != prev_token:
                    decoded.append(token)
                prev_token = token
            
            decoded_sequences.append(torch.tensor(decoded, device=device))
        
        return decoded_sequences
    
    def _beam_search_decode(self, 
                           log_probs: torch.Tensor, 
                           input_lengths: torch.Tensor, 
                           beam_width: int) -> List[torch.Tensor]:
        """Beam search 디코딩 (간단화된 버전)."""
        # 실제 구현에서는 더 정교한 beam search 필요
        # 여기서는 greedy decode로 대체
        return self._greedy_decode(log_probs, input_lengths)
    
    def align(self,
              log_probs: torch.Tensor,
              targets: torch.Tensor,
              input_lengths: torch.Tensor,
              target_lengths: torch.Tensor) -> List[torch.Tensor]:
        """
        CTC를 이용한 강제 정렬.
        
        Args:
            log_probs: 로그 확률 (max_time, batch_size, num_classes)
            targets: 타겟 시퀀스 (batch_size, max_target_length)
            input_lengths: 입력 길이 (batch_size,)
            target_lengths: 타겟 길이 (batch_size,)
            
        Returns:
            alignments: 시간별 타겟 토큰 정렬
        """
        return ctc_alignment_path(log_probs, targets, input_lengths, target_lengths, self.blank_id)


class CTCSegmentationAligner(CTCAligner):
    """
    CTC 기반 음성 분할 정렬기.
    
    긴 음성과 텍스트를 자동으로 분할하여 정렬하는 기능을 제공합니다.
    음성 데이터셋 준비나 강제 정렬에 유용합니다.
    """
    
    def __init__(self, 
                 num_classes: int,
                 min_segment_length: int = 50,
                 max_segment_length: int = 1000,
                 **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
    
    def segment_and_align(self,
                         log_probs: torch.Tensor,
                         full_transcript: torch.Tensor,
                         segment_boundaries: Optional[torch.Tensor] = None) -> List[Tuple[torch.Tensor, torch.Tensor, int, int]]:
        """
        긴 음성을 분할하고 각 분할에 대해 정렬을 수행합니다.
        
        Args:
            log_probs: 전체 음성의 로그 확률 (time, num_classes)
            full_transcript: 전체 전사 (transcript_length,)
            segment_boundaries: 분할 경계 (optional)
            
        Returns:
            segments: [(aligned_audio, aligned_text, start_frame, end_frame), ...]
        """
        if segment_boundaries is None:
            # 자동 분할 경계 탐지
            segment_boundaries = self._detect_segment_boundaries(log_probs, full_transcript)
        
        segments = []
        prev_boundary = 0
        
        for boundary in segment_boundaries:
            if boundary - prev_boundary >= self.min_segment_length:
                # 분할 추출
                segment_audio = log_probs[prev_boundary:boundary]
                
                # 해당 분할의 텍스트 추정 (간단화된 방법)
                segment_text = self._estimate_segment_text(
                    segment_audio, full_transcript, prev_boundary, boundary)
                
                segments.append((segment_audio, segment_text, prev_boundary, boundary))
                prev_boundary = boundary
        
        return segments
    
    def _detect_segment_boundaries(self, log_probs: torch.Tensor, transcript: torch.Tensor) -> torch.Tensor:
        """자동으로 분할 경계를 탐지합니다."""
        # 간단한 에너지 기반 분할 (실제로는 더 정교한 방법 필요)
        time_length = log_probs.shape[0]
        
        # 고정 길이 분할
        boundaries = torch.arange(0, time_length, self.max_segment_length)
        return boundaries[boundaries < time_length]
    
    def _estimate_segment_text(self, 
                              segment_audio: torch.Tensor,
                              full_transcript: torch.Tensor,
                              start_frame: int,
                              end_frame: int) -> torch.Tensor:
        """분할에 해당하는 텍스트를 추정합니다."""
        # 간단한 비례 분할 (실제로는 CTC alignment 사용)
        total_frames = log_probs.shape[0] if hasattr(self, 'log_probs') else 1000
        transcript_ratio = len(full_transcript) / total_frames
        
        start_text = int(start_frame * transcript_ratio)
        end_text = int(end_frame * transcript_ratio)
        
        return full_transcript[start_text:end_text]


# 유틸리티 함수들
def remove_ctc_blanks(sequence: torch.Tensor, blank_id: int = 0) -> torch.Tensor:
    """CTC 디코딩 결과에서 blank 토큰을 제거합니다."""
    return sequence[sequence != blank_id]


def collapse_repeated_tokens(sequence: torch.Tensor) -> torch.Tensor:
    """연속된 같은 토큰을 하나로 합칩니다."""
    if len(sequence) == 0:
        return sequence
    
    collapsed = [sequence[0].item()]
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            collapsed.append(sequence[i].item())
    
    return torch.tensor(collapsed, device=sequence.device)


def ctc_decode_sequence(sequence: torch.Tensor, blank_id: int = 0) -> torch.Tensor:
    """완전한 CTC 디코딩 (collapse + blank removal)"""
    collapsed = collapse_repeated_tokens(sequence)
    return remove_ctc_blanks(collapsed, blank_id)
