"""
Speech synthesis quality evaluation metrics.

This module provides various metrics commonly used for evaluating
speech synthesis and alignment quality:

- MCD: Mel-Cepstral Distortion
- F0 RMSE: Fundamental frequency root mean square error
- Alignment accuracy metrics
- Spectral distortion measures
- Duration modeling evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import math


def mel_cepstral_distortion(mfcc_true: torch.Tensor, 
                          mfcc_pred: torch.Tensor,
                          exclude_c0: bool = True) -> torch.Tensor:
    """
    Mel-Cepstral Distortion (MCD) 계산.
    
    음성 합성 품질 평가의 가장 중요한 객관적 지표 중 하나입니다.
    두 MFCC 특징 간의 유클리드 거리를 측정합니다.
    
    Args:
        mfcc_true: 참조 MFCC (B, T, D) 또는 (T, D)
        mfcc_pred: 예측 MFCC (B, T, D) 또는 (T, D)
        exclude_c0: c0 계수 제외 여부 (일반적으로 제외)
        
    Returns:
        mcd: MCD 값 (dB 단위)
    """
    if mfcc_true.dim() == 2:
        mfcc_true = mfcc_true.unsqueeze(0)
        mfcc_pred = mfcc_pred.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # c0 계수 제외 (일반적으로 에너지 성분 제외)
    if exclude_c0:
        mfcc_true = mfcc_true[:, :, 1:]
        mfcc_pred = mfcc_pred[:, :, 1:]
    
    # 길이 맞추기 (짧은 쪽에 맞춤)
    min_length = min(mfcc_true.shape[1], mfcc_pred.shape[1])
    mfcc_true = mfcc_true[:, :min_length]
    mfcc_pred = mfcc_pred[:, :min_length]
    
    # MCD 계산: √2 * 10/ln(10) * √(Σ(c_true - c_pred)²)
    diff = mfcc_true - mfcc_pred
    squared_diff = torch.sum(diff ** 2, dim=-1)  # (B, T)
    
    # MCD 공식
    K = math.sqrt(2) * 10 / math.log(10)  # ≈ 6.14
    mcd = K * torch.sqrt(squared_diff)
    
    # 시간 평균
    mcd = torch.mean(mcd, dim=1)  # (B,)
    
    if squeeze_output:
        mcd = mcd.squeeze(0)
    
    return mcd


def f0_root_mean_square_error(f0_true: torch.Tensor,
                             f0_pred: torch.Tensor,
                             voiced_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    F0 (기본 주파수) RMSE 계산.
    
    운율 모델링 품질을 평가하는 중요한 지표입니다.
    유성음 구간에서만 계산하는 것이 일반적입니다.
    
    Args:
        f0_true: 참조 F0 (B, T) 또는 (T,)
        f0_pred: 예측 F0 (B, T) 또는 (T,)
        voiced_mask: 유성음 마스크 (B, T) 또는 (T,) - None이면 F0 > 0인 구간 사용
        
    Returns:
        f0_rmse: F0 RMSE (Hz 단위)
    """
    if f0_true.dim() == 1:
        f0_true = f0_true.unsqueeze(0)
        f0_pred = f0_pred.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 길이 맞추기
    min_length = min(f0_true.shape[1], f0_pred.shape[1])
    f0_true = f0_true[:, :min_length]
    f0_pred = f0_pred[:, :min_length]
    
    # 유성음 마스크 설정
    if voiced_mask is None:
        # F0가 0보다 큰 구간을 유성음으로 간주
        voiced_mask = (f0_true > 0) & (f0_pred > 0)
    else:
        if voiced_mask.dim() == 1:
            voiced_mask = voiced_mask.unsqueeze(0)
        voiced_mask = voiced_mask[:, :min_length]
    
    # 유성음 구간에서만 RMSE 계산
    diff = f0_true - f0_pred
    squared_diff = diff ** 2
    
    # 마스킹된 구간만 계산
    masked_squared_diff = squared_diff * voiced_mask.float()
    
    # 배치별 RMSE
    rmse_values = []
    for b in range(f0_true.shape[0]):
        voiced_frames = voiced_mask[b].sum()
        if voiced_frames > 0:
            mse = masked_squared_diff[b].sum() / voiced_frames
            rmse = torch.sqrt(mse)
        else:
            rmse = torch.tensor(0.0, device=f0_true.device)
        rmse_values.append(rmse)
    
    f0_rmse = torch.stack(rmse_values)
    
    if squeeze_output:
        f0_rmse = f0_rmse.squeeze(0)
    
    return f0_rmse


def log_f0_rmse(f0_true: torch.Tensor,
               f0_pred: torch.Tensor,
               voiced_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Log F0 RMSE 계산.
    
    F0의 로그 값에 대한 RMSE로, 인간의 음높이 지각에 더 가깝습니다.
    
    Args:
        f0_true: 참조 F0 (B, T) 또는 (T,)
        f0_pred: 예측 F0 (B, T) 또는 (T,)
        voiced_mask: 유성음 마스크
        
    Returns:
        log_f0_rmse: Log F0 RMSE
    """
    # 0값 방지를 위한 작은 값 추가
    eps = 1e-8
    log_f0_true = torch.log(f0_true + eps)
    log_f0_pred = torch.log(f0_pred + eps)
    
    return f0_root_mean_square_error(log_f0_true, log_f0_pred, voiced_mask)


def alignment_accuracy(predicted_alignment: torch.Tensor,
                      ground_truth_alignment: torch.Tensor,
                      tolerance: int = 0) -> torch.Tensor:
    """
    정렬 정확도 계산.
    
    HMM이나 DTW로 예측한 정렬과 ground truth 정렬 간의 정확도를 측정합니다.
    
    Args:
        predicted_alignment: 예측된 정렬 (T,) - 각 프레임의 상태/음소 인덱스
        ground_truth_alignment: 정답 정렬 (T,)
        tolerance: 허용 오차 (프레임 단위)
        
    Returns:
        accuracy: 정렬 정확도 (0~1)
    """
    # 길이 맞추기
    min_length = min(len(predicted_alignment), len(ground_truth_alignment))
    pred = predicted_alignment[:min_length]
    gt = ground_truth_alignment[:min_length]
    
    if tolerance == 0:
        # 정확히 일치하는 경우만 정확
        correct = (pred == gt).float()
    else:
        # 허용 오차 내에서 일치하는 경우
        correct = torch.zeros_like(pred, dtype=torch.float)
        for i in range(len(pred)):
            # 허용 오차 범위 내의 GT 값들과 비교
            start_idx = max(0, i - tolerance)
            end_idx = min(len(gt), i + tolerance + 1)
            if pred[i] in gt[start_idx:end_idx]:
                correct[i] = 1.0
    
    accuracy = correct.mean()
    return accuracy


def boundary_accuracy(predicted_boundaries: torch.Tensor,
                     ground_truth_boundaries: torch.Tensor,
                     tolerance: int = 2) -> Dict[str, torch.Tensor]:
    """
    음소 경계 정확도 계산.
    
    음소나 단어 경계 탐지 성능을 평가합니다.
    
    Args:
        predicted_boundaries: 예측된 경계 시점들 (num_boundaries,)
        ground_truth_boundaries: 정답 경계 시점들 (num_boundaries,)
        tolerance: 허용 오차 (프레임 단위)
        
    Returns:
        metrics: precision, recall, f1 포함한 딕셔너리
    """
    device = predicted_boundaries.device
    
    # 정확한 경계 탐지 (허용 오차 내)
    true_positives = 0
    for gt_boundary in ground_truth_boundaries:
        # 허용 오차 내에 예측 경계가 있는지 확인
        distances = torch.abs(predicted_boundaries - gt_boundary)
        if torch.any(distances <= tolerance):
            true_positives += 1
    
    false_positives = len(predicted_boundaries) - true_positives
    false_negatives = len(ground_truth_boundaries) - true_positives
    
    # 메트릭 계산
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': torch.tensor(precision, device=device),
        'recall': torch.tensor(recall, device=device),
        'f1': torch.tensor(f1, device=device),
        'true_positives': torch.tensor(true_positives, device=device),
        'false_positives': torch.tensor(false_positives, device=device),
        'false_negatives': torch.tensor(false_negatives, device=device)
    }


def duration_accuracy(predicted_durations: torch.Tensor,
                     ground_truth_durations: torch.Tensor,
                     relative_tolerance: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    지속시간 예측 정확도 계산.
    
    음소별 지속시간 모델링 성능을 평가합니다.
    
    Args:
        predicted_durations: 예측된 지속시간들 (num_phonemes,)
        ground_truth_durations: 정답 지속시간들 (num_phonemes,)
        relative_tolerance: 상대적 허용 오차 (0.2 = 20%)
        
    Returns:
        metrics: MAE, RMSE, 상대 오차 등 포함한 딕셔너리
    """
    # 길이 맞추기
    min_length = min(len(predicted_durations), len(ground_truth_durations))
    pred = predicted_durations[:min_length].float()
    gt = ground_truth_durations[:min_length].float()
    
    # 절대 오차
    absolute_error = torch.abs(pred - gt)
    mae = absolute_error.mean()
    
    # 제곱 오차
    squared_error = (pred - gt) ** 2
    rmse = torch.sqrt(squared_error.mean())
    
    # 상대 오차
    relative_error = absolute_error / (gt + 1e-8)  # 0 나누기 방지
    mean_relative_error = relative_error.mean()
    
    # 허용 오차 내 정확도
    tolerance_mask = relative_error <= relative_tolerance
    accuracy_within_tolerance = tolerance_mask.float().mean()
    
    # 상관계수
    pred_centered = pred - pred.mean()
    gt_centered = gt - gt.mean()
    correlation = torch.sum(pred_centered * gt_centered) / (
        torch.sqrt(torch.sum(pred_centered ** 2)) * torch.sqrt(torch.sum(gt_centered ** 2)) + 1e-8)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mean_relative_error': mean_relative_error,
        'accuracy_within_tolerance': accuracy_within_tolerance,
        'correlation': correlation
    }


def spectral_distortion(spec_true: torch.Tensor,
                       spec_pred: torch.Tensor,
                       distance_type: str = 'euclidean') -> torch.Tensor:
    """
    스펙트럼 왜곡 측정.
    
    멜 스펙트로그램이나 기타 스펙트럼 특징 간의 거리를 계산합니다.
    
    Args:
        spec_true: 참조 스펙트럼 (B, T, D) 또는 (T, D)
        spec_pred: 예측 스펙트럼 (B, T, D) 또는 (T, D)
        distance_type: 거리 타입 ('euclidean', 'cosine', 'kl_divergence')
        
    Returns:
        distortion: 스펙트럼 왜곡 값
    """
    if spec_true.dim() == 2:
        spec_true = spec_true.unsqueeze(0)
        spec_pred = spec_pred.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 길이 맞추기
    min_length = min(spec_true.shape[1], spec_pred.shape[1])
    spec_true = spec_true[:, :min_length]
    spec_pred = spec_pred[:, :min_length]
    
    if distance_type == 'euclidean':
        diff = spec_true - spec_pred
        distortion = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        distortion = distortion.mean(dim=1)
        
    elif distance_type == 'cosine':
        # 코사인 거리
        spec_true_norm = F.normalize(spec_true, p=2, dim=-1)
        spec_pred_norm = F.normalize(spec_pred, p=2, dim=-1)
        cosine_sim = torch.sum(spec_true_norm * spec_pred_norm, dim=-1)
        distortion = 1 - cosine_sim.mean(dim=1)
        
    elif distance_type == 'kl_divergence':
        # KL Divergence (스펙트럼을 확률 분포로 해석)
        eps = 1e-8
        spec_true_prob = F.softmax(spec_true, dim=-1)
        spec_pred_prob = F.softmax(spec_pred, dim=-1)
        
        kl_div = spec_true_prob * torch.log(spec_true_prob / (spec_pred_prob + eps) + eps)
        distortion = kl_div.sum(dim=-1).mean(dim=1)
        
    else:
        raise ValueError(f"Unknown distance_type: {distance_type}")
    
    if squeeze_output:
        distortion = distortion.squeeze(0)
    
    return distortion


def perceptual_evaluation_speech_quality(clean_audio: torch.Tensor,
                                       degraded_audio: torch.Tensor,
                                       sample_rate: int = 16000) -> torch.Tensor:
    """
    PESQ (Perceptual Evaluation of Speech Quality) 유사 메트릭.
    
    실제 PESQ는 복잡한 psychoacoustic 모델을 사용하므로,
    여기서는 스펙트럼 기반 근사를 제공합니다.
    
    Args:
        clean_audio: 깨끗한 음성 (T,)
        degraded_audio: 품질 저하된 음성 (T,)
        sample_rate: 샘플링 레이트
        
    Returns:
        pesq_score: PESQ 유사 점수 (높을수록 좋음)
    """
    # 간단한 스펙트럼 기반 품질 측정
    # 실제 PESQ 구현은 매우 복잡하므로 근사 버전 제공
    
    # STFT
    window_size = int(0.025 * sample_rate)  # 25ms
    hop_size = int(0.010 * sample_rate)     # 10ms
    
    clean_spec = torch.stft(clean_audio, n_fft=window_size, hop_length=hop_size, 
                           return_complex=True)
    degraded_spec = torch.stft(degraded_audio, n_fft=window_size, hop_length=hop_size,
                              return_complex=True)
    
    # 크기 스펙트럼
    clean_mag = torch.abs(clean_spec)
    degraded_mag = torch.abs(degraded_spec)
    
    # 길이 맞추기
    min_time = min(clean_mag.shape[-1], degraded_mag.shape[-1])
    clean_mag = clean_mag[..., :min_time]
    degraded_mag = degraded_mag[..., :min_time]
    
    # 로그 스케일로 변환
    clean_log = torch.log(clean_mag + 1e-8)
    degraded_log = torch.log(degraded_mag + 1e-8)
    
    # MSE 기반 품질 점수 (간단화된 버전)
    mse = F.mse_loss(degraded_log, clean_log)
    
    # PESQ 유사 점수 (1~5 범위로 스케일링)
    pesq_like_score = torch.exp(-mse) * 4 + 1
    pesq_like_score = torch.clamp(pesq_like_score, 1, 5)
    
    return pesq_like_score


def comprehensive_speech_evaluation(predicted_features: Dict[str, torch.Tensor],
                                  ground_truth_features: Dict[str, torch.Tensor],
                                  evaluation_config: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """
    종합적인 음성 합성 품질 평가.
    
    여러 메트릭을 한 번에 계산하여 전반적인 성능을 평가합니다.
    
    Args:
        predicted_features: 예측된 특징들 {'mfcc', 'f0', 'alignment', ...}
        ground_truth_features: 정답 특징들
        evaluation_config: 평가 설정
        
    Returns:
        metrics: 모든 메트릭 결과
    """
    if evaluation_config is None:
        evaluation_config = {
            'mcd_exclude_c0': True,
            'f0_tolerance': 2,
            'alignment_tolerance': 0,
            'boundary_tolerance': 2,
            'duration_relative_tolerance': 0.2
        }
    
    metrics = {}
    
    # MCD 계산
    if 'mfcc' in predicted_features and 'mfcc' in ground_truth_features:
        mcd = mel_cepstral_distortion(
            ground_truth_features['mfcc'],
            predicted_features['mfcc'],
            exclude_c0=evaluation_config['mcd_exclude_c0']
        )
        metrics['mcd'] = mcd
    
    # F0 RMSE 계산
    if 'f0' in predicted_features and 'f0' in ground_truth_features:
        voiced_mask = ground_truth_features.get('voiced_mask', None)
        f0_rmse = f0_root_mean_square_error(
            ground_truth_features['f0'],
            predicted_features['f0'],
            voiced_mask=voiced_mask
        )
        metrics['f0_rmse'] = f0_rmse
        
        # Log F0 RMSE도 계산
        log_f0_rmse_val = log_f0_rmse(
            ground_truth_features['f0'],
            predicted_features['f0'],
            voiced_mask=voiced_mask
        )
        metrics['log_f0_rmse'] = log_f0_rmse_val
    
    # 정렬 정확도 계산
    if 'alignment' in predicted_features and 'alignment' in ground_truth_features:
        align_acc = alignment_accuracy(
            predicted_features['alignment'],
            ground_truth_features['alignment'],
            tolerance=evaluation_config['alignment_tolerance']
        )
        metrics['alignment_accuracy'] = align_acc
    
    # 경계 정확도 계산
    if 'boundaries' in predicted_features and 'boundaries' in ground_truth_features:
        boundary_metrics = boundary_accuracy(
            predicted_features['boundaries'],
            ground_truth_features['boundaries'],
            tolerance=evaluation_config['boundary_tolerance']
        )
        metrics.update({f'boundary_{k}': v for k, v in boundary_metrics.items()})
    
    # 지속시간 정확도 계산
    if 'durations' in predicted_features and 'durations' in ground_truth_features:
        duration_metrics = duration_accuracy(
            predicted_features['durations'],
            ground_truth_features['durations'],
            relative_tolerance=evaluation_config['duration_relative_tolerance']
        )
        metrics.update({f'duration_{k}': v for k, v in duration_metrics.items()})
    
    # 스펙트럼 왜곡 계산
    if 'mel_spectrogram' in predicted_features and 'mel_spectrogram' in ground_truth_features:
        spec_distortion = spectral_distortion(
            ground_truth_features['mel_spectrogram'],
            predicted_features['mel_spectrogram']
        )
        metrics['spectral_distortion'] = spec_distortion
    
    return metrics


# 유틸리티 함수들
def print_evaluation_summary(metrics: Dict[str, torch.Tensor]):
    """평가 결과 요약 출력."""
    print("=" * 50)
    print("Speech Synthesis Evaluation Summary")
    print("=" * 50)
    
    for metric_name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"{metric_name:25s}: {value.item():.4f}")
            else:
                print(f"{metric_name:25s}: {value.mean().item():.4f} (±{value.std().item():.4f})")
        else:
            print(f"{metric_name:25s}: {value}")
    
    print("=" * 50)


def save_evaluation_results(metrics: Dict[str, torch.Tensor], 
                          save_path: str):
    """평가 결과를 파일로 저장."""
    import json
    
    # 텐서를 리스트로 변환
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Evaluation results saved to {save_path}")
