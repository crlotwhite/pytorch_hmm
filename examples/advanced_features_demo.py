"""
PyTorch HMM Library Feature Demonstration

이 스크립트는 새로 구현된 고급 기능들의 사용법을 보여줍니다:
1. Dynamic Time Warping (DTW) 정렬
2. Connectionist Temporal Classification (CTC) 정렬  
3. Neural HMM with contextual modeling
4. Hidden Semi-Markov Model (HSMM) with duration modeling
5. Speech quality evaluation metrics
6. 통합 워크플로우 예제
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

# PyTorch HMM import
try:
    from pytorch_hmm import (
        # Core components
        HMMPyTorch, HMMLayer, 
        # Advanced models
        NeuralHMM, ContextualNeuralHMM, SemiMarkovHMM, DurationModel,
        # Alignment algorithms  
        DTWAligner, CTCAligner,
        # Evaluation metrics
        mel_cepstral_distortion, f0_root_mean_square_error, alignment_accuracy,
        comprehensive_speech_evaluation, print_evaluation_summary,
        # Utilities
        create_left_to_right_matrix, compute_state_durations
    )
    print("✓ Successfully imported PyTorch HMM library")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install the library first: pip install -e .")
    exit(1)


def demo_dtw_alignment():
    """DTW 정렬 데모"""
    print("\n" + "="*60)
    print("🎯 DEMO 1: Dynamic Time Warping (DTW) Alignment")
    print("="*60)
    
    # 음소 특징 시뮬레이션 (실제로는 MFCC 등)
    print("Creating simulated phoneme and audio features...")
    
    # 5개 음소의 특징 벡터
    phoneme_features = torch.randn(5, 12)  # 5 phonemes, 12-dim features
    print(f"Phoneme features shape: {phoneme_features.shape}")
    
    # 100 프레임의 음성 특징 (실제로는 더 긴 시퀀스)
    audio_features = torch.randn(100, 12)  # 100 frames, 12-dim features  
    print(f"Audio features shape: {audio_features.shape}")
    
    # DTW 정렬 수행
    print("\nPerforming DTW alignment...")
    aligner = DTWAligner(distance_fn='cosine', step_pattern='symmetric')
    
    start_time = time.time()
    path_i, path_j, total_cost = aligner(phoneme_features, audio_features)
    dtw_time = time.time() - start_time
    
    print(f"DTW alignment completed in {dtw_time:.4f}s")
    print(f"Alignment path length: {len(path_i)}")
    print(f"Total DTW cost: {total_cost:.4f}")
    
    # 음소 경계 추출
    phoneme_boundaries = []
    current_phoneme = path_i[0].item()
    current_start = 0
    
    for i, phoneme_idx in enumerate(path_i):
        if phoneme_idx != current_phoneme:
            phoneme_boundaries.append((current_phoneme, current_start, path_j[i-1].item()))
            current_phoneme = phoneme_idx.item()
            current_start = path_j[i].item()
    
    # 마지막 음소
    phoneme_boundaries.append((current_phoneme, current_start, path_j[-1].item()))
    
    print("\nExtracted phoneme boundaries:")
    for phoneme_id, start_frame, end_frame in phoneme_boundaries:
        duration_ms = (end_frame - start_frame) * 10  # 10ms per frame
        print(f"  Phoneme {phoneme_id}: frames {start_frame:3d}-{end_frame:3d} ({duration_ms:3d}ms)")
    
    return phoneme_boundaries


def demo_ctc_alignment():
    """CTC 정렬 데모"""
    print("\n" + "="*60)
    print("🎯 DEMO 2: CTC Alignment for Speech Recognition")
    print("="*60)
    
    # 음성 인식 시뮬레이션
    vocab_size = 28  # 26 letters + blank + space
    sequence_length = 80
    batch_size = 2
    
    print(f"Simulating ASR with vocab_size={vocab_size}, seq_length={sequence_length}")
    
    # 음향 모델의 출력 (로그 확률)
    log_probs = torch.log_softmax(torch.randn(sequence_length, batch_size, vocab_size), dim=-1)
    
    # 타겟 텍스트 ("HELLO" 와 "WORLD")
    targets = torch.tensor([[8, 5, 12, 12, 15],   # HELLO (H=8, E=5, L=12, L=12, O=15)
                           [23, 15, 18, 12, 4]])  # WORLD (W=23, O=15, R=18, L=12, D=4)
    
    input_lengths = torch.full((batch_size,), sequence_length)
    target_lengths = torch.tensor([5, 5])
    
    print(f"Target texts: ['HELLO', 'WORLD']")
    print(f"Input lengths: {input_lengths.tolist()}")
    print(f"Target lengths: {target_lengths.tolist()}")
    
    # CTC 정렬 및 디코딩
    print("\nPerforming CTC alignment...")
    ctc_aligner = CTCAligner(num_classes=vocab_size, blank_id=0)
    
    # Loss 계산
    ctc_loss = ctc_aligner(log_probs, targets, input_lengths, target_lengths)
    print(f"CTC Loss: {ctc_loss.item():.4f}")
    
    # Greedy 디코딩
    decoded_sequences = ctc_aligner.decode(log_probs, input_lengths)
    
    print("\nDecoded sequences:")
    for i, decoded in enumerate(decoded_sequences):
        decoded_chars = [chr(ord('A') + idx - 1) if idx > 0 else '_' for idx in decoded.tolist()]
        print(f"  Sequence {i}: {decoded.tolist()} -> {''.join(decoded_chars)}")
    
    # 강제 정렬
    alignments = ctc_aligner.align(log_probs, targets, input_lengths, target_lengths)
    
    print("\nForced alignment paths:")
    for i, alignment in enumerate(alignments):
        print(f"  Sequence {i}: {len(alignment)} frames")
        # 처음 20 프레임의 정렬 표시
        first_20 = alignment[:20].tolist()
        alignment_chars = [chr(ord('A') + idx - 1) if idx > 0 else '_' for idx in first_20]
        print(f"    First 20 frames: {''.join(alignment_chars)}")


def demo_neural_hmm():
    """Neural HMM 데모"""
    print("\n" + "="*60)
    print("🎯 DEMO 3: Neural HMM with Contextual Modeling")
    print("="*60)
    
    # 음성 합성 시나리오: 음소 시퀀스 -> 음향 특징
    num_phonemes = 10
    num_states = 3  # 각 음소당 3개 상태 (시작-중간-끝)
    total_states = num_phonemes * num_states
    
    observation_dim = 80  # 멜 스펙트로그램 차원
    context_dim = 64
    
    print(f"Setting up Neural HMM: {num_phonemes} phonemes, {num_states} states each")
    print(f"Observation dim: {observation_dim}, Context dim: {context_dim}")
    
    # Neural HMM 생성
    neural_hmm = NeuralHMM(
        num_states=total_states,
        observation_dim=observation_dim, 
        context_dim=context_dim,
        hidden_dim=128,
        transition_type='rnn',  # RNN 기반 전이 모델
        observation_type='mixture'  # Mixture Gaussian 관측 모델
    )
    
    # 컨텍스트 정보 (언어적 특징, 운율 등)
    batch_size, seq_length = 2, 200
    context = torch.randn(batch_size, seq_length, context_dim)
    observations = torch.randn(batch_size, seq_length, observation_dim)
    
    print(f"\nInput shapes:")
    print(f"  Context: {context.shape}")  
    print(f"  Observations: {observations.shape}")
    
    # 추론 수행
    print("\nPerforming Neural HMM inference...")
    start_time = time.time()
    
    with torch.no_grad():  # 빠른 추론을 위해
        # Forward-backward
        posteriors, forward, backward = neural_hmm(observations, context)
        
        # Viterbi 디코딩
        optimal_states, state_scores = neural_hmm.viterbi_decode(observations, context)
    
    inference_time = time.time() - start_time
    
    print(f"Inference completed in {inference_time:.4f}s")
    print(f"Posterior shape: {posteriors.shape}")
    print(f"Optimal states shape: {optimal_states.shape}")
    
    # 상태 점유 분석
    print("\nState occupancy analysis:")
    for b in range(batch_size):
        state_counts = torch.bincount(optimal_states[b], minlength=total_states)
        occupied_states = (state_counts > 0).sum().item()
        print(f"  Sequence {b}: {occupied_states}/{total_states} states used")
        
        # 지속시간 분석
        durations = compute_state_durations(optimal_states[b])
        avg_duration = durations.float().mean().item()
        print(f"    Average state duration: {avg_duration:.2f} frames")


def demo_contextual_neural_hmm():
    """Contextual Neural HMM 데모"""
    print("\n" + "="*60)
    print("🎯 DEMO 4: Contextual Neural HMM for TTS")
    print("="*60)
    
    # TTS 시나리오: 음소 + 컨텍스트 -> 음향 정렬
    phoneme_vocab_size = 50  # 한국어 음소 개수
    num_states = 5
    observation_dim = 80
    
    print(f"TTS setup: {phoneme_vocab_size} phonemes, {num_states} HMM states")
    
    # Contextual Neural HMM 생성
    contextual_hmm = ContextualNeuralHMM(
        num_states=num_states,
        observation_dim=observation_dim,
        phoneme_vocab_size=phoneme_vocab_size,
        linguistic_context_dim=32,
        prosody_dim=8
    )
    
    # 입력 데이터
    batch_size, seq_length = 1, 150
    
    # 음소 시퀀스 (예: "안녕하세요")
    phoneme_sequence = torch.tensor([[
        10, 11, 25, 26, 15, 16, 30, 31, 40, 41  # 5개 음소, 각각 2프레임
    ] * 15]).long()  # 150 프레임으로 확장
    
    # 운율 특징 (F0, 에너지 등)
    prosody_features = torch.randn(batch_size, seq_length, 8)
    
    # 음향 특징 (멜 스펙트로그램)
    acoustic_features = torch.randn(batch_size, seq_length, observation_dim)
    
    print(f"Input shapes:")
    print(f"  Phoneme sequence: {phoneme_sequence.shape}")
    print(f"  Prosody features: {prosody_features.shape}")
    print(f"  Acoustic features: {acoustic_features.shape}")
    
    # 컨텍스트 기반 추론
    print("\nPerforming contextual inference...")
    start_time = time.time()
    
    with torch.no_grad():
        posteriors, forward, backward = contextual_hmm.forward_with_context(
            acoustic_features, phoneme_sequence, prosody_features)
    
    inference_time = time.time() - start_time
    
    print(f"Contextual inference completed in {inference_time:.4f}s")
    print(f"Posterior probabilities shape: {posteriors.shape}")
    
    # 가장 확률이 높은 상태 경로
    most_likely_states = torch.argmax(posteriors, dim=-1)
    
    print("\nState transition analysis:")
    transitions = []
    for t in range(1, seq_length):
        if most_likely_states[0, t] != most_likely_states[0, t-1]:
            transitions.append((t, most_likely_states[0, t-1].item(), most_likely_states[0, t].item()))
    
    print(f"Total state transitions: {len(transitions)}")
    for i, (frame, from_state, to_state) in enumerate(transitions[:10]):  # 처음 10개만 표시
        print(f"  Frame {frame}: State {from_state} -> {to_state}")


def demo_semi_markov_hmm():
    """Semi-Markov HMM 데모"""
    print("\n" + "="*60)
    print("🎯 DEMO 5: Hidden Semi-Markov Model (HSMM)")
    print("="*60)
    
    # 음소 지속시간 모델링
    num_phonemes = 8
    observation_dim = 13  # MFCC 차원
    max_duration = 20
    
    print(f"HSMM setup: {num_phonemes} phonemes, max duration {max_duration} frames")
    
    # HSMM 생성
    hsmm = SemiMarkovHMM(
        num_states=num_phonemes,
        observation_dim=observation_dim,
        max_duration=max_duration,
        duration_distribution='gamma',  # 감마 분포로 지속시간 모델링
        observation_model='gaussian'
    )
    
    print(f"Duration distribution: Gamma")
    print(f"Observation model: Gaussian")
    
    # 지속시간 모델 분석
    print("\nDuration model analysis:")
    test_phonemes = torch.arange(num_phonemes)
    
    # 각 음소의 지속시간 분포 확인
    duration_distributions = hsmm.duration_model(test_phonemes)
    
    for phoneme_id in range(min(5, num_phonemes)):  # 처음 5개 음소만 표시
        probs = torch.exp(duration_distributions[phoneme_id])  # log -> prob
        most_likely_duration = torch.argmax(probs).item() + 1
        print(f"  Phoneme {phoneme_id}: Most likely duration = {most_likely_duration} frames")
    
    # 시퀀스 샘플링
    print("\nSampling sequence from HSMM...")
    sampled_states, sampled_durations, sampled_observations = hsmm.sample(
        num_states=6, max_length=100)
    
    print(f"Sampled sequence:")
    print(f"  States: {sampled_states.tolist()}")
    print(f"  Durations: {sampled_durations.tolist()}")
    print(f"  Total length: {sampled_durations.sum().item()} frames")
    print(f"  Observations shape: {sampled_observations.shape}")
    
    # 지속시간 통계
    avg_duration = sampled_durations.float().mean().item()
    std_duration = sampled_durations.float().std().item()
    print(f"  Average duration: {avg_duration:.2f} ± {std_duration:.2f} frames")


def demo_evaluation_metrics():
    """음성 품질 평가 메트릭 데모"""
    print("\n" + "="*60)
    print("🎯 DEMO 6: Speech Quality Evaluation Metrics")
    print("="*60)
    
    # 시뮬레이션된 TTS 시스템 평가
    seq_length = 200
    mfcc_dim = 13
    
    print(f"Evaluating TTS system: {seq_length} frames, {mfcc_dim}-dim MFCC")
    
    # Ground truth 특징들
    gt_mfcc = torch.randn(seq_length, mfcc_dim)
    gt_f0 = torch.abs(torch.randn(seq_length)) * 100 + 120  # 120-220 Hz
    gt_alignment = torch.randint(0, 10, (seq_length,))
    
    # 예측된 특징들 (약간의 오차 포함)
    noise_level = 0.1
    pred_mfcc = gt_mfcc + noise_level * torch.randn(seq_length, mfcc_dim)
    pred_f0 = gt_f0 + 5 * torch.randn(seq_length)  # 5Hz 오차
    pred_alignment = gt_alignment.clone()
    pred_alignment[::20] = (pred_alignment[::20] + 1) % 10  # 5% 오류
    
    # 개별 메트릭 계산
    print("\nComputing individual metrics...")
    
    # MCD (Mel-Cepstral Distortion)
    mcd = mel_cepstral_distortion(gt_mfcc, pred_mfcc, exclude_c0=True)
    print(f"MCD: {mcd.item():.4f} dB")
    
    # F0 RMSE
    voiced_mask = gt_f0 > 0  # 유성음 구간
    f0_rmse = f0_root_mean_square_error(gt_f0, pred_f0, voiced_mask)
    print(f"F0 RMSE: {f0_rmse.item():.4f} Hz")
    
    # 정렬 정확도
    align_acc = alignment_accuracy(pred_alignment, gt_alignment, tolerance=0)
    print(f"Alignment Accuracy: {align_acc.item():.4f} ({align_acc.item()*100:.1f}%)")
    
    # 종합 평가
    print("\nComprehensive evaluation...")
    predicted_features = {
        'mfcc': pred_mfcc.unsqueeze(0),  # 배치 차원 추가
        'f0': pred_f0.unsqueeze(0),
        'alignment': pred_alignment
    }
    
    ground_truth_features = {
        'mfcc': gt_mfcc.unsqueeze(0),
        'f0': gt_f0.unsqueeze(0),
        'alignment': gt_alignment
    }
    
    comprehensive_metrics = comprehensive_speech_evaluation(
        predicted_features, ground_truth_features)
    
    print_evaluation_summary(comprehensive_metrics)
    
    # 품질 평가
    print("\nQuality Assessment:")
    if mcd < 5.0:
        print("  ✓ Excellent MCD (< 5.0 dB)")
    elif mcd < 8.0:
        print("  ✓ Good MCD (< 8.0 dB)")
    else:
        print("  ⚠ Poor MCD (> 8.0 dB)")
    
    if f0_rmse < 10.0:
        print("  ✓ Excellent F0 accuracy (< 10 Hz)")
    elif f0_rmse < 20.0:
        print("  ✓ Good F0 accuracy (< 20 Hz)")
    else:
        print("  ⚠ Poor F0 accuracy (> 20 Hz)")
    
    if align_acc > 0.9:
        print("  ✓ Excellent alignment (> 90%)")
    elif align_acc > 0.8:
        print("  ✓ Good alignment (> 80%)")
    else:
        print("  ⚠ Poor alignment (< 80%)")


def demo_integration_workflow():
    """통합 워크플로우 데모"""
    print("\n" + "="*60)
    print("🎯 DEMO 7: Complete TTS Pipeline Integration")
    print("="*60)
    
    print("Simulating complete Text-to-Speech pipeline...")
    
    # 1. 텍스트 전처리 (시뮬레이션)
    print("\n1. Text preprocessing:")
    text = "안녕하세요"  # "Hello" in Korean
    phonemes = [10, 15, 20, 25, 30]  # 음소 ID들
    print(f"   Text: '{text}' -> Phonemes: {phonemes}")
    
    # 2. 언어적 특징 추출
    print("\n2. Linguistic feature extraction:")
    phoneme_sequence = torch.tensor([phonemes * 20]).long()  # 100 프레임으로 확장
    prosody_features = torch.randn(1, 100, 8)  # 운율 특징
    
    print(f"   Phoneme sequence: {phoneme_sequence.shape}")
    print(f"   Prosody features: {prosody_features.shape}")
    
    # 3. 지속시간 예측 (HSMM 사용)
    print("\n3. Duration prediction with HSMM:")
    duration_model = DurationModel(
        num_states=len(set(phonemes)),
        max_duration=30,
        distribution_type='neural'
    )
    
    predicted_durations = duration_model.sample(torch.tensor(phonemes))
    print(f"   Predicted durations: {predicted_durations.tolist()}")
    total_frames = predicted_durations.sum().item()
    print(f"   Total frames: {total_frames}")
    
    # 4. 음향 특징 생성 (Neural HMM 사용)
    print("\n4. Acoustic feature generation with Neural HMM:")
    acoustic_model = NeuralHMM(
        num_states=5,
        observation_dim=80,
        context_dim=40,  # 음소 + 운율 특징
        hidden_dim=128
    )
    
    # 컨텍스트 결합
    phoneme_emb = torch.randn(1, 100, 32)  # 음소 임베딩
    context = torch.cat([phoneme_emb, prosody_features], dim=-1)
    
    # 음향 특징 생성
    dummy_acoustic = torch.randn(1, 100, 80)
    posteriors, _, _ = acoustic_model(dummy_acoustic, context)
    
    print(f"   Generated acoustic features: {posteriors.shape}")
    
    # 5. 정렬 및 후처리 (DTW 사용)
    print("\n5. Alignment refinement with DTW:")
    target_length = int(total_frames * 1.1)  # 10% 길이 조정
    
    dtw_aligner = DTWAligner(step_pattern='asymmetric')
    source_features = posteriors[0]  # (100, 80)
    target_features = torch.randn(target_length, 80)
    
    path_i, path_j, _ = dtw_aligner(source_features, target_features)
    print(f"   DTW alignment: {len(path_i)} -> {len(path_j)} frames")
    
    # 6. 품질 평가
    print("\n6. Quality evaluation:")
    
    # 시뮬레이션된 ground truth
    gt_acoustic = torch.randn(1, 100, 80)
    gt_alignment = torch.randint(0, 5, (100,))
    
    pred_alignment = torch.argmax(posteriors, dim=-1)[0]
    align_acc = alignment_accuracy(pred_alignment, gt_alignment, tolerance=1)
    
    print(f"   Alignment accuracy: {align_acc.item():.4f}")
    
    # 7. 최종 결과
    print("\n7. Pipeline summary:")
    total_time = 100 * 0.01  # 100 frames × 10ms
    print(f"   Generated speech duration: {total_time:.2f} seconds")
    print(f"   Average phoneme duration: {predicted_durations.float().mean():.1f} frames")
    print(f"   Model complexity: Neural transitions + Gaussian observations")
    print(f"   Alignment method: DTW with cosine distance")
    
    print("\n✓ Complete TTS pipeline demonstration finished!")


def main():
    """메인 데모 실행"""
    print("🚀 PyTorch HMM Advanced Features Demonstration")
    print("=" * 70)
    
    # 시드 설정으로 재현 가능한 결과
    torch.manual_seed(42)
    np.random.seed(42)
    
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device.upper()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # 모든 데모 실행
    demos = [
        demo_dtw_alignment,
        demo_ctc_alignment, 
        demo_neural_hmm,
        demo_contextual_neural_hmm,
        demo_semi_markov_hmm,
        demo_evaluation_metrics,
        demo_integration_workflow
    ]
    
    for i, demo_func in enumerate(demos, 1):
        try:
            demo_func()
            print(f"\n✅ Demo {i} completed successfully!")
        except Exception as e:
            print(f"\n❌ Demo {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 요약
    print("\n" + "="*70)
    print("🎉 All demonstrations completed!")
    print("="*70)
    
    print("\nKey takeaways:")
    print("• DTW provides flexible sequence alignment for variable-length data")
    print("• CTC enables end-to-end learning without explicit alignment")
    print("• Neural HMMs incorporate contextual information for better modeling")
    print("• HSMMs explicitly model state duration distributions")
    print("• Comprehensive metrics enable thorough quality evaluation")
    print("• Integration workflows demonstrate real-world applicability")
    
    print(f"\n📚 For more information, check the documentation and examples!")
    print("   GitHub: https://github.com/crlotwhite/pytorch_hmm")


if __name__ == "__main__":
    main()
