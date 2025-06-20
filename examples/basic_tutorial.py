#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch>=1.12.0",
#     "numpy>=1.21.0",
#     "matplotlib>=3.5.0",
# ]
# ///
"""
PyTorch HMM Basic Tutorial

이 튜토리얼은 PyTorch HMM 라이브러리의 기본 사용법을 보여줍니다.
음성 합성과 시퀀스 모델링의 기초부터 고급 기능까지 단계별로 학습할 수 있습니다.

목차:
1. 기본 HMM 사용법
2. Forward-backward vs Viterbi 비교
3. HMMLayer를 이용한 신경망 통합
4. 배치 처리
5. GPU 사용법
6. 실제 응용 예제
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

# PyTorch HMM import
from pytorch_hmm import (
    HMMPyTorch, HMMLayer, GaussianHMMLayer,
    create_left_to_right_matrix, create_transition_matrix,
    compute_state_durations
)

def tutorial_1_basic_hmm():
    """튜토리얼 1: 기본 HMM 사용법"""
    print("=" * 60)
    print("📚 Tutorial 1: Basic HMM Usage")
    print("=" * 60)

    # Step 1: Transition matrix 생성
    print("\n1. Creating transition matrix...")
    num_states = 5

    # Left-to-right transition matrix (음성 합성에서 일반적)
    P = create_left_to_right_matrix(num_states, self_loop_prob=0.7)
    print(f"   Transition matrix shape: {P.shape}")
    print(f"   Self-loop probability: 0.7")

    # 전이 행렬 내용 확인
    print("\n   Transition matrix:")
    for i in range(num_states):
        row_str = "   " + " ".join([f"{P[i,j]:.3f}" for j in range(num_states)])
        print(row_str)

    # Step 2: HMM 모델 생성
    print("\n2. Creating HMM model...")
    hmm = HMMPyTorch(P)
    print(f"   Number of states: {hmm.K}")
    print(f"   Device: {hmm.device}")

    # Step 3: 관측 데이터 준비
    print("\n3. Preparing observation data...")
    batch_size, seq_len = 2, 20
    observations = torch.softmax(torch.randn(batch_size, seq_len, num_states), dim=-1)
    print(f"   Observation shape: {observations.shape}")
    print(f"   Data type: Probabilistic observations (sum to 1)")

    # Step 4: Forward-backward 알고리즘
    print("\n4. Running forward-backward algorithm...")
    start_time = time.time()
    posteriors, forward, backward = hmm.forward_backward(observations)
    fb_time = time.time() - start_time

    print(f"   Forward-backward time: {fb_time:.4f}s")
    print(f"   Posterior shape: {posteriors.shape}")
    print(f"   Posterior sum check: {posteriors.sum(dim=-1)[0, :5]}")  # Should be ~1.0

    # Step 5: Viterbi 디코딩
    print("\n5. Running Viterbi decoding...")
    start_time = time.time()
    states, scores = hmm.viterbi_decode(observations)
    viterbi_time = time.time() - start_time

    print(f"   Viterbi time: {viterbi_time:.4f}s")
    print(f"   Optimal states shape: {states.shape}")
    print(f"   State sequence (first 10): {states[0, :10].tolist()}")

    # Step 6: Likelihood 계산
    print("\n6. Computing sequence likelihood...")
    log_likelihood = hmm.compute_likelihood(observations)
    print(f"   Log-likelihood: {log_likelihood}")
    print(f"   Likelihood: {torch.exp(log_likelihood)}")

    return hmm, observations, posteriors, states


def tutorial_2_forward_backward_vs_viterbi():
    """튜토리얼 2: Forward-backward vs Viterbi 비교"""
    print("\n" + "=" * 60)
    print("📚 Tutorial 2: Forward-backward vs Viterbi Comparison")
    print("=" * 60)

    # 테스트 설정
    num_states = 6
    P = create_left_to_right_matrix(num_states, self_loop_prob=0.8)
    hmm = HMMPyTorch(P)

    # 더 긴 시퀀스로 테스트
    seq_len = 100
    observations = torch.softmax(torch.randn(1, seq_len, num_states), dim=-1)

    print(f"\nComparing algorithms on sequence length: {seq_len}")

    # Forward-backward
    print("\n1. Forward-backward algorithm:")
    start_time = time.time()
    posteriors, _, _ = hmm.forward_backward(observations)
    fb_time = time.time() - start_time

    # Soft alignment (가장 확률이 높은 상태)
    soft_states = torch.argmax(posteriors, dim=-1)[0]

    print(f"   Time: {fb_time:.4f}s")
    print(f"   Output: Soft posteriors (probabilistic)")
    print(f"   Max posterior states: {soft_states[:15].tolist()}")

    # Viterbi
    print("\n2. Viterbi algorithm:")
    start_time = time.time()
    hard_states, scores = hmm.viterbi_decode(observations)
    viterbi_time = time.time() - start_time

    print(f"   Time: {viterbi_time:.4f}s")
    print(f"   Output: Hard alignment (deterministic)")
    print(f"   Optimal states: {hard_states[0, :15].tolist()}")

    # 결과 비교
    print("\n3. Comparison:")
    print(f"   Speed ratio (Viterbi/FB): {viterbi_time/fb_time:.2f}x")

    # 정렬 차이 분석
    agreement = (soft_states == hard_states[0]).float().mean()
    print(f"   State agreement: {agreement:.3f} ({agreement*100:.1f}%)")

    # 지속시간 분석
    soft_durations = compute_state_durations(soft_states)
    hard_durations = compute_state_durations(hard_states[0])

    print(f"   Avg duration (soft): {soft_durations.float().mean():.2f}")
    print(f"   Avg duration (hard): {hard_durations.float().mean():.2f}")

    # 언제 어떤 알고리즘을 사용할지 가이드
    print("\n4. When to use which:")
    print("   📊 Forward-backward:")
    print("      - Training (gradient computation)")
    print("      - Uncertainty quantification")
    print("      - Soft alignment for fusion")
    print("   🎯 Viterbi:")
    print("      - Inference (final alignment)")
    print("      - Real-time applications")
    print("      - Hard decision making")

    return soft_states, hard_states[0]


def tutorial_3_hmm_layer_integration():
    """튜토리얼 3: HMMLayer를 이용한 신경망 통합"""
    print("\n" + "=" * 60)
    print("📚 Tutorial 3: HMM Integration with Neural Networks")
    print("=" * 60)

    # 음성 합성 모델 시뮬레이션
    print("\n1. Building TTS model with HMM alignment...")

    class SimpleTTSModel(nn.Module):
        def __init__(self, input_dim, num_phonemes, output_dim):
            super().__init__()

            # 텍스트 인코더
            self.text_encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU()
            )

            # HMM 정렬 레이어
            self.hmm_layer = HMMLayer(
                num_states=num_phonemes,
                learnable_transitions=True,
                transition_type="left_to_right",
                viterbi_inference=False  # Training에서는 soft alignment
            )

            # 음향 디코더
            self.acoustic_decoder = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )

        def forward(self, text_features, return_alignment=False):
            # 텍스트 인코딩
            encoded = self.text_encoder(text_features)

            # HMM 정렬
            if return_alignment:
                aligned, alignment = self.hmm_layer(encoded, return_alignment=True)
                acoustic_output = self.acoustic_decoder(aligned)
                return acoustic_output, aligned, alignment
            else:
                aligned = self.hmm_layer(encoded)
                acoustic_output = self.acoustic_decoder(aligned)
                return acoustic_output, aligned

    # 모델 생성
    input_dim, num_phonemes, output_dim = 50, 8, 80
    model = SimpleTTSModel(input_dim, num_phonemes, output_dim)

    print(f"   Model created:")
    print(f"   - Input dimension: {input_dim}")
    print(f"   - Number of phonemes: {num_phonemes}")
    print(f"   - Output dimension: {output_dim}")

    # 2. 훈련 시뮬레이션
    print("\n2. Training simulation...")

    # 더미 데이터
    batch_size, seq_len = 4, 30
    text_features = torch.randn(batch_size, seq_len, input_dim)
    target_acoustic = torch.randn(batch_size, seq_len, output_dim)

    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"   Training data shape: {text_features.shape}")
    print(f"   Target shape: {target_acoustic.shape}")

    # 훈련 루프
    losses = []
    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass
        predicted_acoustic, alignment = model(text_features)

        # Loss 계산
        reconstruction_loss = nn.MSELoss()(predicted_acoustic, target_acoustic)

        # HMM regularization (옵션)
        transition_matrix = model.hmm_layer.get_transition_matrix()
        regularization = 0.01 * torch.sum(transition_matrix ** 2)

        total_loss = reconstruction_loss + regularization

        # Backward pass
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        print(f"   Epoch {epoch+1}: Loss = {total_loss.item():.4f}")

    # 3. 추론 테스트
    print("\n3. Inference test...")
    model.eval()

    with torch.no_grad():
        # Soft alignment (training mode)
        model.hmm_layer.viterbi_inference = False
        soft_output, soft_alignment = model(text_features[:1])

        # Hard alignment (inference mode)
        model.hmm_layer.viterbi_inference = True
        hard_output, hard_alignment, viterbi_path = model(
            text_features[:1], return_alignment=True)

    print(f"   Soft alignment shape: {soft_alignment.shape}")
    print(f"   Hard alignment shape: {viterbi_path.shape}")
    print(f"   Output difference: {torch.norm(soft_output - hard_output):.4f}")

    # 4. HMM 파라미터 분석
    print("\n4. Learned HMM parameters:")
    learned_transitions = model.hmm_layer.get_transition_matrix()
    learned_initial = model.hmm_layer.get_initial_probabilities()

    print(f"   Transition matrix shape: {learned_transitions.shape}")
    print(f"   Initial probabilities: {learned_initial[:5]}")

    # 가장 확률이 높은 전이 표시
    max_transitions = torch.argmax(learned_transitions, dim=1)
    print(f"   Most likely transitions: {max_transitions.tolist()}")

    return model, losses


def tutorial_4_batch_processing():
    """튜토리얼 4: 배치 처리"""
    print("\n" + "=" * 60)
    print("📚 Tutorial 4: Efficient Batch Processing")
    print("=" * 60)

    # 다양한 배치 크기로 성능 테스트
    num_states = 10
    P = create_left_to_right_matrix(num_states)
    hmm = HMMPyTorch(P)

    batch_sizes = [1, 4, 8, 16, 32]
    seq_len = 50

    print(f"\nBatch processing performance test:")
    print(f"Sequence length: {seq_len}, States: {num_states}")
    print(f"{'Batch Size':>10} {'Time (s)':>10} {'FPS':>10} {'Memory (MB)':>12}")
    print("-" * 50)

    for batch_size in batch_sizes:
        observations = torch.softmax(
            torch.randn(batch_size, seq_len, num_states), dim=-1)

        # 메모리 사용량 체크 (PyTorch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 성능 측정
        start_time = time.time()
        with torch.no_grad():
            posteriors, _, _ = hmm.forward_backward(observations)
            states, _ = hmm.viterbi_decode(observations)
        process_time = time.time() - start_time

        # 초당 프레임 수 계산
        total_frames = batch_size * seq_len
        fps = total_frames / process_time

        # 메모리 사용량
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            memory_mb = 0  # CPU 메모리는 추정이 어려움

        print(f"{batch_size:>10} {process_time:>10.4f} {fps:>10.0f} {memory_mb:>10.1f}")

    # 배치 처리 최적화 팁
    print("\n💡 Batch Processing Tips:")
    print("   1. Use larger batches for better GPU utilization")
    print("   2. Consider memory constraints with very long sequences")
    print("   3. Use torch.no_grad() for inference to save memory")
    print("   4. Pad sequences to same length for efficient batching")

    # 가변 길이 시퀀스 처리 예제
    print("\n📝 Variable length sequence handling:")

    # 서로 다른 길이의 시퀀스들
    sequences = [
        torch.softmax(torch.randn(20, num_states), dim=-1),
        torch.softmax(torch.randn(35, num_states), dim=-1),
        torch.softmax(torch.randn(28, num_states), dim=-1),
    ]

    print(f"   Sequence lengths: {[len(seq) for seq in sequences]}")

    # 패딩으로 동일한 길이로 만들기
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    masks = []

    for seq in sequences:
        pad_len = max_len - len(seq)
        if pad_len > 0:
            padding = torch.zeros(pad_len, num_states)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq

        # 마스크 생성 (실제 데이터 vs 패딩)
        mask = torch.cat([
            torch.ones(len(seq)),
            torch.zeros(pad_len)
        ]).bool()

        padded_sequences.append(padded_seq)
        masks.append(mask)

    # 배치로 스택
    batch_observations = torch.stack(padded_sequences)
    batch_masks = torch.stack(masks)

    print(f"   Padded batch shape: {batch_observations.shape}")
    print(f"   Mask shape: {batch_masks.shape}")

    # 마스킹된 처리
    with torch.no_grad():
        posteriors, _, _ = hmm.forward_backward(batch_observations)

        # 마스크를 적용하여 패딩 부분 제거
        masked_posteriors = posteriors * batch_masks.unsqueeze(-1)

    print(f"   Masked posteriors shape: {masked_posteriors.shape}")
    print("   ✓ Variable length sequences processed successfully!")


def tutorial_5_gpu_usage():
    """튜토리얼 5: GPU 사용법"""
    print("\n" + "=" * 60)
    print("📚 Tutorial 5: GPU Acceleration")
    print("=" * 60)

    # GPU 사용 가능 여부 확인
    print(f"\n1. GPU Availability Check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   GPU device: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("   Using CPU for demonstration")

    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Selected device: {device}")

    # 2. CPU vs GPU 성능 비교
    print(f"\n2. Performance Comparison:")

    num_states = 15
    batch_size = 8
    seq_len = 100

    # HMM 모델 생성
    P = create_left_to_right_matrix(num_states)

    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')

    print(f"   Test configuration:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Number of states: {num_states}")

    print(f"\n   {'Device':>8} {'Forward-Backward':>15} {'Viterbi':>10} {'Speedup':>10}")
    print("   " + "-" * 50)

    cpu_fb_time = None
    cpu_viterbi_time = None

    for test_device in devices_to_test:
        # 모델과 데이터를 디바이스로 이동
        hmm = HMMPyTorch(P).to(test_device)
        observations = torch.softmax(
            torch.randn(batch_size, seq_len, num_states), dim=-1).to(test_device)

        # GPU warming up (GPU의 경우)
        if test_device == 'cuda':
            for _ in range(3):
                hmm.forward_backward(observations)
                hmm.viterbi_decode(observations)
            torch.cuda.synchronize()

        # Forward-backward 벤치마크
        start_time = time.time()
        for _ in range(10):  # 여러 번 실행하여 평균
            posteriors, _, _ = hmm.forward_backward(observations)
        if test_device == 'cuda':
            torch.cuda.synchronize()
        fb_time = (time.time() - start_time) / 10

        # Viterbi 벤치마크
        start_time = time.time()
        for _ in range(10):
            states, _ = hmm.viterbi_decode(observations)
        if test_device == 'cuda':
            torch.cuda.synchronize()
        viterbi_time = (time.time() - start_time) / 10

        # 스피드업 계산
        if test_device == 'cpu':
            cpu_fb_time = fb_time
            cpu_viterbi_time = viterbi_time
            speedup_str = "1.0x"
        else:
            fb_speedup = cpu_fb_time / fb_time
            viterbi_speedup = cpu_viterbi_time / viterbi_time
            speedup_str = f"{fb_speedup:.1f}x/{viterbi_speedup:.1f}x"

        print(f"   {test_device.upper():>8} {fb_time:>13.4f}s {viterbi_time:>8.4f}s {speedup_str:>10}")

    # 3. GPU 메모리 관리
    if torch.cuda.is_available():
        print(f"\n3. GPU Memory Management:")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 큰 배치로 메모리 사용량 테스트
        large_batch = 32
        large_seq = 200

        hmm = HMMPyTorch(P).cuda()
        large_obs = torch.softmax(
            torch.randn(large_batch, large_seq, num_states), dim=-1).cuda()

        print(f"   Processing large batch: {large_batch}x{large_seq}")

        initial_memory = torch.cuda.memory_allocated()
        posteriors, _, _ = hmm.forward_backward(large_obs)
        peak_memory = torch.cuda.max_memory_allocated()

        print(f"   Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
        print(f"   Peak GPU memory: {peak_memory / 1024**2:.1f} MB")
        print(f"   Additional memory used: {(peak_memory - initial_memory) / 1024**2:.1f} MB")

        # 메모리 정리
        del large_obs, posteriors
        torch.cuda.empty_cache()
        print("   ✓ Memory cleaned up")

    # 4. 디바이스 이동 팁
    print(f"\n4. 💡 GPU Usage Tips:")
    print("   • Move model to GPU: model.to('cuda')")
    print("   • Move data to GPU: data.to('cuda')")
    print("   • Use torch.cuda.synchronize() for accurate timing")
    print("   • Clear cache with torch.cuda.empty_cache()")
    print("   • Monitor memory with torch.cuda.memory_allocated()")
    print("   • Use larger batches to maximize GPU utilization")


def tutorial_6_real_world_application():
    """튜토리얼 6: 실제 응용 예제"""
    print("\n" + "=" * 60)
    print("📚 Tutorial 6: Real-world Application Example")
    print("=" * 60)

    print("\n🎯 Application: Phoneme Duration Modeling for TTS")
    print("Goal: Learn natural phoneme durations from speech data")

    # 1. 시뮬레이션된 음성 데이터
    print("\n1. Simulated Speech Data:")

    # 한국어 음소 예제
    korean_phonemes = ["sil", "a", "n", "n", "y", "eo", "ng", "h", "a", "s", "e", "y", "o", "sil"]
    phoneme_to_id = {p: i for i, p in enumerate(set(korean_phonemes))}
    id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

    print(f"   Text: '안녕하세요' (Hello in Korean)")
    print(f"   Phonemes: {korean_phonemes}")
    print(f"   Vocabulary size: {len(phoneme_to_id)}")

    # 음성 특징 시뮬레이션 (실제로는 MFCC, 멜 스펙트로그램 등)
    num_frames = 150  # 1.5초 @ 100fps
    feature_dim = 40

    # 각 음소마다 다른 특징 패턴
    speech_features = []
    true_alignment = []

    frame_idx = 0
    for phoneme in korean_phonemes:
        phoneme_id = phoneme_to_id[phoneme]

        # 음소별 지속시간 (실제로는 데이터에서 추출)
        if phoneme == "sil":
            duration = np.random.randint(8, 15)  # 침묵은 길게
        elif phoneme in ["a", "e", "o", "y"]:
            duration = np.random.randint(12, 20)  # 모음은 길게
        else:
            duration = np.random.randint(6, 12)   # 자음은 짧게

        # 음소별 특징 생성 (각 음소마다 고유한 패턴)
        base_feature = torch.randn(feature_dim) * 0.5
        for _ in range(duration):
            if frame_idx < num_frames:
                noise = torch.randn(feature_dim) * 0.1
                frame_feature = base_feature + noise
                speech_features.append(frame_feature)
                true_alignment.append(phoneme_id)
                frame_idx += 1

    # 부족한 프레임 채우기
    while len(speech_features) < num_frames:
        speech_features.append(torch.zeros(feature_dim))
        true_alignment.append(phoneme_to_id["sil"])

    speech_features = torch.stack(speech_features[:num_frames])
    true_alignment = torch.tensor(true_alignment[:num_frames])

    print(f"   Speech features shape: {speech_features.shape}")
    print(f"   True alignment length: {len(true_alignment)}")

    # 2. HMM 기반 정렬 모델
    print("\n2. HMM Alignment Model:")

    num_phonemes = len(phoneme_to_id)

    # Gaussian HMM으로 음소 모델링
    hmm_model = GaussianHMMLayer(
        num_states=num_phonemes,
        feature_dim=feature_dim,
        covariance_type='diag',
        learnable_transitions=True,
        transition_type="left_to_right"
    )

    print(f"   Model type: Gaussian HMM")
    print(f"   Number of phonemes: {num_phonemes}")
    print(f"   Feature dimension: {feature_dim}")

    # 3. 모델 훈련
    print("\n3. Training Alignment Model:")

    optimizer = optim.Adam(hmm_model.parameters(), lr=0.01)

    # 훈련 데이터 준비
    batch_speech = speech_features.unsqueeze(0)  # (1, T, D)

    training_losses = []

    for epoch in range(50):  # 빠른 데모를 위해 50 에포크
        optimizer.zero_grad()

        # Forward pass
        posteriors = hmm_model(batch_speech)

        # 지도학습: 실제 정렬과의 cross-entropy loss
        loss = nn.CrossEntropyLoss()(
            posteriors.view(-1, num_phonemes),
            true_alignment
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:2d}: Loss = {loss.item():.4f}")

    # 4. 정렬 결과 평가
    print("\n4. Alignment Evaluation:")

    with torch.no_grad():
        hmm_model.eval()

        # HMM 정렬 수행
        predicted_posteriors = hmm_model(batch_speech)
        predicted_alignment = torch.argmax(predicted_posteriors, dim=-1)[0]

        # 정확도 계산
        accuracy = (predicted_alignment == true_alignment).float().mean()
        print(f"   Alignment accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

        # 음소별 지속시간 분석
        true_durations = compute_state_durations(true_alignment)
        pred_durations = compute_state_durations(predicted_alignment)

        print(f"\n   Duration comparison:")
        print(f"   {'Phoneme':>8} {'True':>6} {'Pred':>6} {'Error':>7}")
        print("   " + "-" * 30)

        for phoneme_id in range(min(8, num_phonemes)):  # 처음 8개만 표시
            phoneme = id_to_phoneme[phoneme_id]
            true_dur = true_durations[phoneme_id].item() if phoneme_id < len(true_durations) else 0
            pred_dur = pred_durations[phoneme_id].item() if phoneme_id < len(pred_durations) else 0
            error = abs(true_dur - pred_dur)

            print(f"   {phoneme:>8} {true_dur:>6} {pred_dur:>6} {error:>7}")

    # 5. 결과 시각화 (텍스트 기반)
    print("\n5. Alignment Visualization:")

    # 처음 50 프레임의 정렬 비교
    print("   Frame-by-frame alignment (first 50 frames):")
    print("   Frame:  ", end="")
    for i in range(0, 50, 5):
        print(f"{i:>3}", end="")
    print()

    print("   True:   ", end="")
    for i in range(0, 50, 5):
        phoneme = id_to_phoneme[true_alignment[i].item()][:2]
        print(f"{phoneme:>3}", end="")
    print()

    print("   Pred:   ", end="")
    for i in range(0, 50, 5):
        phoneme = id_to_phoneme[predicted_alignment[i].item()][:2]
        print(f"{phoneme:>3}", end="")
    print()

    # 6. 실제 응용 가이드
    print("\n6. 💡 Real-world Application Guide:")
    print("   📊 Data Preparation:")
    print("      - Extract MFCC/mel-spectrogram from audio")
    print("      - Obtain phoneme transcriptions")
    print("      - Align text and audio (forced alignment)")

    print("   🎯 Model Training:")
    print("      - Use larger datasets (hours of speech)")
    print("      - Add speaker adaptation")
    print("      - Include prosodic features")

    print("   🚀 Deployment:")
    print("      - Optimize for real-time inference")
    print("      - Use quantization for mobile devices")
    print("      - Cache frequently used models")

    return hmm_model, accuracy, training_losses


def main():
    """모든 튜토리얼 실행"""
    print("🎓 PyTorch HMM Basic Tutorial")
    print("=" * 70)

    # 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)

    print("Welcome to PyTorch HMM Basic Tutorial!")
    print("This tutorial covers essential concepts and practical usage.")

    try:
        # 튜토리얼 1: 기본 사용법
        hmm, observations, posteriors, states = tutorial_1_basic_hmm()

        # 튜토리얼 2: 알고리즘 비교
        soft_states, hard_states = tutorial_2_forward_backward_vs_viterbi()

        # 튜토리얼 3: 신경망 통합
        model, losses = tutorial_3_hmm_layer_integration()

        # 튜토리얼 4: 배치 처리
        tutorial_4_batch_processing()

        # 튜토리얼 5: GPU 사용법
        tutorial_5_gpu_usage()

        # 튜토리얼 6: 실제 응용
        tts_model, accuracy, training_losses = tutorial_6_real_world_application()

        # 최종 요약
        print("\n" + "=" * 70)
        print("🎉 Tutorial Completed Successfully!")
        print("=" * 70)

        print("\nWhat you've learned:")
        print("✓ Basic HMM operations (forward-backward, Viterbi)")
        print("✓ Algorithm trade-offs and when to use each")
        print("✓ Integration with PyTorch neural networks")
        print("✓ Efficient batch processing techniques")
        print("✓ GPU acceleration and optimization")
        print("✓ Real-world application development")

        print(f"\nKey Results:")
        print(f"• Final TTS model alignment accuracy: {accuracy:.1%}")
        print(f"• Learned transition parameters successfully")
        print(f"• Demonstrated {len(losses)} epochs of training")

        print(f"\nNext Steps:")
        print("📚 Explore advanced_features_demo.py for cutting-edge features")
        print("🔬 Try the integration tests: python tests/test_integration.py")
        print("🚀 Build your own speech synthesis application!")

    except Exception as e:
        print(f"\n❌ Tutorial failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
