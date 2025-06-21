import torch
import numpy as np
import pytest
import sys
import os

# Add parent directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytorch_hmm import (
    # Core HMM
    HMM, HMMPyTorch, HMMLayer, GaussianHMMLayer,
    # Neural HMM
    NeuralHMM, ContextualNeuralHMM,
    # Semi-Markov HMM
    SemiMarkovHMM, DurationModel,
    # Alignment
    DTWAligner, CTCAligner,
    # Metrics
    mel_cepstral_distortion, f0_root_mean_square_error, alignment_accuracy,
    comprehensive_speech_evaluation,
    # Utilities
    create_left_to_right_matrix, create_transition_matrix,
    compute_state_durations
)


class TestAdvancedFeatures:
    """새로 구현된 고급 기능들의 통합 테스트"""
    
    def setup_method(self):
        """각 테스트 전 실행"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(42)  # 재현 가능한 결과를 위한 시드 설정
    
    def test_dtw_alignment(self):
        """DTW 정렬 테스트"""
        print("Testing DTW Alignment...")
        
        # 테스트 데이터 생성
        seq_len1, seq_len2, feature_dim = 20, 25, 10
        x = torch.randn(seq_len1, feature_dim)
        y = torch.randn(seq_len2, feature_dim)
        
        # DTW 정렬 수행
        aligner = DTWAligner(distance_fn='euclidean', step_pattern='symmetric')
        path_i, path_j, total_cost = aligner(x, y)
        
        # 결과 검증
        assert len(path_i) == len(path_j), "Path lengths must match"
        assert path_i[0] == 0 and path_i[-1] == seq_len1 - 1, "Path must start and end correctly"
        assert path_j[0] == 0 and path_j[-1] == seq_len2 - 1, "Path must start and end correctly"
        assert torch.isfinite(total_cost), "Total cost must be finite"
        
        print(f"✓ DTW alignment successful. Path length: {len(path_i)}, Cost: {total_cost:.4f}")
    
    def test_ctc_alignment(self):
        """CTC 정렬 테스트"""
        print("Testing CTC Alignment...")
        
        # 테스트 데이터 생성
        batch_size, seq_len, num_classes = 2, 50, 10
        log_probs = torch.log_softmax(torch.randn(seq_len, batch_size, num_classes), dim=-1)
        
        targets = torch.randint(1, num_classes, (batch_size, 5))  # blank=0 제외
        input_lengths = torch.full((batch_size,), seq_len)
        target_lengths = torch.full((batch_size,), 5)
        
        # CTC 정렬 수행
        aligner = CTCAligner(num_classes=num_classes, blank_id=0)
        loss = aligner(log_probs, targets, input_lengths, target_lengths)
        
        # 디코딩 테스트
        decoded = aligner.decode(log_probs, input_lengths)
        
        # 결과 검증
        assert torch.isfinite(loss), "CTC loss must be finite"
        assert len(decoded) == batch_size, "Must decode for each batch item"
        
        print(f"✓ CTC alignment successful. Loss: {loss:.4f}, Decoded lengths: {[len(d) for d in decoded]}")
    
    def test_neural_hmm(self):
        """Neural HMM 테스트"""
        print("Testing Neural HMM...")
        
        # 테스트 파라미터
        num_states, observation_dim, context_dim = 5, 8, 12
        batch_size, seq_len = 2, 20
        
        # Neural HMM 생성
        neural_hmm = NeuralHMM(
            num_states=num_states,
            observation_dim=observation_dim,
            context_dim=context_dim,
            hidden_dim=64,
            transition_type='mlp',
            observation_type='gaussian'
        )
        
        # 테스트 데이터
        observations = torch.randn(batch_size, seq_len, observation_dim)
        context = torch.randn(batch_size, seq_len, context_dim)
        
        # Forward pass
        posteriors, forward, backward = neural_hmm(observations, context)
        
        # Viterbi 디코딩
        states, scores = neural_hmm.viterbi_decode(observations, context)
        
        # 결과 검증
        assert posteriors.shape == (batch_size, seq_len, num_states), f"Wrong posterior shape: {posteriors.shape}"
        assert torch.allclose(posteriors.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5), "Posteriors must sum to 1"
        assert states.shape == (batch_size, seq_len), f"Wrong states shape: {states.shape}"
        assert torch.all(states >= 0) and torch.all(states < num_states), "Invalid state indices"
        
        print(f"✓ Neural HMM successful. Posterior shape: {posteriors.shape}, States shape: {states.shape}")
    
    def test_contextual_neural_hmm(self):
        """Contextual Neural HMM 테스트"""
        print("Testing Contextual Neural HMM...")
        
        # 테스트 파라미터
        num_states, observation_dim = 6, 10
        phoneme_vocab_size = 50
        batch_size, seq_len = 2, 15
        
        # Contextual Neural HMM 생성
        contextual_hmm = ContextualNeuralHMM(
            num_states=num_states,
            observation_dim=observation_dim,
            phoneme_vocab_size=phoneme_vocab_size,
            linguistic_context_dim=32,
            prosody_dim=8
        )
        
        # 테스트 데이터
        observations = torch.randn(batch_size, seq_len, observation_dim)
        phoneme_sequence = torch.randint(0, phoneme_vocab_size, (batch_size, seq_len))
        prosody_features = torch.randn(batch_size, seq_len, 8)
        
        # Forward pass with context
        posteriors, forward, backward = contextual_hmm.forward_with_context(
            observations, phoneme_sequence, prosody_features)
        
        # 결과 검증
        assert posteriors.shape == (batch_size, seq_len, num_states), f"Wrong posterior shape: {posteriors.shape}"
        assert torch.allclose(posteriors.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5), "Posteriors must sum to 1"
        
        print(f"✓ Contextual Neural HMM successful. Posterior shape: {posteriors.shape}")
    
    def test_semi_markov_hmm(self):
        """Semi-Markov HMM 테스트"""
        print("Testing Semi-Markov HMM...")
        
        # 테스트 파라미터
        num_states, observation_dim, max_duration = 4, 6, 10
        
        # HSMM 생성
        hsmm = SemiMarkovHMM(
            num_states=num_states,
            observation_dim=observation_dim,
            max_duration=max_duration,
            duration_distribution='gamma',
            observation_model='gaussian'
        )
        
        # 지속시간 모델 테스트
        state_indices = torch.randint(0, num_states, (3,))
        duration_probs = hsmm.duration_model(state_indices)
        
        assert duration_probs.shape == (3, max_duration), f"Wrong duration prob shape: {duration_probs.shape}"
        
        # 샘플링 테스트
        sampled_durations = hsmm.duration_model.sample(state_indices)
        assert len(sampled_durations) == len(state_indices), "Wrong number of sampled durations"
        assert torch.all(sampled_durations >= 1), "Durations must be at least 1"
        
        # 시퀀스 샘플링 테스트
        state_seq, duration_seq, obs_seq = hsmm.sample(num_states=5, max_length=50)
        
        assert len(state_seq) == len(duration_seq), "State and duration sequences must have same length"
        assert obs_seq.shape[1] == observation_dim, f"Wrong observation dimension: {obs_seq.shape[1]}"
        
        print(f"✓ Semi-Markov HMM successful. Sampled {len(state_seq)} states, {obs_seq.shape[0]} observations")
    
    def test_duration_model(self):
        """Duration Model 테스트"""
        print("Testing Duration Model...")
        
        # 다양한 분포 타입 테스트
        for distribution_type in ['gamma', 'poisson', 'gaussian', 'neural']:
            duration_model = DurationModel(
                num_states=5,
                max_duration=20,
                distribution_type=distribution_type
            )
            
            state_indices = torch.randint(0, 5, (3,))
            
            # 분포 계산
            log_probs = duration_model(state_indices)
            assert log_probs.shape == (3, 20), f"Wrong shape for {distribution_type}: {log_probs.shape}"
            
            # 샘플링
            samples = duration_model.sample(state_indices)
            assert len(samples) == 3, f"Wrong number of samples for {distribution_type}"
            assert torch.all(samples >= 1), f"Invalid samples for {distribution_type}"
            
            print(f"  ✓ {distribution_type} distribution working")
        
        print("✓ Duration Model tests completed")
    
    def test_metrics(self):
        """Metrics 테스트"""
        print("Testing Speech Quality Metrics...")
        
        # 테스트 데이터 생성
        seq_len, mfcc_dim = 100, 13
        mfcc_true = torch.randn(seq_len, mfcc_dim)
        mfcc_pred = mfcc_true + 0.1 * torch.randn(seq_len, mfcc_dim)  # 약간의 노이즈 추가
        
        # MCD 테스트
        mcd = mel_cepstral_distortion(mfcc_true, mfcc_pred)
        assert torch.isfinite(mcd), "MCD must be finite"
        assert mcd >= 0, "MCD must be non-negative"
        
        # F0 RMSE 테스트
        f0_true = torch.abs(torch.randn(seq_len)) * 200 + 100  # 100-300 Hz 범위
        f0_pred = f0_true + 10 * torch.randn(seq_len)  # 10Hz 노이즈
        f0_rmse = f0_root_mean_square_error(f0_true, f0_pred)
        assert torch.isfinite(f0_rmse), "F0 RMSE must be finite"
        assert f0_rmse >= 0, "F0 RMSE must be non-negative"
        
        # 정렬 정확도 테스트
        alignment_true = torch.randint(0, 5, (seq_len,))
        alignment_pred = alignment_true.clone()
        alignment_pred[::10] = (alignment_pred[::10] + 1) % 5  # 10%의 오류 추가
        
        acc = alignment_accuracy(alignment_pred, alignment_true)
        assert 0 <= acc <= 1, "Accuracy must be between 0 and 1"
        assert acc > 0.8, "Accuracy should be reasonably high with small errors"
        
        # 종합 평가 테스트
        predicted_features = {
            'mfcc': mfcc_pred.unsqueeze(0),
            'f0': f0_pred.unsqueeze(0),
            'alignment': alignment_pred
        }
        ground_truth_features = {
            'mfcc': mfcc_true.unsqueeze(0),
            'f0': f0_true.unsqueeze(0),
            'alignment': alignment_true
        }
        
        metrics = comprehensive_speech_evaluation(predicted_features, ground_truth_features)
        
        assert 'mcd' in metrics, "MCD metric missing"
        assert 'f0_rmse' in metrics, "F0 RMSE metric missing"
        assert 'alignment_accuracy' in metrics, "Alignment accuracy metric missing"
        
        print(f"✓ Metrics test successful. MCD: {mcd:.4f}, F0 RMSE: {f0_rmse:.4f}, Accuracy: {acc:.4f}")
    
    def test_integration_workflow(self):
        """전체 워크플로우 통합 테스트"""
        print("Testing Complete Integration Workflow...")
        
        # 1. 기본 HMM으로 시작
        num_states = 5
        transition_matrix = create_left_to_right_matrix(num_states, self_loop_prob=0.7)
        basic_hmm = HMMPyTorch(transition_matrix)
        
        # 2. 관측 데이터 생성
        batch_size, seq_len, obs_dim = 2, 30, num_states
        observations = torch.softmax(torch.randn(batch_size, seq_len, obs_dim), dim=-1)
        
        # 3. 기본 HMM으로 정렬
        posteriors, _, _ = basic_hmm.forward_backward(observations)
        basic_states, _ = basic_hmm.viterbi_decode(observations)
        
        # 4. DTW로 재정렬
        dtw_aligner = DTWAligner()
        ref_features = torch.randn(num_states, obs_dim)
        for b in range(batch_size):
            path_i, path_j, _ = dtw_aligner(ref_features, observations[b])
            assert len(path_i) > 0, "DTW must produce valid path"
        
        # 5. Neural HMM으로 고급 모델링
        neural_hmm = NeuralHMM(
            num_states=num_states,
            observation_dim=num_states,
            context_dim=8,
            hidden_dim=32
        )
        
        context = torch.randn(batch_size, seq_len, 8)
        neural_posteriors, _, _ = neural_hmm(observations, context)
        neural_states, _ = neural_hmm.viterbi_decode(observations, context)
        
        # 6. 결과 평가
        for b in range(batch_size):
            acc = alignment_accuracy(neural_states[b], basic_states[b], tolerance=1)
            assert 0 <= acc <= 1, "Accuracy must be valid"
        
        # 7. 지속시간 분석
        for b in range(batch_size):
            durations = compute_state_durations(basic_states[b])
            assert len(durations) > 0, "Must have some state durations"
            assert torch.all(durations > 0), "All durations must be positive"
        
        print("✓ Complete integration workflow successful!")
    
    def test_batch_processing(self):
        """배치 처리 테스트"""
        print("Testing Batch Processing...")
        
        # 다양한 배치 크기로 테스트
        for batch_size in [1, 4, 8]:
            seq_len, num_states, obs_dim = 25, 6, 10
            
            # Neural HMM
            neural_hmm = NeuralHMM(
                num_states=num_states,
                observation_dim=obs_dim,
                context_dim=5,
                hidden_dim=32
            )
            
            observations = torch.randn(batch_size, seq_len, obs_dim)
            context = torch.randn(batch_size, seq_len, 5)
            
            # 배치 처리
            posteriors, _, _ = neural_hmm(observations, context)
            states, _ = neural_hmm.viterbi_decode(observations, context)
            
            assert posteriors.shape == (batch_size, seq_len, num_states), f"Wrong shape for batch_size={batch_size}"
            assert states.shape == (batch_size, seq_len), f"Wrong states shape for batch_size={batch_size}"
            
            print(f"  ✓ Batch size {batch_size} successful")
        
        print("✓ Batch processing tests completed")
    
    def test_device_compatibility(self):
        """Device 호환성 테스트"""
        print("Testing Device Compatibility...")
        
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            print(f"  Testing on {device}...")
            
            # Neural HMM
            neural_hmm = NeuralHMM(
                num_states=4,
                observation_dim=6,
                context_dim=4,
                hidden_dim=16
            ).to(device)
            
            observations = torch.randn(2, 15, 6, device=device)
            context = torch.randn(2, 15, 4, device=device)
            
            posteriors, _, _ = neural_hmm(observations, context)
            states, _ = neural_hmm.viterbi_decode(observations, context)
            
            assert posteriors.device.type == device, f"Wrong device for posteriors"
            assert states.device.type == device, f"Wrong device for states"
            
            print(f"  ✓ {device.upper()} device successful")
        
        print("✓ Device compatibility tests completed")
    
    def test_error_handling(self):
        """에러 처리 테스트"""
        print("Testing Error Handling...")
        
        # 잘못된 입력 차원
        neural_hmm = NeuralHMM(num_states=5, observation_dim=10, context_dim=8)
        
        try:
            wrong_obs = torch.randn(2, 20, 15)  # Wrong observation dim
            context = torch.randn(2, 20, 8)
            neural_hmm(wrong_obs, context)
            assert False, "Should have raised an error"
        except:
            print("  ✓ Correctly caught dimension mismatch error")
        
        # DTW with empty sequences
        dtw_aligner = DTWAligner()
        try:
            empty_seq = torch.empty(0, 5)
            normal_seq = torch.randn(10, 5)
            dtw_aligner(empty_seq, normal_seq)
            print("  ✓ DTW handles empty sequences gracefully")
        except:
            print("  ✓ DTW correctly rejects empty sequences")
        
        print("✓ Error handling tests completed")


def test_performance_benchmark():
    """성능 벤치마크 테스트"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    import time
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on: {device.upper()}")
    
    # 테스트 설정
    configs = [
        (2, 50, 10, 5),    # Small: batch=2, seq=50, obs=10, states=5
        (4, 100, 20, 10),  # Medium: batch=4, seq=100, obs=20, states=10
        (8, 200, 40, 15),  # Large: batch=8, seq=200, obs=40, states=15
    ]
    
    for batch_size, seq_len, obs_dim, num_states in configs:
        print(f"\nTesting: batch={batch_size}, seq={seq_len}, obs={obs_dim}, states={num_states}")
        
        # 데이터 준비
        observations = torch.randn(batch_size, seq_len, num_states, device=device)
        context = torch.randn(batch_size, seq_len, obs_dim//2, device=device)
        
        # Basic HMM 벤치마크
        transition_matrix = create_left_to_right_matrix(num_states)
        basic_hmm = HMMPyTorch(transition_matrix).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            posteriors, _, _ = basic_hmm.forward_backward(observations)
        basic_time = time.time() - start_time
        
        # Neural HMM 벤치마크
        neural_hmm = NeuralHMM(
            num_states=num_states,
            observation_dim=num_states,
            context_dim=obs_dim//2,
            hidden_dim=64
        ).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            neural_posteriors, _, _ = neural_hmm(observations, context)
        neural_time = time.time() - start_time
        
        # DTW 벤치마크 (단일 시퀀스)
        dtw_aligner = DTWAligner()
        ref_seq = torch.randn(num_states, obs_dim, device=device)
        
        start_time = time.time()
        with torch.no_grad():
            path_i, path_j, _ = dtw_aligner(ref_seq, observations[0])
        dtw_time = time.time() - start_time
        
        # 결과 출력
        frames_per_sec_basic = (batch_size * seq_len) / basic_time
        frames_per_sec_neural = (batch_size * seq_len) / neural_time
        frames_per_sec_dtw = seq_len / dtw_time
        
        print(f"  Basic HMM:   {basic_time:.4f}s ({frames_per_sec_basic:.0f} frames/sec)")
        print(f"  Neural HMM:  {neural_time:.4f}s ({frames_per_sec_neural:.0f} frames/sec)")
        print(f"  DTW:         {dtw_time:.4f}s ({frames_per_sec_dtw:.0f} frames/sec)")
        
        # 실시간 처리 가능 여부 (80fps 기준)
        realtime_threshold = 80
        print(f"  Realtime capable (>80fps): Basic={frames_per_sec_basic>realtime_threshold}, "
              f"Neural={frames_per_sec_neural>realtime_threshold}, DTW={frames_per_sec_dtw>realtime_threshold}")
    
    print("\n" + "="*50)


def run_comprehensive_tests():
    """모든 테스트 실행"""
    print("Starting Comprehensive PyTorch HMM Tests...")
    print("="*60)
    
    test_suite = TestAdvancedFeatures()
    test_suite.setup_method()
    
    # 모든 테스트 메서드 실행
    test_methods = [
        test_suite.test_dtw_alignment,
        test_suite.test_ctc_alignment,
        test_suite.test_neural_hmm,
        test_suite.test_contextual_neural_hmm,
        test_suite.test_semi_markov_hmm,
        test_suite.test_duration_model,
        test_suite.test_metrics,
        test_suite.test_integration_workflow,
        test_suite.test_batch_processing,
        test_suite.test_device_compatibility,
        test_suite.test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        
        # 성공적으로 모든 테스트가 통과하면 성능 벤치마크 실행
        test_performance_benchmark()
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    # 전체 테스트 실행
    success = run_comprehensive_tests()
    
    # pytest 호환성을 위한 반환값
    if not success:
        sys.exit(1)
