"""
Comprehensive tests for pytorch_hmm.utils module
"""

import pytest
import torch
import numpy as np
from pytorch_hmm.utils import (
    create_transition_matrix,
    create_left_to_right_matrix,
    create_skip_state_matrix,
    create_phoneme_aware_transitions,
    AdaptiveTransitionMatrix,
    create_duration_constrained_matrix,
    create_gaussian_observation_model,
    gaussian_log_likelihood,
    compute_state_durations,
    interpolate_features,
    create_attention_based_transitions,
    validate_transition_matrix,
    benchmark_transition_operations,
    create_prosody_aware_transitions,
    analyze_transition_patterns
)


class TestTransitionMatrixCreation:
    """Test transition matrix creation functions"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_create_transition_matrix_ergodic(self, device):
        """Test ergodic transition matrix creation"""
        num_states = 5
        P = create_transition_matrix(
            num_states=num_states,
            transition_type="ergodic",
            device=device
        )
        
        assert P.shape == (num_states, num_states)
        assert P.device.type == device.type
        
        # 확률 행렬 검증
        assert torch.allclose(P.sum(dim=1), torch.ones(num_states, device=device))
        assert (P >= 0).all()
        
        # 모든 전이가 가능해야 함
        assert (P > 0).all()
    
    def test_create_transition_matrix_left_to_right(self, device):
        """Test left-to-right transition matrix creation"""
        num_states = 5
        P = create_transition_matrix(
            num_states=num_states,
            transition_type="left_to_right",
            self_loop_prob=0.7,
            forward_prob=0.3,
            device=device
        )
        
        assert P.shape == (num_states, num_states)
        assert torch.allclose(P.sum(dim=1), torch.ones(num_states, device=device))
        
        # Left-to-right 구조 검증
        for i in range(num_states - 1):
            assert P[i, i] > 0  # Self-loop
            assert P[i, i+1] > 0  # Forward transition
            # 뒤로 가는 전이는 없어야 함
            assert torch.allclose(P[i, :i], torch.zeros(i, device=device))
        
        # 마지막 상태는 자기 자신으로만 전이
        assert P[num_states-1, num_states-1] == 1.0
    
    def test_create_transition_matrix_left_to_right_skip(self, device):
        """Test left-to-right with skip transition matrix"""
        num_states = 6
        P = create_transition_matrix(
            num_states=num_states,
            transition_type="left_to_right_skip",
            self_loop_prob=0.6,
            forward_prob=0.3,
            skip_prob=0.1,
            device=device
        )
        
        assert P.shape == (num_states, num_states)
        assert torch.allclose(P.sum(dim=1), torch.ones(num_states, device=device))
        
        # Skip 전이 검증
        for i in range(num_states - 2):
            assert P[i, i] > 0      # Self-loop
            assert P[i, i+1] > 0    # Forward
            assert P[i, i+2] > 0    # Skip
    
    def test_create_transition_matrix_circular(self, device):
        """Test circular transition matrix"""
        num_states = 4
        P = create_transition_matrix(
            num_states=num_states,
            transition_type="circular",
            device=device
        )
        
        assert P.shape == (num_states, num_states)
        assert torch.allclose(P.sum(dim=1), torch.ones(num_states, device=device))
        
        # 순환 구조 검증
        for i in range(num_states):
            assert P[i, i] > 0  # Self-loop
            assert P[i, (i+1) % num_states] > 0  # Circular forward
    
    def test_create_transition_matrix_invalid_type(self):
        """Test invalid transition type"""
        with pytest.raises(ValueError, match="Unknown transition_type"):
            create_transition_matrix(5, transition_type="invalid")
    
    def test_create_left_to_right_matrix(self, device):
        """Test convenience function for left-to-right matrix"""
        num_states = 4
        self_loop_prob = 0.8
        
        P = create_left_to_right_matrix(
            num_states=num_states,
            self_loop_prob=self_loop_prob,
            device=device
        )
        
        assert P.shape == (num_states, num_states)
        assert P.device.type == device.type
        
        # 확률 검증
        for i in range(num_states - 1):
            assert abs(P[i, i].item() - self_loop_prob) < 0.01
            assert abs(P[i, i+1].item() - (1.0 - self_loop_prob)) < 0.01


class TestAdvancedTransitionMatrices:
    """Test advanced transition matrix functions"""
    
    def test_create_skip_state_matrix(self):
        """Test skip-state transition matrix"""
        num_states = 6
        P = create_skip_state_matrix(
            num_states=num_states,
            self_loop_prob=0.6,
            forward_prob=0.3,
            skip_prob=0.1,
            max_skip=2
        )
        
        assert P.shape == (num_states, num_states)
        assert torch.allclose(P.sum(dim=1), torch.ones(num_states))
        
        # Skip 전이 검증
        for i in range(num_states - 2):
            assert P[i, i] > 0      # Self-loop
            assert P[i, i+1] > 0    # Forward
            # Skip transitions should exist
            assert P[i, i+2] > 0 if i+2 < num_states else True
    
    def test_create_phoneme_aware_transitions(self):
        """Test phoneme-aware transition matrix"""
        phoneme_durations = [5.0, 8.0, 3.0, 12.0]
        
        P = create_phoneme_aware_transitions(
            phoneme_durations=phoneme_durations,
            duration_variance=0.2
        )
        
        num_phonemes = len(phoneme_durations)
        assert P.shape == (num_phonemes, num_phonemes)
        assert torch.allclose(P.sum(dim=1), torch.ones(num_phonemes))
        
        # 긴 음소일수록 높은 self-loop 확률
        for i in range(num_phonemes - 1):
            self_prob = P[i, i].item()
            assert 0.3 <= self_prob <= 0.95


class TestAdaptiveTransitionMatrix:
    """Test adaptive transition matrix module"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_adaptive_transition_matrix_init(self, device):
        """Test AdaptiveTransitionMatrix initialization"""
        num_states = 5
        context_dim = 64
        
        model = AdaptiveTransitionMatrix(
            num_states=num_states,
            context_dim=context_dim
        ).to(device)
        
        assert model.num_states == num_states
        assert model.context_dim == context_dim
        
        # 기본 전이 행렬 확인
        assert model.base_transition.shape == (num_states, num_states)
    
    def test_adaptive_transition_matrix_forward(self, device):
        """Test AdaptiveTransitionMatrix forward pass"""
        num_states = 4
        context_dim = 32
        batch_size = 2
        seq_len = 10
        
        model = AdaptiveTransitionMatrix(
            num_states=num_states,
            context_dim=context_dim
        ).to(device)
        
        # 컨텍스트 없이
        P_base = model()
        assert P_base.shape == (num_states, num_states)
        assert torch.allclose(P_base.sum(dim=1), torch.ones(num_states, device=device))
        
        # 컨텍스트와 함께
        context = torch.randn(batch_size, seq_len, context_dim, device=device)
        P_adaptive = model(context)
        assert P_adaptive.shape == (batch_size, seq_len, num_states, num_states)
        
        # 각 전이 행렬이 확률 분포인지 확인
        row_sums = P_adaptive.sum(dim=-1)
        expected_sums = torch.ones(batch_size, seq_len, num_states, device=device)
        assert torch.allclose(row_sums, expected_sums, atol=1e-6)


class TestDurationConstrainedMatrix:
    """Test duration-constrained transition matrix"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_create_duration_constrained_matrix_basic(self, device):
        """Test basic duration-constrained matrix"""
        num_states = 6
        min_duration = 2
        max_duration = 4
        
        P = create_duration_constrained_matrix(
            num_states=num_states,
            min_duration=min_duration,
            max_duration=max_duration,
            device=device
        )
        
        # 확장된 상태 공간 크기 계산
        extended_states = num_states * max_duration
        assert P.shape == (extended_states, extended_states)
        assert P.device.type == device.type
        assert torch.allclose(P.sum(dim=1), torch.ones(extended_states, device=device))
    
    def test_create_duration_constrained_matrix_no_max(self, device):
        """Test duration-constrained matrix without max duration"""
        num_states = 4
        min_duration = 3
        
        P = create_duration_constrained_matrix(
            num_states=num_states,
            min_duration=min_duration,
            device=device
        )
        
        # max_duration이 None이면 min_duration * 3으로 설정됨
        expected_max = min_duration * 3
        extended_states = num_states * expected_max
        assert P.shape == (extended_states, extended_states)


class TestObservationModels:
    """Test observation model functions"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_create_gaussian_observation_model(self, device):
        """Test Gaussian observation model creation"""
        num_states = 4
        feature_dim = 8
        
        means, covariances = create_gaussian_observation_model(
            num_states=num_states,
            feature_dim=feature_dim,
            device=device
        )
        
        assert means.shape == (num_states, feature_dim)
        assert covariances.shape == (num_states, feature_dim, feature_dim)
        assert means.device.type == device.type
        assert covariances.device.type == device.type
        
        # 공분산 행렬이 positive definite인지 확인
        for k in range(num_states):
            eigenvals = torch.linalg.eigvals(covariances[k])
            assert (eigenvals.real > 0).all()
    
    def test_create_gaussian_observation_model_with_params(self, device):
        """Test Gaussian observation model with provided parameters"""
        num_states = 3
        feature_dim = 4
        
        # 사용자 정의 평균과 공분산
        means = torch.randn(num_states, feature_dim, device=device)
        covariances = torch.stack([
            torch.eye(feature_dim, device=device) * (i + 1) 
            for i in range(num_states)
        ])
        
        means_out, covariances_out = create_gaussian_observation_model(
            num_states=num_states,
            feature_dim=feature_dim,
            means=means,
            covariances=covariances,
            device=device
        )
        
        assert torch.allclose(means_out, means)
        assert torch.allclose(covariances_out, covariances)
    
    def test_gaussian_log_likelihood(self, device):
        """Test Gaussian log-likelihood computation"""
        batch_size = 2
        seq_len = 10
        feature_dim = 6
        num_states = 4
        
        observations = torch.randn(batch_size, seq_len, feature_dim, device=device)
        means = torch.randn(num_states, feature_dim, device=device)
        covariances = torch.stack([
            torch.eye(feature_dim, device=device) * 0.5 
            for _ in range(num_states)
        ])
        
        log_likes = gaussian_log_likelihood(observations, means, covariances)
        
        assert log_likes.shape == (batch_size, seq_len, num_states)
        assert log_likes.device.type == device.type
        
        # 로그 우도는 음수여야 함
        assert (log_likes <= 0).all()
    
    def test_gaussian_log_likelihood_single_batch(self, device):
        """Test Gaussian log-likelihood with single batch"""
        seq_len = 8
        feature_dim = 4
        num_states = 3
        
        observations = torch.randn(seq_len, feature_dim, device=device)
        means = torch.randn(num_states, feature_dim, device=device)
        covariances = torch.stack([
            torch.eye(feature_dim, device=device) 
            for _ in range(num_states)
        ])
        
        log_likes = gaussian_log_likelihood(observations, means, covariances)
        
        assert log_likes.shape == (seq_len, num_states)


class TestSequenceUtilities:
    """Test sequence processing utilities"""
    
    def test_compute_state_durations(self):
        """Test state duration computation"""
        # 테스트 시퀀스: [0, 0, 0, 1, 1, 2, 2, 2, 2, 1]
        state_sequence = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 1])
        
        durations = compute_state_durations(state_sequence)
        expected = torch.tensor([3, 2, 4, 1])  
        
        assert torch.equal(durations, expected)
    
    def test_compute_state_durations_empty(self):
        """Test state duration computation with empty sequence"""
        state_sequence = torch.tensor([])
        durations = compute_state_durations(state_sequence)
        
        assert len(durations) == 0
    
    def test_compute_state_durations_single(self):
        """Test state duration computation with single state"""
        state_sequence = torch.tensor([5])
        durations = compute_state_durations(state_sequence)
        
        assert torch.equal(durations, torch.tensor([1]))
    
    def test_interpolate_features(self):
        """Test feature interpolation"""
        # 원본 특징: 6 프레임, 4 차원
        features = torch.randn(6, 4)
        source_durations = torch.tensor([2, 3, 1])  # 3개 상태
        target_durations = torch.tensor([3, 2, 2])  # 새로운 지속시간
        
        interpolated = interpolate_features(features, source_durations, target_durations)
        
        expected_length = target_durations.sum().item()
        assert interpolated.shape == (expected_length, 4)
        
        # 각 상태별 특징이 올바르게 복제되었는지 확인
        assert interpolated.shape[0] == 7  # 3 + 2 + 2


class TestAdvancedUtilities:
    """Test advanced utility functions"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_create_attention_based_transitions(self, device):
        """Test attention-based transition creation"""
        num_states = 4
        attention_dim = 32
        
        attention_module = create_attention_based_transitions(
            num_states=num_states,
            attention_dim=attention_dim
        ).to(device)
        
        batch_size = 2
        seq_len = 8
        context = torch.randn(batch_size, seq_len, attention_dim, device=device)
        
        transitions = attention_module(context)
        
        assert transitions.shape == (batch_size, seq_len, num_states, num_states)
        
        # 각 전이 행렬이 확률 분포인지 확인
        row_sums = transitions.sum(dim=-1)
        expected_sums = torch.ones(batch_size, seq_len, num_states, device=device)
        assert torch.allclose(row_sums, expected_sums, atol=1e-6)
    
    def test_validate_transition_matrix(self):
        """Test transition matrix validation"""
        # 유효한 전이 행렬
        P_valid = torch.tensor([
            [0.7, 0.3, 0.0],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0]
        ])
        
        validation_result = validate_transition_matrix(P_valid)
        
        assert validation_result['is_stochastic'] == True
        assert validation_result['is_non_negative'] == True
        assert validation_result['is_square'] == True
        
        # 무효한 전이 행렬 (확률 합이 1이 아님)
        P_invalid = torch.tensor([
            [0.5, 0.3, 0.0],  # 합이 0.8
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0]
        ])
        
        validation_result = validate_transition_matrix(P_invalid)
        assert validation_result['is_stochastic'] == False
    
    def test_benchmark_transition_operations(self):
        """Test transition operation benchmarking"""
        num_states_list = [3, 5]
        num_trials = 5  # 빠른 테스트를 위해 적은 수로 설정
        
        results = benchmark_transition_operations(
            num_states_list=num_states_list,
            num_trials=num_trials
        )
        
        assert 'creation_time' in results
        assert 'validation_time' in results
        
        for num_states in num_states_list:
            assert num_states in results['creation_time']
            assert num_states in results['validation_time']
            assert results['creation_time'][num_states] > 0
            assert results['validation_time'][num_states] > 0
    
    def test_create_prosody_aware_transitions(self):
        """Test prosody-aware transition creation"""
        seq_len = 20
        num_states = 6
        
        f0_contour = torch.randn(seq_len) * 50 + 150  # F0 around 150Hz
        energy_contour = torch.randn(seq_len) * 0.2 + 0.5  # Energy around 0.5
        
        P = create_prosody_aware_transitions(
            f0_contour=f0_contour,
            energy_contour=energy_contour,
            num_states=num_states
        )
        
        assert P.shape == (num_states, num_states)
        assert torch.allclose(P.sum(dim=1), torch.ones(num_states))
        assert (P >= 0).all()
    
    def test_analyze_transition_patterns(self):
        """Test transition pattern analysis"""
        # 여러 상태 시퀀스 생성
        state_sequences = [
            torch.tensor([0, 0, 1, 1, 1, 2, 2]),
            torch.tensor([0, 1, 1, 2, 2, 2, 2]),
            torch.tensor([0, 0, 0, 1, 2, 2])
        ]
        
        patterns = analyze_transition_patterns(state_sequences)
        
        assert 'avg_duration' in patterns
        assert 'transition_entropy' in patterns
        assert 'self_loop_ratio' in patterns
        assert 'forward_progression_ratio' in patterns
        
        # 값들이 합리적인 범위에 있는지 확인
        assert patterns['avg_duration'] > 0
        assert 0 <= patterns['self_loop_ratio'] <= 1
        assert 0 <= patterns['forward_progression_ratio'] <= 1


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_single_state_matrix(self):
        """Test transition matrix with single state"""
        P = create_transition_matrix(num_states=1, transition_type="ergodic")
        
        assert P.shape == (1, 1)
        assert P[0, 0] == 1.0
    
    def test_zero_duration_interpolation(self):
        """Test interpolation with zero durations"""
        features = torch.randn(4, 3)
        source_durations = torch.tensor([2, 2])
        target_durations = torch.tensor([0, 4])  # 첫 번째 상태 지속시간 0
        
        # 0 지속시간이 있어도 에러가 발생하지 않아야 함
        interpolated = interpolate_features(features, source_durations, target_durations)
        assert interpolated.shape[0] == 4  # 0 + 4
    
    def test_large_skip_matrix(self):
        """Test skip matrix with large skip values"""
        num_states = 10
        max_skip = 5
        
        P = create_skip_state_matrix(
            num_states=num_states,
            max_skip=max_skip
        )
        
        assert P.shape == (num_states, num_states)
        assert torch.allclose(P.sum(dim=1), torch.ones(num_states))


if __name__ == "__main__":
    pytest.main([__file__])