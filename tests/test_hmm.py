import torch
import numpy as np
import pytest
import sys
import os

# Add parent directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytorch_hmm import (
    HMM, HMMPyTorch, HMMLayer, GaussianHMMLayer,
    create_left_to_right_matrix, create_transition_matrix,
    compute_state_durations
)


class TestHMM:
    """기본 HMM 클래스 테스트"""
    
    def test_hmm_initialization(self):
        """HMM 초기화 테스트"""
        # Create a simple transition matrix
        P = torch.tensor([[0.7, 0.3], 
                         [0.4, 0.6]], dtype=torch.float32)
        
        hmm = HMM(P)
        
        assert hmm.K == 2
        assert torch.allclose(hmm.P.sum(dim=1), torch.ones(2))
        assert torch.allclose(hmm.p0.sum(), torch.tensor(1.0))
    
    def test_hmm_with_custom_initial_probs(self):
        """커스텀 초기 확률로 HMM 초기화 테스트"""
        P = torch.eye(3)
        p0 = torch.tensor([0.5, 0.3, 0.2])
        
        hmm = HMM(P, p0)
        
        assert torch.allclose(hmm.p0, p0)
    
    def test_hmm_device_transfer(self):
        """Device 이동 테스트"""
        P = torch.rand(3, 3)
        hmm = HMM(P, device='cpu')
        
        assert hmm.P.device.type == 'cpu'
        
        # GPU가 사용 가능한 경우에만 테스트
        if torch.cuda.is_available():
            hmm_gpu = HMM(P, device='cuda')
            assert hmm_gpu.P.device.type == 'cuda'


class TestHMMPyTorch:
    """HMMPyTorch 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전 실행"""
        # 3-state left-to-right HMM
        self.P = create_left_to_right_matrix(3, self_loop_prob=0.7)
        self.hmm = HMMPyTorch(self.P)
    
    def test_forward_backward(self):
        """Forward-backward 알고리즘 테스트"""
        # Generate random observations
        batch_size, seq_len, num_states = 2, 10, 3
        observations = torch.rand(batch_size, seq_len, num_states)
        
        posterior, forward, backward = self.hmm.forward_backward(observations)
        
        # Check shapes
        assert posterior.shape == (batch_size, seq_len, num_states)
        assert forward.shape == (batch_size, seq_len, num_states)
        assert backward.shape == (batch_size, seq_len, num_states)
        
        # Check that posteriors sum to 1
        assert torch.allclose(posterior.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)
    
    def test_viterbi_decode(self):
        """Viterbi 디코딩 테스트"""
        seq_len, num_states = 15, 3
        observations = torch.rand(seq_len, num_states)
        
        states, scores = self.hmm.viterbi_decode(observations)
        
        # Check shapes
        assert states.shape == (seq_len,)
        assert scores.shape == (seq_len, num_states)
        
        # Check that states are valid indices
        assert torch.all(states >= 0)
        assert torch.all(states < num_states)
        
        # For left-to-right model, states should be non-decreasing
        assert torch.all(states[1:] >= states[:-1])
    
    def test_batch_viterbi_decode(self):
        """배치 Viterbi 디코딩 테스트"""
        batch_size, seq_len, num_states = 3, 12, 3
        observations = torch.rand(batch_size, seq_len, num_states)
        
        states, scores = self.hmm.viterbi_decode(observations)
        
        # Check shapes
        assert states.shape == (batch_size, seq_len)
        assert scores.shape == (batch_size, seq_len, num_states)
    
    def test_compute_likelihood(self):
        """Likelihood 계산 테스트"""
        batch_size, seq_len, num_states = 2, 8, 3
        observations = torch.rand(batch_size, seq_len, num_states)
        
        log_likelihood = self.hmm.compute_likelihood(observations)
        
        # Check shape
        assert log_likelihood.shape == (batch_size,)
        
        # Likelihood should be finite
        assert torch.all(torch.isfinite(log_likelihood))
    
    def test_sampling(self):
        """시퀀스 샘플링 테스트"""
        seq_length, batch_size = 20, 4
        
        states, observations = self.hmm.sample(seq_length, batch_size)
        
        # Check shapes
        assert states.shape == (batch_size, seq_length)
        assert observations.shape == (batch_size, seq_length, self.hmm.K)
        
        # Check that observations are one-hot
        assert torch.allclose(observations.sum(dim=-1), torch.ones(batch_size, seq_length))


class TestHMMLayer:
    """HMMLayer 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전 실행"""
        self.num_states = 4
        self.layer = HMMLayer(self.num_states, learnable_transitions=True)
    
    def test_layer_initialization(self):
        """레이어 초기화 테스트"""
        assert self.layer.num_states == self.num_states
        assert self.layer.log_transition_logits is not None
        assert self.layer.log_initial_logits is not None
    
    def test_forward_pass(self):
        """Forward pass 테스트"""
        batch_size, seq_len = 3, 15
        x = torch.randn(batch_size, seq_len, self.num_states)
        
        # Training mode
        self.layer.train()
        posteriors = self.layer(x)
        
        assert posteriors.shape == (batch_size, seq_len, self.num_states)
        assert torch.allclose(posteriors.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)
    
    def test_inference_mode(self):
        """Inference 모드 테스트"""
        seq_len = 12
        x = torch.randn(seq_len, self.num_states)
        
        # Inference mode
        self.layer.eval()
        posteriors, alignment = self.layer(x, return_alignment=True)
        
        assert posteriors.shape == (1, seq_len, self.num_states)
        assert alignment.shape == (1, seq_len)
    
    def test_loss_computation(self):
        """Loss 계산 테스트"""
        batch_size, seq_len = 2, 10
        observations = torch.rand(batch_size, seq_len, self.num_states)
        
        # Unsupervised loss
        loss = self.layer.compute_loss(observations)
        assert loss.dim() == 0  # scalar
        assert torch.isfinite(loss)
        
        # Supervised loss
        target_alignment = torch.randint(0, self.num_states, (batch_size, seq_len))
        supervised_loss = self.layer.compute_loss(observations, target_alignment)
        assert supervised_loss.dim() == 0
        assert torch.isfinite(supervised_loss)
    
    def test_parameter_learning(self):
        """파라미터 학습 테스트"""
        batch_size, seq_len = 2, 10
        observations = torch.rand(batch_size, seq_len, self.num_states)
        
        optimizer = torch.optim.Adam(self.layer.parameters(), lr=0.01)
        
        # Get initial parameters
        initial_transitions = self.layer.get_transition_matrix().clone()
        
        # Training step
        self.layer.train()
        loss = self.layer.compute_loss(observations)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Parameters should have changed
        updated_transitions = self.layer.get_transition_matrix()
        assert not torch.allclose(initial_transitions, updated_transitions)


class TestGaussianHMMLayer:
    """GaussianHMMLayer 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전 실행"""
        self.num_states = 3
        self.feature_dim = 5
        self.layer = GaussianHMMLayer(
            self.num_states, 
            self.feature_dim, 
            covariance_type='diag'
        )
    
    def test_gaussian_layer_initialization(self):
        """가우시안 레이어 초기화 테스트"""
        assert self.layer.num_states == self.num_states
        assert self.layer.feature_dim == self.feature_dim
        assert self.layer.means.shape == (self.num_states, self.feature_dim)
    
    def test_gaussian_forward_pass(self):
        """가우시안 모델 forward pass 테스트"""
        batch_size, seq_len = 2, 12
        observations = torch.randn(batch_size, seq_len, self.feature_dim)
        
        posteriors = self.layer(observations)
        
        assert posteriors.shape == (batch_size, seq_len, self.num_states)
        assert torch.allclose(posteriors.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)
    
    def test_gaussian_loss_computation(self):
        """가우시안 모델 loss 계산 테스트"""
        batch_size, seq_len = 2, 8
        observations = torch.randn(batch_size, seq_len, self.feature_dim)
        
        loss = self.layer.compute_loss(observations)
        
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestUtilityFunctions:
    """유틸리티 함수 테스트"""
    
    def test_create_left_to_right_matrix(self):
        """Left-to-right matrix 생성 테스트"""
        num_states = 5
        self_loop_prob = 0.8
        
        P = create_left_to_right_matrix(num_states, self_loop_prob)
        
        # Check shape
        assert P.shape == (num_states, num_states)
        
        # Check row sums to 1
        assert torch.allclose(P.sum(dim=1), torch.ones(num_states))
        
        # Check left-to-right structure (upper triangular)
        for i in range(num_states):
            for j in range(i):
                assert P[i, j] == 0.0  # No backward transitions
    
    def test_create_transition_matrix_types(self):
        """다양한 transition matrix 타입 테스트"""
        num_states = 4
        
        # Test different types
        for transition_type in ["ergodic", "left_to_right", "left_to_right_skip", "circular"]:
            P = create_transition_matrix(num_states, transition_type=transition_type)
            
            # Check basic properties
            assert P.shape == (num_states, num_states)
            assert torch.allclose(P.sum(dim=1), torch.ones(num_states))
            assert torch.all(P >= 0)
    
    def test_compute_state_durations(self):
        """상태 지속시간 계산 테스트"""
        # Example state sequence: [0, 0, 0, 1, 1, 2, 2, 2, 2]
        state_sequence = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        
        durations = compute_state_durations(state_sequence)
        expected_durations = torch.tensor([3, 2, 4])
        
        assert torch.equal(durations, expected_durations)
    
    def test_empty_state_sequence(self):
        """빈 상태 시퀀스 테스트"""
        empty_sequence = torch.tensor([])
        durations = compute_state_durations(empty_sequence)
        
        assert len(durations) == 0


class TestCompatibilityWithOriginal:
    """원본 TensorFlow 구현과의 호환성 테스트"""
    
    def test_wikipedia_viterbi_example(self):
        """Wikipedia Viterbi 예제 재현"""
        # Simplified version of the classic weather prediction example
        # States: Rainy=0, Sunny=1
        # Observations: walk=0, shop=1, clean=2
        
        # Transition probabilities
        P = torch.tensor([[0.7, 0.3],   # Rainy -> [Rainy, Sunny]
                         [0.4, 0.6]])   # Sunny -> [Rainy, Sunny]
        
        # Initial probabilities  
        p0 = torch.tensor([0.6, 0.4])  # [Rainy, Sunny]
        
        hmm = HMMPyTorch(P, p0)
        
        # Emission probabilities for observations [walk, shop, clean]
        # Given states [Rainy, Sunny]
        observations = torch.tensor([
            [0.1, 0.4, 0.5],  # Rainy state emission probs
            [0.6, 0.3, 0.1]   # Sunny state emission probs
        ]).T  # Transpose to get (3, 2) shape
        
        states, scores = hmm.viterbi_decode(observations)
        
        # Basic sanity checks
        assert len(states) == 3
        assert torch.all(states >= 0)
        assert torch.all(states < 2)


def run_performance_test():
    """성능 테스트 (옵션)"""
    print("Running performance tests...")
    
    # Large sequence test
    num_states = 10
    batch_size = 16
    seq_len = 100
    
    P = create_left_to_right_matrix(num_states)
    hmm = HMMPyTorch(P)
    
    observations = torch.rand(batch_size, seq_len, num_states)
    
    import time
    
    # Forward-backward timing
    start_time = time.time()
    posterior, _, _ = hmm.forward_backward(observations)
    fb_time = time.time() - start_time
    
    # Viterbi timing
    start_time = time.time()
    states, _ = hmm.viterbi_decode(observations)
    viterbi_time = time.time() - start_time
    
    print(f"Forward-backward time: {fb_time:.4f}s")
    print(f"Viterbi time: {viterbi_time:.4f}s")
    print(f"Posterior shape: {posterior.shape}")
    print(f"States shape: {states.shape}")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
    
    # Optionally run performance test
    run_performance_test()
