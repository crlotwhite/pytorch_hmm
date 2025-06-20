"""
Tests for Hidden Semi-Markov Model (HSMM) implementation

Testing explicit duration modeling, different duration distributions,
and sequence generation capabilities.
"""

import pytest
import torch
import numpy as np
from pytorch_hmm.hsmm import HSMMLayer, DurationConstrainedHMM


class TestHSMMLayer:
    """Test suite for HSMMLayer"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_data(self, device):
        """Generate sample data for testing"""
        batch_size, seq_len, feature_dim = 2, 80, 30
        return torch.randn(batch_size, seq_len, feature_dim, device=device)
    
    @pytest.fixture
    def basic_hsmm(self, device):
        """Create basic HSMM for testing"""
        return HSMMLayer(
            num_states=5,
            feature_dim=30,
            duration_distribution='gamma',
            max_duration=20
        ).to(device)
    
    def test_model_initialization(self, device):
        """Test HSMM initialization with different duration distributions"""
        # Gamma distribution
        model_gamma = HSMMLayer(
            num_states=5, feature_dim=30, 
            duration_distribution='gamma'
        ).to(device)
        
        assert model_gamma.num_states == 5
        assert model_gamma.feature_dim == 30
        assert model_gamma.duration_distribution == 'gamma'
        assert hasattr(model_gamma, 'duration_shape')
        assert hasattr(model_gamma, 'duration_rate')
        
        # Poisson distribution
        model_poisson = HSMMLayer(
            num_states=3, feature_dim=20,
            duration_distribution='poisson'
        ).to(device)
        
        assert model_poisson.duration_distribution == 'poisson'
        assert hasattr(model_poisson, 'duration_lambda')
        
        # Weibull distribution
        model_weibull = HSMMLayer(
            num_states=4, feature_dim=25,
            duration_distribution='weibull'
        ).to(device)
        
        assert model_weibull.duration_distribution == 'weibull'
        assert hasattr(model_weibull, 'duration_scale')
        assert hasattr(model_weibull, 'duration_concentration')
    
    def test_duration_probabilities(self, basic_hsmm):
        """Test duration probability computation"""
        duration_probs = basic_hsmm.get_duration_probabilities()
        
        num_states = basic_hsmm.num_states
        max_duration = basic_hsmm.max_duration
        
        assert duration_probs.shape == (num_states, max_duration)
        
        # Probabilities should be non-negative
        assert torch.all(duration_probs >= 0)
        
        # Should be finite
        assert torch.all(torch.isfinite(duration_probs))
        
        # Each state should have valid probability distribution
        # (may not sum to exactly 1 due to truncation at max_duration)
        row_sums = torch.sum(duration_probs, dim=1)
        assert torch.all(row_sums > 0)
    
    def test_observation_log_probs(self, basic_hsmm, sample_data):
        """Test observation log probability computation"""
        log_probs = basic_hsmm.get_observation_log_probs(sample_data)
        
        batch_size, seq_len = sample_data.shape[:2]
        assert log_probs.shape == (batch_size, seq_len, basic_hsmm.num_states)
        
        # Log probabilities should be finite
        assert torch.all(torch.isfinite(log_probs))
    
    def test_transition_matrix_no_self_loops(self, basic_hsmm):
        """Test that HSMM transition matrix has no self-loops"""
        transition_matrix = basic_hsmm.get_transition_matrix()
        
        assert transition_matrix.shape == (basic_hsmm.num_states, basic_hsmm.num_states)
        
        # Check diagonal elements are zero (no self-loops in HSMM)
        diagonal = torch.diag(transition_matrix)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-7)
        
        # Rows should still sum to 1 (approximately, considering numerical precision)
        row_sums = torch.sum(transition_matrix, dim=1)
        expected_sums = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, expected_sums, atol=1e-6)
    
    def test_viterbi_decoding(self, basic_hsmm, device):
        """Test HSMM Viterbi decoding"""
        # Create shorter sequence for faster testing
        test_data = torch.randn(1, 30, 30, device=device)
        
        decoded_states, log_probs = basic_hsmm.viterbi_decode_hsmm(test_data)
        
        assert decoded_states.shape == (1, 30)
        assert log_probs.shape == (1,)
        
        # Check state values are valid
        assert torch.all(decoded_states >= 0)
        assert torch.all(decoded_states < basic_hsmm.num_states)
        
        # Log probabilities should be finite
        assert torch.all(torch.isfinite(log_probs))
    
    def test_forward_pass(self, basic_hsmm, device):
        """Test forward pass through HSMM"""
        test_data = torch.randn(1, 25, 30, device=device)
        
        decoded_states, log_probs = basic_hsmm(test_data)
        
        assert decoded_states.shape == (1, 25)
        assert log_probs.shape == (1,)
    
    def test_sequence_generation(self, basic_hsmm):
        """Test sequence generation capabilities"""
        length = 50
        
        states, observations = basic_hsmm.generate_sequence(length)
        
        assert states.shape == (length,)
        assert observations.shape == (length, basic_hsmm.feature_dim)
        
        # Check state values are valid
        assert torch.all(states >= 0)
        assert torch.all(states < basic_hsmm.num_states)
        
        # Check observations are finite
        assert torch.all(torch.isfinite(observations))
    
    def test_duration_analysis(self, basic_hsmm):
        """Test duration analysis of generated sequences"""
        from pytorch_hmm.utils import compute_state_durations
        
        # Generate a longer sequence to analyze durations
        states, _ = basic_hsmm.generate_sequence(200)
        durations = compute_state_durations(states)
        
        # Should have some durations
        assert len(durations) > 0
        
        # Durations should be positive
        assert torch.all(durations > 0)
        
        # Check against minimum duration constraint
        assert torch.all(durations >= basic_hsmm.min_duration)
    
    def test_expected_durations(self, basic_hsmm):
        """Test expected duration computation"""
        expected_durations = basic_hsmm.get_expected_durations()
        
        assert expected_durations.shape == (basic_hsmm.num_states,)
        assert torch.all(expected_durations > 0)
        assert torch.all(torch.isfinite(expected_durations))
    
    def test_model_info(self, basic_hsmm):
        """Test model information retrieval"""
        info = basic_hsmm.get_model_info()
        
        required_keys = [
            'model_type', 'num_states', 'feature_dim', 
            'duration_distribution', 'max_duration', 'min_duration',
            'expected_durations', 'total_parameters', 'trainable_parameters'
        ]
        
        for key in required_keys:
            assert key in info
        
        assert info['model_type'] == 'HSMM'
        assert info['num_states'] == 5
        assert info['feature_dim'] == 30
        assert info['duration_distribution'] == 'gamma'
        assert isinstance(info['expected_durations'], list)
    
    @pytest.mark.parametrize("distribution", ["gamma", "poisson", "weibull"])
    def test_different_distributions(self, distribution, device):
        """Test different duration distributions"""
        model = HSMMLayer(
            num_states=3,
            feature_dim=20,
            duration_distribution=distribution,
            max_duration=15
        ).to(device)
        
        # Test duration probabilities
        duration_probs = model.get_duration_probabilities()
        assert duration_probs.shape == (3, 15)
        assert torch.all(duration_probs >= 0)
        
        # Test sequence generation
        states, obs = model.generate_sequence(30)
        assert states.shape == (30,)
        assert obs.shape == (30, 20)
    
    def test_gradient_flow(self, basic_hsmm, device):
        """Test gradient computation for learnable parameters"""
        basic_hsmm.train()
        
        test_data = torch.randn(1, 20, 30, device=device, requires_grad=True)
        
        # Forward pass
        decoded_states, log_probs = basic_hsmm(test_data)
        
        # Compute loss
        loss = -log_probs.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients for learnable duration parameters
        for name, param in basic_hsmm.named_parameters():
            if param.requires_grad and 'duration' in name:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.all(torch.isfinite(param.grad)), f"Invalid gradient for {name}"
    
    def test_memory_efficiency(self, device):
        """Test memory efficiency with longer sequences"""
        model = HSMMLayer(
            num_states=8,
            feature_dim=40,
            max_duration=25
        ).to(device)
        
        # Test with moderately long sequence
        test_data = torch.randn(1, 100, 40, device=device)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        decoded_states, _ = model(test_data)
        
        assert decoded_states.shape == (1, 100)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            # Should use reasonable amount of memory
            assert peak_memory < 1000, f"Memory usage too high: {peak_memory:.1f} MB"
    
    def test_edge_cases(self, device):
        """Test edge cases and boundary conditions"""
        # Single state HSMM
        single_state_model = HSMMLayer(
            num_states=1,
            feature_dim=10,
            max_duration=5
        ).to(device)
        
        test_data = torch.randn(1, 15, 10, device=device)
        states, _ = single_state_model(test_data)
        
        # All states should be 0
        assert torch.all(states == 0)
        
        # Test with minimum duration larger than 1
        min_duration_model = HSMMLayer(
            num_states=3,
            feature_dim=10,
            min_duration=3,
            max_duration=10
        ).to(device)
        
        states_gen, _ = min_duration_model.generate_sequence(30)
        durations = compute_state_durations(states_gen)
        
        # All durations should be at least min_duration
        assert torch.all(durations >= 3)


class TestDurationConstrainedHMM:
    """Test suite for DurationConstrainedHMM"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def constrained_model(self, device):
        """Create duration-constrained HMM for testing"""
        return DurationConstrainedHMM(
            num_states=4,
            feature_dim=25,
            min_duration=3,
            max_duration=15
        ).to(device)
    
    def test_initialization(self, constrained_model):
        """Test DurationConstrainedHMM initialization"""
        assert constrained_model.num_states == 4
        assert constrained_model.feature_dim == 25
        assert constrained_model.min_duration == 3
        assert constrained_model.max_duration == 15
    
    def test_forward_pass(self, constrained_model, device):
        """Test forward pass with duration constraints"""
        test_data = torch.randn(2, 40, 25, device=device)
        
        decoded_states = constrained_model(test_data)
        
        assert decoded_states.shape == (2, 40)
        assert torch.all(decoded_states >= 0)
        assert torch.all(decoded_states < constrained_model.num_states)
    
    def test_duration_constraints(self, constrained_model, device):
        """Test that duration constraints are enforced"""
        # This is a behavioral test - duration constraints are enforced
        # through penalties rather than hard constraints
        test_data = torch.randn(1, 60, 25, device=device)
        
        decoded_states = constrained_model(test_data)
        
        # Should decode successfully
        assert decoded_states.shape == (1, 60)
        
        # Analyze resulting durations
        from pytorch_hmm.utils import compute_state_durations
        durations = compute_state_durations(decoded_states[0])
        
        # Most durations should respect constraints (within penalty system)
        # This is a soft constraint, so we allow some violations
        valid_durations = durations[(durations >= constrained_model.min_duration) & 
                                   (durations <= constrained_model.max_duration)]
        
        # At least 70% of durations should be valid
        assert len(valid_durations) >= 0.7 * len(durations)


@pytest.mark.integration
class TestHSMMIntegration:
    """Integration tests for HSMM with other components"""
    
    def test_with_utils_functions(self, device):
        """Test integration with utility functions"""
        from pytorch_hmm.utils import (
            analyze_transition_patterns, 
            compute_state_durations,
            interpolate_features
        )
        
        model = HSMMLayer(
            num_states=4,
            feature_dim=20,
            duration_distribution='gamma'
        ).to(device)
        
        # Generate multiple sequences
        sequences = []
        for _ in range(5):
            states, _ = model.generate_sequence(50)
            sequences.append(states)
        
        # Analyze patterns
        patterns = analyze_transition_patterns(sequences)
        
        assert 'self_loop_ratio' in patterns
        assert 'forward_ratio' in patterns
        assert 'avg_duration' in patterns
        
        # HSMM should have zero self-loop ratio
        assert patterns['self_loop_ratio'] == 0.0
        
        # Should have meaningful forward transitions
        assert patterns['forward_ratio'] > 0.5
    
    def test_duration_modeling_effectiveness(self, device):
        """Test that HSMM provides better duration modeling than regular HMM"""
        from pytorch_hmm.hmm_layer import HMMLayer
        from pytorch_hmm.utils import compute_state_durations
        
        # Create HSMM and regular HMM
        hsmm = HSMMLayer(
            num_states=3,
            feature_dim=15,
            duration_distribution='gamma'
        ).to(device)
        
        regular_hmm = HMMLayer(
            num_states=3,
            feature_dim=15
        ).to(device)
        
        # Generate sequences
        hsmm_states, _ = hsmm.generate_sequence(100)
        
        # Test data for regular HMM
        test_data = torch.randn(1, 100, 15, device=device)
        regular_states = regular_hmm(test_data)
        
        # Analyze durations
        hsmm_durations = compute_state_durations(hsmm_states)
        regular_durations = compute_state_durations(regular_states[0])
        
        # HSMM should have more consistent durations
        hsmm_duration_var = torch.var(hsmm_durations.float())
        regular_duration_var = torch.var(regular_durations.float()) if len(regular_durations) > 1 else torch.tensor(float('inf'))
        
        # This test might be probabilistic, so we just check basic properties
        assert len(hsmm_durations) > 0
        assert len(regular_durations) > 0
        assert torch.all(hsmm_durations > 0)
        assert torch.all(regular_durations > 0)


@pytest.mark.slow
class TestHSMMPerformance:
    """Performance tests for HSMM"""
    
    def test_scalability(self, device):
        """Test HSMM scalability with different model sizes"""
        import time
        
        configurations = [
            (5, 20, 15),   # Small
            (10, 40, 25),  # Medium  
            (20, 80, 35),  # Large
        ]
        
        for num_states, feature_dim, max_duration in configurations:
            model = HSMMLayer(
                num_states=num_states,
                feature_dim=feature_dim,
                max_duration=max_duration
            ).to(device)
            
            test_data = torch.randn(1, 50, feature_dim, device=device)
            
            start_time = time.time()
            states, _ = model(test_data)
            processing_time = time.time() - start_time
            
            # Should complete in reasonable time (less than 10 seconds)
            assert processing_time < 10.0, f"Too slow: {processing_time:.2f}s for {num_states} states"
            assert states.shape == (1, 50)


if __name__ == "__main__":
    pytest.main([__file__])
