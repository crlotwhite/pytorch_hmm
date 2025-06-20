"""
Tests for MixtureGaussianHMMLayer

Testing comprehensive functionality of the Mixture Gaussian HMM implementation
including different covariance types, numerical stability, and performance.
"""

import pytest
import torch
import numpy as np
from pytorch_hmm.mixture_gaussian import MixtureGaussianHMMLayer


class TestMixtureGaussianHMM:
    """Test suite for MixtureGaussianHMMLayer"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_data(self, device):
        """Generate sample data for testing"""
        batch_size, seq_len, feature_dim = 2, 50, 40
        return torch.randn(batch_size, seq_len, feature_dim, device=device)
    
    @pytest.fixture
    def basic_model(self, device):
        """Create basic model for testing"""
        return MixtureGaussianHMMLayer(
            num_states=5, 
            feature_dim=40, 
            num_components=2,
            covariance_type='diag'
        ).to(device)
    
    def test_model_initialization(self, device):
        """Test model initialization with different parameters"""
        # Test diagonal covariance
        model = MixtureGaussianHMMLayer(
            num_states=5, 
            feature_dim=40, 
            num_components=3,
            covariance_type='diag'
        ).to(device)
        
        assert model.num_states == 5
        assert model.feature_dim == 40
        assert model.num_components == 3
        assert model.covariance_type == 'diag'
        
        # Check parameter shapes
        assert model.transition_logits.shape == (5, 5)
        assert model.mixture_weights_logits.shape == (5, 3)
        assert model.means.shape == (5, 3, 40)
        assert model.log_vars.shape == (5, 3, 40)
    
    def test_covariance_types(self, device):
        """Test different covariance matrix types"""
        feature_dim = 20  # Smaller for full covariance test
        
        # Diagonal covariance
        model_diag = MixtureGaussianHMMLayer(
            num_states=3, feature_dim=feature_dim, num_components=2,
            covariance_type='diag'
        ).to(device)
        
        # Tied covariance
        model_tied = MixtureGaussianHMMLayer(
            num_states=3, feature_dim=feature_dim, num_components=2,
            covariance_type='tied'
        ).to(device)
        
        # Spherical covariance
        model_spherical = MixtureGaussianHMMLayer(
            num_states=3, feature_dim=feature_dim, num_components=2,
            covariance_type='spherical'
        ).to(device)
        
        test_data = torch.randn(1, 30, feature_dim, device=device)
        
        # All should work without errors
        states_diag, _ = model_diag(test_data)
        states_tied, _ = model_tied(test_data)
        states_spherical, _ = model_spherical(test_data)
        
        assert states_diag.shape == (1, 30)
        assert states_tied.shape == (1, 30)
        assert states_spherical.shape == (1, 30)
    
    def test_forward_pass(self, basic_model, sample_data):
        """Test forward pass functionality"""
        # Test without log probabilities
        decoded_states, log_probs = basic_model(sample_data, return_log_probs=False)
        
        batch_size, seq_len = sample_data.shape[:2]
        assert decoded_states.shape == (batch_size, seq_len)
        assert log_probs is None
        
        # Test with log probabilities
        decoded_states, log_probs = basic_model(sample_data, return_log_probs=True)
        
        assert decoded_states.shape == (batch_size, seq_len)
        assert log_probs.shape == (batch_size,)
        
        # Check state values are valid
        assert torch.all(decoded_states >= 0)
        assert torch.all(decoded_states < basic_model.num_states)
    
    def test_observation_log_probs(self, basic_model, sample_data):
        """Test observation log probability computation"""
        log_probs = basic_model.get_observation_log_probs(sample_data)
        
        batch_size, seq_len = sample_data.shape[:2]
        assert log_probs.shape == (batch_size, seq_len, basic_model.num_states)
        
        # Log probabilities should be negative (or zero)
        assert torch.all(log_probs <= 0)
        
        # Should be finite
        assert torch.all(torch.isfinite(log_probs))
    
    def test_transition_matrix(self, basic_model):
        """Test transition matrix computation"""
        # Test learnable transitions
        transition_matrix = basic_model.get_transition_matrix()
        
        assert transition_matrix.shape == (basic_model.num_states, basic_model.num_states)
        
        # Should be valid probability matrix
        assert torch.all(transition_matrix >= 0)
        assert torch.all(transition_matrix <= 1)
        
        # Rows should sum to 1
        row_sums = torch.sum(transition_matrix, dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    
    def test_numerical_stability(self, device):
        """Test numerical stability with extreme values"""
        model = MixtureGaussianHMMLayer(
            num_states=3, feature_dim=10, num_components=2
        ).to(device)
        
        # Test with very small values
        small_data = torch.full((1, 20, 10), 1e-10, device=device)
        decoded_states, _ = model(small_data)
        assert torch.all(torch.isfinite(decoded_states))
        
        # Test with very large values
        large_data = torch.full((1, 20, 10), 1e10, device=device)
        decoded_states, _ = model(large_data)
        assert torch.all(torch.isfinite(decoded_states))
        
        # Test with mixed extreme values
        mixed_data = torch.randn(1, 20, 10, device=device) * 1e5
        decoded_states, _ = model(mixed_data)
        assert torch.all(torch.isfinite(decoded_states))
    
    def test_gradient_flow(self, basic_model, sample_data):
        """Test gradient computation"""
        basic_model.train()
        
        # Forward pass
        decoded_states, log_probs = basic_model(sample_data, return_log_probs=True)
        
        # Compute loss
        loss = -log_probs.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are finite
        for name, param in basic_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.all(torch.isfinite(param.grad)), f"Invalid gradient for {name}"
    
    def test_different_sequence_lengths(self, device):
        """Test with different sequence lengths"""
        model = MixtureGaussianHMMLayer(
            num_states=4, feature_dim=30, num_components=2
        ).to(device)
        
        # Short sequence
        short_data = torch.randn(1, 5, 30, device=device)
        states_short, _ = model(short_data)
        assert states_short.shape == (1, 5)
        
        # Medium sequence
        medium_data = torch.randn(1, 100, 30, device=device)
        states_medium, _ = model(medium_data)
        assert states_medium.shape == (1, 100)
        
        # Long sequence (test chunking)
        long_data = torch.randn(1, 1500, 30, device=device)
        states_long, _ = model(long_data)
        assert states_long.shape == (1, 1500)
    
    def test_batch_processing(self, device):
        """Test batch processing capabilities"""
        model = MixtureGaussianHMMLayer(
            num_states=5, feature_dim=25, num_components=2
        ).to(device)
        
        # Different batch sizes
        for batch_size in [1, 4, 8, 16]:
            data = torch.randn(batch_size, 50, 25, device=device)
            states, log_probs = model(data, return_log_probs=True)
            
            assert states.shape == (batch_size, 50)
            assert log_probs.shape == (batch_size,)
    
    def test_model_info(self, basic_model):
        """Test model information retrieval"""
        info = basic_model.get_model_info()
        
        assert 'num_states' in info
        assert 'feature_dim' in info
        assert 'num_components' in info
        assert 'covariance_type' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        
        assert info['num_states'] == 5
        assert info['feature_dim'] == 40
        assert info['num_components'] == 2
        assert info['covariance_type'] == 'diag'
        assert isinstance(info['total_parameters'], int)
        assert isinstance(info['trainable_parameters'], int)
    
    @pytest.mark.slow
    def test_memory_efficiency(self, device):
        """Test memory efficiency with large models"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory test")
        
        # Large model
        model = MixtureGaussianHMMLayer(
            num_states=50, 
            feature_dim=80, 
            num_components=5,
            max_sequence_length=5000
        ).to(device)
        
        # Monitor memory usage
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # Process large batch
        large_data = torch.randn(8, 2000, 80, device=device)
        states, _ = model(large_data)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - initial_memory) / (1024 ** 2)  # MB
        
        # Should process successfully
        assert states.shape == (8, 2000)
        
        # Memory usage should be reasonable (less than 2GB for this test)
        assert memory_used < 2000, f"Memory usage too high: {memory_used:.1f} MB"
    
    def test_edge_cases(self, device):
        """Test edge cases and error conditions"""
        # Single state
        single_state_model = MixtureGaussianHMMLayer(
            num_states=1, feature_dim=10, num_components=1
        ).to(device)
        
        data = torch.randn(1, 20, 10, device=device)
        states, _ = single_state_model(data)
        assert torch.all(states == 0)  # All states should be 0
        
        # Single component
        single_component_model = MixtureGaussianHMMLayer(
            num_states=3, feature_dim=10, num_components=1
        ).to(device)
        
        states, _ = single_component_model(data)
        assert states.shape == (1, 20)
    
    def test_state_consistency(self, basic_model, device):
        """Test that model produces consistent results"""
        # Set model to eval mode
        basic_model.eval()
        
        # Generate test data
        torch.manual_seed(42)
        test_data = torch.randn(1, 30, 40, device=device)
        
        # Run multiple times with same data
        with torch.no_grad():
            states1, _ = basic_model(test_data)
            states2, _ = basic_model(test_data)
        
        # Results should be identical in eval mode
        assert torch.all(states1 == states2)
    
    @pytest.mark.parametrize("num_components", [1, 2, 4, 8])
    def test_different_component_counts(self, num_components, device):
        """Test with different numbers of mixture components"""
        model = MixtureGaussianHMMLayer(
            num_states=4, 
            feature_dim=20, 
            num_components=num_components
        ).to(device)
        
        data = torch.randn(2, 25, 20, device=device)
        states, log_probs = model(data, return_log_probs=True)
        
        assert states.shape == (2, 25)
        assert log_probs.shape == (2,)
        assert torch.all(torch.isfinite(log_probs))


@pytest.mark.integration
class TestMixtureGaussianIntegration:
    """Integration tests for MixtureGaussianHMM with other components"""
    
    def test_with_utils_functions(self):
        """Test integration with utility functions"""
        from pytorch_hmm.utils import create_left_to_right_matrix, validate_transition_matrix
        
        # Create model with custom transition matrix
        model = MixtureGaussianHMMLayer(
            num_states=5, feature_dim=30, num_components=2,
            learnable_transitions=False
        )
        
        # Set custom transition matrix
        custom_transitions = create_left_to_right_matrix(5, self_loop_prob=0.8)
        model.transition_matrix = custom_transitions
        
        # Validate transition matrix
        validation_results = validate_transition_matrix(custom_transitions)
        assert validation_results['row_sums_valid']
        assert validation_results['non_negative']
        assert validation_results['finite']
        
        # Test model works with custom transitions
        test_data = torch.randn(1, 40, 30)
        states, _ = model(test_data)
        assert states.shape == (1, 40)


if __name__ == "__main__":
    pytest.main([__file__])
