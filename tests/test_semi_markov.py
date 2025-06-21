"""
Comprehensive tests for pytorch_hmm.semi_markov module
"""

import pytest
import torch
import numpy as np
from pytorch_hmm.semi_markov import (
    DurationModel,
    SemiMarkovHMM,
    AdaptiveDurationHSMM
)


class TestDurationModel:
    """Test DurationModel class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def duration_models(self, device):
        """Create duration models with different distributions"""
        models = {}
        num_states = 5
        max_duration = 20
        
        models['gamma'] = DurationModel(
            num_states=num_states,
            max_duration=max_duration,
            distribution_type='gamma'
        ).to(device)
        
        models['poisson'] = DurationModel(
            num_states=num_states,
            max_duration=max_duration,
            distribution_type='poisson'
        ).to(device)
        
        models['gaussian'] = DurationModel(
            num_states=num_states,
            max_duration=max_duration,
            distribution_type='gaussian'
        ).to(device)
        
        models['neural'] = DurationModel(
            num_states=num_states,
            max_duration=max_duration,
            distribution_type='neural',
            hidden_dim=64
        ).to(device)
        
        return models
    
    def test_duration_model_init(self, duration_models):
        """Test DurationModel initialization"""
        for dist_type, model in duration_models.items():
            assert model.num_states == 5
            assert model.max_duration == 20
            assert model.distribution_type == dist_type
            assert model.min_duration == 1
    
    def test_duration_distribution_computation(self, duration_models, device):
        """Test duration distribution computation"""
        batch_size = 3
        state_indices = torch.randint(0, 5, (batch_size,), device=device)
        
        for dist_type, model in duration_models.items():
            log_probs = model(state_indices)
            
            assert log_probs.shape == (batch_size, 20)  # max_duration
            assert torch.isfinite(log_probs).all()
            
            # Check if probabilities sum to approximately 1 (in log space)
            probs = torch.exp(log_probs)
            prob_sums = torch.sum(probs, dim=1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-2)
    
    def test_duration_probability_computation(self, duration_models, device):
        """Test specific duration probability computation"""
        batch_size = 3
        state_indices = torch.randint(0, 5, (batch_size,), device=device)
        durations = torch.randint(1, 11, (batch_size,), device=device)
        
        for dist_type, model in duration_models.items():
            log_probs = model(state_indices, durations)
            
            assert log_probs.shape == (batch_size,)
            assert torch.isfinite(log_probs).all()
            assert torch.all(log_probs <= 0)  # log probabilities should be <= 0
    
    def test_duration_sampling(self, duration_models, device):
        """Test duration sampling"""
        batch_size = 3
        state_indices = torch.randint(0, 5, (batch_size,), device=device)
        num_samples = 5
        
        for dist_type, model in duration_models.items():
            # Single sample
            samples_single = model.sample(state_indices, num_samples=1)
            assert samples_single.shape == (batch_size,)
            assert torch.all(samples_single >= model.min_duration)
            assert torch.all(samples_single <= model.max_duration)
            
            # Multiple samples
            samples_multi = model.sample(state_indices, num_samples=num_samples)
            assert samples_multi.shape == (batch_size, num_samples)
            assert torch.all(samples_multi >= model.min_duration)
            assert torch.all(samples_multi <= model.max_duration)
    
    def test_gamma_distribution_parameters(self, device):
        """Test gamma distribution parameter handling"""
        model = DurationModel(
            num_states=3,
            max_duration=10,
            distribution_type='gamma'
        ).to(device)
        
        # Check parameter shapes
        assert model.alpha_params.shape == (3,)
        assert model.beta_params.shape == (3,)
        
        # Check parameter positivity after softplus
        import torch.nn.functional as F
        alpha = F.softplus(model.alpha_params) + 1e-6
        beta = F.softplus(model.beta_params) + 1e-6
        
        assert torch.all(alpha > 0)
        assert torch.all(beta > 0)
    
    def test_poisson_distribution_parameters(self, device):
        """Test Poisson distribution parameter handling"""
        model = DurationModel(
            num_states=3,
            max_duration=10,
            distribution_type='poisson'
        ).to(device)
        
        # Check parameter shapes
        assert model.lambda_params.shape == (3,)
        
        # Check parameter positivity after softplus
        import torch.nn.functional as F
        lambda_param = F.softplus(model.lambda_params) + 1e-6
        assert torch.all(lambda_param > 0)
    
    def test_gaussian_distribution_parameters(self, device):
        """Test Gaussian distribution parameter handling"""
        model = DurationModel(
            num_states=3,
            max_duration=10,
            distribution_type='gaussian'
        ).to(device)
        
        # Check parameter shapes
        assert model.mean_params.shape == (3,)
        assert model.std_params.shape == (3,)
        
        # Check parameter constraints
        import torch.nn.functional as F
        mean = F.softplus(model.mean_params) + model.min_duration
        std = F.softplus(model.std_params) + 1e-6
        
        assert torch.all(mean >= model.min_duration)
        assert torch.all(std > 0)
    
    def test_neural_duration_model(self, device):
        """Test neural duration model"""
        model = DurationModel(
            num_states=5,
            max_duration=15,
            distribution_type='neural',
            hidden_dim=32
        ).to(device)
        
        # Check network structure
        assert hasattr(model, 'duration_net')
        assert isinstance(model.duration_net, torch.nn.Sequential)
        
        # Test forward pass
        state_indices = torch.randint(0, 5, (3,), device=device)
        log_probs = model(state_indices)
        
        assert log_probs.shape == (3, 15)
        assert torch.isfinite(log_probs).all()
        
        # Check if it's a valid log probability distribution
        probs = torch.exp(log_probs)
        prob_sums = torch.sum(probs, dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    
    def test_min_duration_constraint(self, device):
        """Test minimum duration constraint"""
        min_duration = 3
        model = DurationModel(
            num_states=3,
            max_duration=10,
            distribution_type='gamma',
            min_duration=min_duration
        ).to(device)
        
        state_indices = torch.tensor([0, 1, 2], device=device)
        durations = torch.tensor([1, 2, 5], device=device)  # First two below min_duration
        
        log_probs = model(state_indices, durations)
        
        # First two should be -inf due to min_duration constraint
        assert log_probs[0] == float('-inf')
        assert log_probs[1] == float('-inf')
        assert torch.isfinite(log_probs[2])


class TestSemiMarkovHMM:
    """Test SemiMarkovHMM class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def hsmm_models(self, device):
        """Create HSMM models with different configurations"""
        models = {}
        num_states = 4
        observation_dim = 8
        max_duration = 10
        
        models['gamma_gaussian'] = SemiMarkovHMM(
            num_states=num_states,
            observation_dim=observation_dim,
            max_duration=max_duration,
            duration_distribution='gamma',
            observation_model='gaussian'
        ).to(device)
        
        models['poisson_gaussian'] = SemiMarkovHMM(
            num_states=num_states,
            observation_dim=observation_dim,
            max_duration=max_duration,
            duration_distribution='poisson',
            observation_model='gaussian'
        ).to(device)
        
        models['neural_gaussian'] = SemiMarkovHMM(
            num_states=num_states,
            observation_dim=observation_dim,
            max_duration=max_duration,
            duration_distribution='neural',
            observation_model='gaussian'
        ).to(device)
        
        return models
    
    def test_hsmm_init(self, hsmm_models):
        """Test HSMM initialization"""
        for config, model in hsmm_models.items():
            assert model.num_states == 4
            assert model.observation_dim == 8
            assert model.max_duration == 10
            assert hasattr(model, 'duration_model')
            assert hasattr(model, 'transition_matrix')
            assert hasattr(model, 'observation_means')
            assert hasattr(model, 'observation_covs')
    
    def test_supervised_forward(self, hsmm_models, device):
        """Test supervised forward pass"""
        batch_size = 2
        seq_len = 15
        observation_dim = 8
        
        observations = torch.randn(batch_size, seq_len, observation_dim, device=device)
        state_sequence = torch.randint(0, 4, (batch_size, 3), device=device)  # 3 segments
        duration_sequence = torch.tensor([[5, 5, 5], [6, 4, 5]], device=device)
        
        for config, model in hsmm_models.items():
            result = model(observations, state_sequence, duration_sequence)
            
            assert 'log_likelihood' in result
            assert 'observation_log_probs' in result
            assert 'duration_log_probs' in result
            assert 'transition_log_probs' in result
            
            assert result['log_likelihood'].shape == (batch_size,)
            assert torch.isfinite(result['log_likelihood']).all()
    
    def test_unsupervised_forward(self, hsmm_models, device):
        """Test unsupervised forward pass (inference)"""
        batch_size = 1  # Unsupervised typically works with single sequences
        seq_len = 12
        observation_dim = 8
        
        observations = torch.randn(batch_size, seq_len, observation_dim, device=device)
        
        for config, model in hsmm_models.items():
            result = model(observations)
            
            assert 'log_likelihood' in result
            assert result['log_likelihood'].shape == (batch_size,)
            assert torch.isfinite(result['log_likelihood']).all()
    
    def test_viterbi_decode(self, hsmm_models, device):
        """Test Viterbi decoding"""
        seq_len = 10
        observation_dim = 8
        
        observations = torch.randn(seq_len, observation_dim, device=device)
        
        for config, model in hsmm_models.items():
            state_path, duration_path, log_prob = model.viterbi_decode(observations)
            
            assert state_path.dtype == torch.long
            assert duration_path.dtype == torch.long
            assert torch.isfinite(log_prob)
            
            # Check path validity
            assert len(state_path) == len(duration_path)
            assert torch.all(state_path >= 0)
            assert torch.all(state_path < model.num_states)
            assert torch.all(duration_path >= model.duration_model.min_duration)
            assert torch.all(duration_path <= model.max_duration)
            
            # Check if durations sum to sequence length
            total_duration = torch.sum(duration_path).item()
            assert total_duration <= seq_len  # May be less due to alignment
    
    def test_sampling(self, hsmm_models, device):
        """Test sequence sampling"""
        num_segments = 5
        max_length = 30
        
        for config, model in hsmm_models.items():
            observations, state_path, duration_path = model.sample(num_segments, max_length)
            
            assert observations.shape[1] == model.observation_dim
            assert len(state_path) == num_segments
            assert len(duration_path) == num_segments
            
            # Check path validity
            assert torch.all(state_path >= 0)
            assert torch.all(state_path < model.num_states)
            assert torch.all(duration_path >= model.duration_model.min_duration)
            assert torch.all(duration_path <= model.max_duration)
            
            # Check if total duration matches observation length
            total_duration = torch.sum(duration_path).item()
            assert observations.shape[0] == total_duration
    
    def test_observation_log_probs(self, device):
        """Test observation log probability computation"""
        model = SemiMarkovHMM(
            num_states=3,
            observation_dim=4,
            max_duration=8,
            duration_distribution='gamma'
        ).to(device)
        
        batch_size = 2
        seq_len = 10
        observations = torch.randn(batch_size, seq_len, 4, device=device)
        state_sequence = torch.tensor([[0, 1, 2], [1, 0, 2]], device=device)
        duration_sequence = torch.tensor([[3, 4, 3], [4, 3, 3]], device=device)
        
        obs_log_probs = model._compute_observation_logprobs(
            observations, state_sequence, duration_sequence
        )
        
        assert obs_log_probs.shape == (batch_size,)
        assert torch.isfinite(obs_log_probs).all()
    
    def test_transition_log_probs(self, device):
        """Test transition log probability computation"""
        model = SemiMarkovHMM(
            num_states=3,
            observation_dim=4,
            max_duration=8
        ).to(device)
        
        batch_size = 2
        state_sequence = torch.tensor([[0, 1, 2], [2, 0, 1]], device=device)
        
        trans_log_probs = model._compute_transition_logprobs(state_sequence)
        
        assert trans_log_probs.shape == (batch_size,)
        assert torch.isfinite(trans_log_probs).all()


class TestAdaptiveDurationHSMM:
    """Test AdaptiveDurationHSMM class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def adaptive_hsmm(self, device):
        return AdaptiveDurationHSMM(
            num_states=4,
            observation_dim=6,
            context_dim=8,
            max_duration=12
        ).to(device)
    
    def test_adaptive_hsmm_init(self, adaptive_hsmm):
        """Test AdaptiveDurationHSMM initialization"""
        assert adaptive_hsmm.num_states == 4
        assert adaptive_hsmm.observation_dim == 6
        assert adaptive_hsmm.context_dim == 8
        assert adaptive_hsmm.max_duration == 12
        assert hasattr(adaptive_hsmm, 'context_projection')
    
    def test_contextual_duration_probs(self, adaptive_hsmm, device):
        """Test contextual duration probability computation"""
        batch_size = 3
        state_indices = torch.randint(0, 4, (batch_size,), device=device)
        context = torch.randn(batch_size, 8, device=device)
        
        log_probs = adaptive_hsmm.compute_contextual_duration_probs(state_indices, context)
        
        assert log_probs.shape == (batch_size, 12)  # max_duration
        assert torch.isfinite(log_probs).all()
        
        # Check if probabilities sum to approximately 1
        probs = torch.exp(log_probs)
        prob_sums = torch.sum(probs, dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-2)


class TestSemiMarkovEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_invalid_distribution_type(self, device):
        """Test invalid distribution type"""
        with pytest.raises(ValueError):
            DurationModel(
                num_states=3,
                max_duration=10,
                distribution_type='invalid'
            )
    
    def test_zero_duration(self, device):
        """Test handling of zero duration"""
        model = DurationModel(
            num_states=3,
            max_duration=10,
            distribution_type='gamma',
            min_duration=1
        ).to(device)
        
        state_indices = torch.tensor([0], device=device)
        durations = torch.tensor([0], device=device)  # Below min_duration
        
        log_probs = model(state_indices, durations)
        assert log_probs[0] == float('-inf')
    
    def test_very_long_duration(self, device):
        """Test handling of very long duration"""
        model = DurationModel(
            num_states=3,
            max_duration=10,
            distribution_type='gamma'
        ).to(device)
        
        state_indices = torch.tensor([0], device=device)
        durations = torch.tensor([100], device=device)  # Much longer than max
        
        log_probs = model(state_indices, durations)
        assert torch.isfinite(log_probs[0])  # Should still compute, just very low probability
    
    def test_single_state_hsmm(self, device):
        """Test HSMM with single state"""
        model = SemiMarkovHMM(
            num_states=1,
            observation_dim=4,
            max_duration=5
        ).to(device)
        
        observations = torch.randn(1, 8, 4, device=device)
        result = model(observations)
        
        assert torch.isfinite(result['log_likelihood'])
    
    def test_very_short_sequence(self, device):
        """Test HSMM with very short sequence"""
        model = SemiMarkovHMM(
            num_states=3,
            observation_dim=4,
            max_duration=10
        ).to(device)
        
        observations = torch.randn(1, 2, 4, device=device)  # Very short
        result = model(observations)
        
        assert torch.isfinite(result['log_likelihood'])
    
    def test_mismatched_sequence_lengths(self, device):
        """Test handling of mismatched sequence lengths"""
        model = SemiMarkovHMM(
            num_states=3,
            observation_dim=4,
            max_duration=10
        ).to(device)
        
        observations = torch.randn(1, 10, 4, device=device)
        state_sequence = torch.tensor([[0, 1, 2]], device=device)
        duration_sequence = torch.tensor([[3, 3, 3]], device=device)  # Sum = 9, obs = 10
        
        # Should handle gracefully
        result = model(observations, state_sequence, duration_sequence)
        assert torch.isfinite(result['log_likelihood']) 