"""
Tests for pytorch_hmm.optimizations module
"""

import pytest
import torch
import torch.nn as nn
from pytorch_hmm.optimizations import (
    MemoryEfficientHMM,
    BatchOptimizer,
    MixedPrecisionTrainer,
    AdaptiveChunkProcessor,
    optimize_for_inference,
    benchmark_memory_usage,
    get_optimization_recommendations
)


class TestMemoryEfficientHMM:
    """Test MemoryEfficientHMM class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        return MemoryEfficientHMM(num_states=5, observation_dim=8).to(device)
    
    @pytest.fixture
    def sample_data(self, device):
        batch_size, seq_len, obs_dim = 2, 10, 8
        return torch.randn(batch_size, seq_len, obs_dim, device=device)
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model.num_states == 5
        assert model.observation_dim == 8
        assert model.use_checkpointing == True
        assert model.chunk_size == 32
        
        # Check parameter shapes
        assert model.transition_logits.shape == (5, 5)
        assert model.emission_means.shape == (5, 8)
        assert model.emission_logvars.shape == (5, 8)
        assert model.initial_logits.shape == (5,)
    
    def test_forward_pass(self, model, sample_data):
        """Test forward pass"""
        model.eval()
        
        with torch.no_grad():
            log_probs, total_log_prob = model(sample_data)
        
        batch_size, seq_len = sample_data.shape[:2]
        
        # Check output shapes
        assert log_probs.shape == (batch_size, seq_len, model.num_states)
        assert total_log_prob.shape == (batch_size,)
        
        # Check output validity
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(total_log_prob).all()
    
    def test_mixed_precision_forward(self, model, sample_data):
        """Test forward pass with mixed precision"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        model.eval()
        
        with torch.no_grad():
            log_probs, total_log_prob = model(sample_data, use_mixed_precision=True)
        
        batch_size, seq_len = sample_data.shape[:2]
        
        # Check output shapes
        assert log_probs.shape == (batch_size, seq_len, model.num_states)
        assert total_log_prob.shape == (batch_size,)
    
    def test_gradient_checkpointing(self, model, sample_data):
        """Test gradient checkpointing during training"""
        model.train()
        model.use_checkpointing = True
        
        log_probs, total_log_prob = model(sample_data)
        loss = -total_log_prob.mean()
        
        # Should not raise error during backward pass
        loss.backward()
        
        # Check gradients exist
        assert model.transition_logits.grad is not None
        assert model.emission_means.grad is not None
    
    def test_emission_computation(self, model, sample_data):
        """Test emission probability computation"""
        model.eval()
        
        with torch.no_grad():
            emission_log_probs = model._compute_emission_log_probs(sample_data)
        
        batch_size, seq_len = sample_data.shape[:2]
        
        # Check shape and validity
        assert emission_log_probs.shape == (batch_size, seq_len, model.num_states)
        assert torch.isfinite(emission_log_probs).all()
        
        # Log probabilities should be negative
        assert (emission_log_probs <= 0).all()


class TestBatchOptimizer:
    """Test BatchOptimizer class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def simple_model(self, device):
        """Simple model for testing"""
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        ).to(device)
        return model
    
    @pytest.fixture
    def optimizer(self, simple_model, device):
        return BatchOptimizer(simple_model, device)
    
    def test_optimizer_initialization(self, optimizer, device):
        """Test optimizer initialization"""
        assert optimizer.device == device
        assert optimizer.optimal_batch_size is None
        assert optimizer.memory_limit_mb > 0
    
    def test_memory_limit_calculation(self, optimizer):
        """Test memory limit calculation"""
        memory_limit = optimizer._get_memory_limit()
        assert memory_limit > 0
        assert isinstance(memory_limit, float)
    
    def test_optimal_batch_size_finding(self, optimizer):
        """Test optimal batch size finding"""
        seq_len, obs_dim = 20, 8
        
        optimal_size = optimizer.find_optimal_batch_size(
            seq_len=seq_len,
            obs_dim=obs_dim,
            max_batch_size=16
        )
        
        assert optimal_size >= 1
        assert optimal_size <= 16
        assert isinstance(optimal_size, int)
        
        # Second call should return cached result
        cached_size = optimizer.find_optimal_batch_size(seq_len, obs_dim)
        assert cached_size == optimal_size


class TestMixedPrecisionTrainer:
    """Test MixedPrecisionTrainer class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        return MemoryEfficientHMM(num_states=3, observation_dim=4).to(device)
    
    @pytest.fixture
    def trainer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return MixedPrecisionTrainer(model, optimizer)
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization"""
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scaler is not None
        assert isinstance(trainer.enabled, bool)
    
    def test_training_step(self, trainer, device):
        """Test training step"""
        batch_size, seq_len, obs_dim = 2, 8, 4
        observations = torch.randn(batch_size, seq_len, obs_dim, device=device)
        targets = torch.randn(batch_size, device=device)
        
        metrics = trainer.train_step(observations, targets)
        
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)
        assert torch.isfinite(torch.tensor(metrics['loss']))


class TestAdaptiveChunkProcessor:
    """Test AdaptiveChunkProcessor class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Simple model that returns input shape for testing"""
        class SimpleModel(nn.Module):
            def forward(self, x):
                return x.sum(dim=-1, keepdim=True)  # [B, T, D] -> [B, T, 1]
        
        return SimpleModel().to(device)
    
    @pytest.fixture
    def processor(self, model):
        return AdaptiveChunkProcessor(model, chunk_size=16)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.chunk_size == 16
        assert processor.overlap == 4  # 25% of chunk_size
    
    def test_short_sequence_processing(self, processor, device):
        """Test processing of short sequences"""
        batch_size, seq_len, obs_dim = 2, 10, 8  # seq_len < chunk_size
        observations = torch.randn(batch_size, seq_len, obs_dim, device=device)
        
        result = processor.process_sequence(observations)
        
        # Should process normally without chunking
        assert result.shape == (batch_size, seq_len, 1)
    
    def test_long_sequence_processing(self, processor, device):
        """Test processing of long sequences"""
        batch_size, seq_len, obs_dim = 2, 50, 8  # seq_len > chunk_size
        observations = torch.randn(batch_size, seq_len, obs_dim, device=device)
        
        result = processor.process_sequence(observations)
        
        # Should process with chunking
        assert result.shape[0] == batch_size
        assert result.shape[2] == 1
        # Length might be slightly different due to overlap handling
        assert abs(result.shape[1] - seq_len) <= processor.overlap


class TestOptimizationUtilities:
    """Test optimization utility functions"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        return MemoryEfficientHMM(num_states=3, observation_dim=4).to(device)
    
    @pytest.fixture
    def sample_data(self, device):
        return torch.randn(2, 10, 4, device=device)
    
    def test_optimize_for_inference(self, model):
        """Test inference optimization"""
        optimized_model = optimize_for_inference(model)
        
        # Should return a model (might be JIT compiled or original)
        assert optimized_model is not None
        assert hasattr(optimized_model, 'forward')
    
    def test_memory_benchmarking(self, model, sample_data, device):
        """Test memory usage benchmarking"""
        results = benchmark_memory_usage(model, sample_data, device)
        
        assert isinstance(results, dict)
        
        if device.type == 'cuda':
            assert 'original_memory_mb' in results
            assert isinstance(results['original_memory_mb'], float)
            assert results['original_memory_mb'] > 0
    
    def test_optimization_recommendations(self, model, device):
        """Test optimization recommendations"""
        recommendations = get_optimization_recommendations(
            model=model,
            typical_batch_size=8,
            typical_seq_len=100,
            device=device
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0
    
    def test_recommendations_for_different_scenarios(self, model, device):
        """Test recommendations for different usage scenarios"""
        # Large model scenario
        large_model_recs = get_optimization_recommendations(
            model=model,
            typical_batch_size=1,
            typical_seq_len=1000,
            device=device
        )
        
        # Small batch scenario
        small_batch_recs = get_optimization_recommendations(
            model=model,
            typical_batch_size=1,
            typical_seq_len=50,
            device=device
        )
        
        # Large batch scenario
        large_batch_recs = get_optimization_recommendations(
            model=model,
            typical_batch_size=64,
            typical_seq_len=50,
            device=device
        )
        
        assert len(large_model_recs) > 0
        assert len(small_batch_recs) > 0
        assert len(large_batch_recs) > 0
        
        # Different scenarios should give different recommendations
        all_recs = [large_model_recs, small_batch_recs, large_batch_recs]
        assert not all(recs == all_recs[0] for recs in all_recs)


class TestIntegration:
    """Integration tests for optimization components"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_end_to_end_optimization_pipeline(self, device):
        """Test complete optimization pipeline"""
        # Create model
        model = MemoryEfficientHMM(num_states=4, observation_dim=6).to(device)
        
        # Optimize for inference
        optimized_model = optimize_for_inference(model)
        
        # Create batch optimizer
        batch_optimizer = BatchOptimizer(optimized_model, device)
        
        # Find optimal batch size
        optimal_batch = batch_optimizer.find_optimal_batch_size(
            seq_len=30, obs_dim=6, max_batch_size=16
        )
        
        # Create chunk processor
        chunk_processor = AdaptiveChunkProcessor(optimized_model, chunk_size=20)
        
        # Test with sample data
        sample_data = torch.randn(optimal_batch, 50, 6, device=device)
        
        # Process with chunking
        result = chunk_processor.process_sequence(sample_data)
        
        # Verify result
        assert result.shape[0] == optimal_batch
        assert result.shape[2] == model.num_states
        assert torch.isfinite(result).all()
    
    def test_mixed_precision_with_memory_efficient_model(self, device):
        """Test mixed precision training with memory efficient model"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        model = MemoryEfficientHMM(num_states=3, observation_dim=5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = MixedPrecisionTrainer(model, optimizer)
        
        # Training data
        observations = torch.randn(4, 15, 5, device=device)
        targets = torch.randn(4, device=device)
        
        # Training step
        metrics = trainer.train_step(observations, targets)
        
        assert 'loss' in metrics
        assert torch.isfinite(torch.tensor(metrics['loss'])) 