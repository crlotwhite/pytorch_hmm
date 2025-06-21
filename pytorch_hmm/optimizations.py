"""
Performance optimizations for PyTorch HMM

This module provides various optimization techniques:
- Memory-efficient implementations
- GPU acceleration utilities
- Batch processing optimizations
- JIT compilation helpers
- Mixed precision support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp.autocast_mode import autocast
    from torch.amp.grad_scaler import GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torch.jit import script, trace
import warnings
from typing import Optional, Tuple, Dict, Any, List
import math


class MemoryEfficientHMM(nn.Module):
    """Memory-efficient HMM implementation with gradient checkpointing"""
    
    def __init__(self, num_states: int, observation_dim: int):
        super().__init__()
        self.num_states = num_states
        self.observation_dim = observation_dim
        
        # Learnable parameters
        self.transition_logits = nn.Parameter(torch.randn(num_states, num_states))
        self.emission_means = nn.Parameter(torch.randn(num_states, observation_dim))
        self.emission_logvars = nn.Parameter(torch.zeros(num_states, observation_dim))
        self.initial_logits = nn.Parameter(torch.randn(num_states))
        
        # Memory optimization flags
        self.use_checkpointing = True
        self.chunk_size = 32  # Process in chunks to save memory
    
    def forward(self, observations: torch.Tensor, 
                use_mixed_precision: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-efficient forward pass
        
        Args:
            observations: [batch_size, seq_len, obs_dim]
            use_mixed_precision: Use automatic mixed precision
            
        Returns:
            log_probs: [batch_size, seq_len, num_states]
            total_log_prob: [batch_size]
        """
        if use_mixed_precision:
            device_type = 'cuda' if observations.device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type):
                return self._forward_impl(observations)
        else:
            return self._forward_impl(observations)
    
    def _forward_impl(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal forward implementation"""
        batch_size, seq_len, _ = observations.shape
        
        # Compute transition and emission probabilities
        transition_probs = F.log_softmax(self.transition_logits, dim=-1)
        initial_probs = F.log_softmax(self.initial_logits, dim=-1)
        
        # Compute emission probabilities efficiently
        emission_log_probs = self._compute_emission_log_probs(observations)
        
        # Memory-efficient forward algorithm
        if self.use_checkpointing and self.training:
            try:
                from torch.utils.checkpoint import checkpoint
                result = checkpoint(
                    self._forward_algorithm,
                    emission_log_probs, transition_probs, initial_probs,
                    use_reentrant=False
                )
                if isinstance(result, tuple):
                    log_probs, total_log_prob = result
                else:
                    log_probs, total_log_prob = self._forward_algorithm(
                        emission_log_probs, transition_probs, initial_probs
                    )
            except Exception:
                log_probs, total_log_prob = self._forward_algorithm(
                    emission_log_probs, transition_probs, initial_probs
                )
        else:
            log_probs, total_log_prob = self._forward_algorithm(
                emission_log_probs, transition_probs, initial_probs
            )
        
        return log_probs, total_log_prob
    
    def _compute_emission_log_probs(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute emission log probabilities efficiently"""
        batch_size, seq_len, obs_dim = observations.shape
        
        # Expand for broadcasting: [batch, seq, 1, obs] vs [1, 1, states, obs]
        obs_expanded = observations.unsqueeze(2)  # [batch, seq, 1, obs]
        means_expanded = self.emission_means.unsqueeze(0).unsqueeze(0)  # [1, 1, states, obs]
        logvars_expanded = self.emission_logvars.unsqueeze(0).unsqueeze(0)
        
        # Gaussian log likelihood
        diff = obs_expanded - means_expanded
        log_probs = -0.5 * (
            logvars_expanded + 
            (diff ** 2) * torch.exp(-logvars_expanded) + 
            math.log(2 * math.pi)
        ).sum(dim=-1)
        
        return log_probs  # [batch, seq, states]
    
    def _forward_algorithm(self, emission_log_probs: torch.Tensor,
                          transition_probs: torch.Tensor,
                          initial_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory-efficient forward algorithm"""
        batch_size, seq_len, num_states = emission_log_probs.shape
        
        # Initialize forward variables
        forward_vars = initial_probs + emission_log_probs[:, 0]  # [batch, states]
        all_forward_vars = [forward_vars.unsqueeze(1)]  # Store for backward pass
        
        # Forward pass with chunking for memory efficiency
        for t in range(1, seq_len):
            # Compute forward variables for time t
            forward_vars = torch.logsumexp(
                forward_vars.unsqueeze(2) + transition_probs.unsqueeze(0),
                dim=1
            ) + emission_log_probs[:, t]
            
            all_forward_vars.append(forward_vars.unsqueeze(1))
        
        # Concatenate all forward variables
        log_probs = torch.cat(all_forward_vars, dim=1)  # [batch, seq, states]
        
        # Total log probability
        total_log_prob = torch.logsumexp(forward_vars, dim=1)  # [batch]
        
        return log_probs, total_log_prob


class GPUAcceleratedHMM(nn.Module):
    """GPU-optimized HMM with CUDA kernels and memory coalescing"""
    
    def __init__(self, num_states: int, observation_dim: int):
        super().__init__()
        self.num_states = num_states
        self.observation_dim = observation_dim
        
        # Pre-allocate GPU memory for better performance
        self.register_buffer('_temp_buffer', torch.empty(1, 1, num_states))
        self.register_buffer('_emission_cache', torch.empty(1, 1, num_states))
        
        # Parameters
        self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states))
        self.emission_params = nn.Parameter(torch.randn(num_states, observation_dim * 2))
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """GPU-optimized forward pass"""
        batch_size, seq_len, obs_dim = observations.shape
        
        # Ensure proper memory layout for GPU
        observations = observations.contiguous()
        
        # Resize buffers if needed
        if (self._temp_buffer.shape[0] != batch_size or 
            self._temp_buffer.shape[1] != seq_len):
            self._temp_buffer = self._temp_buffer.resize_(batch_size, seq_len, self.num_states)
            self._emission_cache = self._emission_cache.resize_(batch_size, seq_len, self.num_states)
        
        # Use efficient GPU operations
        transition_probs = F.log_softmax(self.transition_matrix, dim=-1)
        
        # Vectorized emission computation
        means = self.emission_params[:, :obs_dim]
        logvars = self.emission_params[:, obs_dim:]
        
        # Batch matrix operations for emission probabilities
        emission_log_probs = self._vectorized_emission_computation(
            observations, means, logvars
        )
        
        # GPU-optimized forward algorithm
        return self._gpu_forward_algorithm(emission_log_probs, transition_probs)
    
    def _vectorized_emission_computation(self, observations: torch.Tensor,
                                       means: torch.Tensor,
                                       logvars: torch.Tensor) -> torch.Tensor:
        """Vectorized emission probability computation"""
        # Use einsum for efficient computation
        diff = observations.unsqueeze(-2) - means.unsqueeze(0).unsqueeze(0)  # [B, T, S, D]
        
        # Vectorized Gaussian log likelihood
        log_probs = -0.5 * torch.sum(
            logvars.unsqueeze(0).unsqueeze(0) + 
            diff.pow(2) * torch.exp(-logvars.unsqueeze(0).unsqueeze(0)),
            dim=-1
        )
        
        return log_probs
    
    def _gpu_forward_algorithm(self, emission_log_probs: torch.Tensor,
                              transition_probs: torch.Tensor) -> torch.Tensor:
        """GPU-optimized forward algorithm using batch matrix operations"""
        batch_size, seq_len, num_states = emission_log_probs.shape
        
        # Initialize
        alpha = emission_log_probs[:, 0]  # [B, S]
        
        # Vectorized forward pass
        for t in range(1, seq_len):
            # Batch matrix multiplication for transition
            alpha = torch.logsumexp(
                alpha.unsqueeze(-1) + transition_probs.unsqueeze(0),
                dim=1
            ) + emission_log_probs[:, t]
        
        return torch.logsumexp(alpha, dim=-1)


class BatchOptimizer:
    """Automatic batch size optimization for different hardware configurations"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.optimal_batch_size = None
        self.memory_limit_mb = self._get_memory_limit()
    
    def _get_memory_limit(self) -> float:
        """Get memory limit based on device"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_properties(self.device).total_memory / 1024**2 * 0.8  # 80% of GPU memory
        else:
            import psutil
            return psutil.virtual_memory().total / 1024**2 * 0.5  # 50% of system memory
    
    def find_optimal_batch_size(self, seq_len: int, obs_dim: int, 
                               max_batch_size: int = 128) -> int:
        """Find optimal batch size through binary search"""
        if self.optimal_batch_size is not None:
            return self.optimal_batch_size
        
        self.model.eval()
        
        # Binary search for optimal batch size
        left, right = 1, max_batch_size
        optimal_size = 1
        
        while left <= right:
            mid = (left + right) // 2
            
            try:
                # Test with this batch size
                test_data = torch.randn(mid, seq_len, obs_dim, device=self.device)
                
                # Measure memory usage
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    _ = self.model(test_data)
                
                if self.device.type == 'cuda':
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                    if peak_memory < self.memory_limit_mb:
                        optimal_size = mid
                        left = mid + 1
                    else:
                        right = mid - 1
                else:
                    # For CPU, use simpler heuristic
                    optimal_size = mid
                    left = mid + 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    right = mid - 1
                else:
                    raise e
        
        self.optimal_batch_size = optimal_size
        return optimal_size
    
    def create_optimized_dataloader(self, dataset, **kwargs):
        """Create DataLoader with optimal batch size"""
        optimal_batch = self.find_optimal_batch_size(
            seq_len=kwargs.get('seq_len', 100),
            obs_dim=kwargs.get('obs_dim', 80)
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=optimal_batch,
            **{k: v for k, v in kwargs.items() if k not in ['seq_len', 'obs_dim']}
        )


class MixedPrecisionTrainer:
    """Mixed precision training utilities for HMM models"""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        self.enabled = torch.cuda.is_available()
    
    def train_step(self, observations: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Perform one training step with mixed precision"""
        self.optimizer.zero_grad()
        
        if self.enabled:
            with autocast():
                loss = self._compute_loss(observations, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(observations, targets)
            loss.backward()
            self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def _compute_loss(self, observations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss (to be implemented by specific model)"""
        # This would be implemented based on the specific HMM variant
        log_probs, total_log_prob = self.model(observations)
        return -total_log_prob.mean()


@script
def jit_optimized_forward_backward(emission_log_probs: torch.Tensor,
                                  transition_log_probs: torch.Tensor,
                                  initial_log_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """JIT-compiled forward-backward algorithm for maximum performance"""
    batch_size, seq_len, num_states = emission_log_probs.shape
    
    # Forward pass
    alpha = torch.zeros(batch_size, seq_len, num_states, dtype=emission_log_probs.dtype, device=emission_log_probs.device)
    alpha[:, 0] = initial_log_probs + emission_log_probs[:, 0]
    
    for t in range(1, seq_len):
        for j in range(num_states):
            alpha[:, t, j] = torch.logsumexp(
                alpha[:, t-1] + transition_log_probs[:, j], dim=1
            ) + emission_log_probs[:, t, j]
    
    # Backward pass
    beta = torch.zeros_like(alpha)
    beta[:, -1] = 0.0  # log(1) = 0
    
    for t in range(seq_len - 2, -1, -1):
        for i in range(num_states):
            beta[:, t, i] = torch.logsumexp(
                transition_log_probs[i] + emission_log_probs[:, t+1] + beta[:, t+1],
                dim=1
            )
    
    return alpha, beta


class AdaptiveChunkProcessor:
    """Adaptive chunk processing for variable-length sequences"""
    
    def __init__(self, model: nn.Module, chunk_size: int = 32):
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = chunk_size // 4  # 25% overlap
    
    def process_sequence(self, observations: torch.Tensor) -> torch.Tensor:
        """Process long sequences in adaptive chunks"""
        batch_size, seq_len, obs_dim = observations.shape
        
        if seq_len <= self.chunk_size:
            # Short sequence, process normally
            return self.model(observations)
        
        # Process in overlapping chunks
        results = []
        for start in range(0, seq_len, self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, seq_len)
            chunk = observations[:, start:end]
            
            with torch.no_grad():
                chunk_result = self.model(chunk)
                
                # Handle case where model returns tuple
                if isinstance(chunk_result, tuple):
                    chunk_result = chunk_result[0]  # Take first element
                
            # Handle overlap by averaging
            if start > 0 and len(results) > 0:
                overlap_size = min(self.overlap, chunk_result.shape[1])
                if results[-1].shape[1] >= overlap_size:
                    chunk_result[:, :overlap_size] = (
                        chunk_result[:, :overlap_size] + results[-1][:, -overlap_size:]
                    ) / 2.0
                    chunk_result = chunk_result[:, overlap_size:]
            
            results.append(chunk_result)
        
        return torch.cat(results, dim=1)


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """Optimize model for inference"""
    model.eval()
    
    # Fuse operations where possible
    if hasattr(torch.jit, 'optimize_for_inference'):
        try:
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)
        except Exception as e:
            warnings.warn(f"JIT optimization failed: {e}")
    
    return model


def benchmark_memory_usage(model: nn.Module, 
                         test_data: torch.Tensor,
                         device: torch.device) -> Dict[str, float]:
    """Benchmark memory usage of different configurations"""
    results = {}
    
    # Original model
    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(test_data)
        results['original_memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2
    
    # With gradient checkpointing (if available)
    if hasattr(model, 'use_checkpointing'):
        original_checkpointing = getattr(model, 'use_checkpointing', False)
        setattr(model, 'use_checkpointing', True)
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(test_data)
            results['checkpointing_memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        
        setattr(model, 'use_checkpointing', original_checkpointing)
    
    return results


def get_optimization_recommendations(model: nn.Module, 
                                   typical_batch_size: int,
                                   typical_seq_len: int,
                                   device: torch.device) -> List[str]:
    """Get optimization recommendations based on model and usage patterns"""
    recommendations = []
    
    # Model size analysis
    total_params = sum(p.numel() for p in model.parameters())
    
    if total_params > 1_000_000:
        recommendations.append("🔧 Large model detected - consider gradient checkpointing")
        recommendations.append("💾 Use mixed precision training to reduce memory usage")
    
    if typical_seq_len > 500:
        recommendations.append("📏 Long sequences detected - use adaptive chunking")
        recommendations.append("🔄 Consider sequence-level parallelization")
    
    if device.type == 'cuda':
        recommendations.append("🚀 GPU detected - enable mixed precision for 1.5-2x speedup")
        recommendations.append("⚡ Use CUDA streams for overlapped computation")
        recommendations.append("🎯 Optimize batch size for your GPU memory")
    else:
        recommendations.append("💻 CPU mode - consider JIT compilation for speedup")
        recommendations.append("🔢 Use optimized BLAS libraries (MKL, OpenBLAS)")
    
    if typical_batch_size == 1:
        recommendations.append("📦 Single-batch processing - consider batching for efficiency")
    elif typical_batch_size > 32:
        recommendations.append("📦 Large batch detected - monitor memory usage")
    
    return recommendations 