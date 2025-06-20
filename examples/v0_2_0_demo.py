"""
PyTorch HMM v0.2.0 New Features Demo

This example demonstrates the new features introduced in v0.2.0:
- Mixture Gaussian HMM for complex acoustic modeling
- Hidden Semi-Markov Model (HSMM) with explicit duration modeling  
- Real-time streaming HMM processor
- Advanced transition matrix utilities
- Korean TTS optimizations

Run with: python examples/v0_2_0_demo.py
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Import new v0.2.0 features
from pytorch_hmm import (
    MixtureGaussianHMMLayer,
    HSMMLayer, 
    StreamingHMMProcessor,
    create_skip_state_matrix,
    create_phoneme_aware_transitions,
    create_korean_tts_hmm,
    create_speech_hmm,
    get_speech_transitions,
    analyze_transition_patterns,
    Config
)


def demo_mixture_gaussian_hmm():
    """Demonstrate Mixture Gaussian HMM for complex acoustic modeling"""
    print("🎵 Mixture Gaussian HMM Demo")
    print("=" * 40)
    
    # Create model with 3 Gaussian components per state
    model = MixtureGaussianHMMLayer(
        num_states=8,               # 8 phoneme states
        feature_dim=80,             # 80-dim mel-spectrogram
        num_components=3,           # 3 Gaussians per state
        covariance_type='diag',     # Diagonal covariance
        learnable_transitions=True
    )
    
    # Generate synthetic speech-like data
    batch_size, seq_len = 4, 200
    features = torch.randn(batch_size, seq_len, 80)
    
    # Add some temporal correlation (speech-like)
    for b in range(batch_size):
        for t in range(1, seq_len):
            features[b, t] = 0.7 * features[b, t-1] + 0.3 * features[b, t]
    
    print(f"Input features shape: {features.shape}")
    
    # Decode with HMM
    start_time = time.time()
    decoded_states, log_probs = model(features, return_log_probs=True)
    processing_time = time.time() - start_time
    
    print(f"Decoded states shape: {decoded_states.shape}")
    print(f"Log probabilities shape: {log_probs.shape}")
    print(f"Processing time: {processing_time:.3f}s")
    print(f"Throughput: {batch_size * seq_len / processing_time:.1f} frames/sec")
    
    # Analyze state transitions
    from pytorch_hmm.utils import compute_state_durations
    
    for b in range(min(2, batch_size)):
        durations = compute_state_durations(decoded_states[b])
        print(f"Batch {b} - State durations: {durations[:10].tolist()}...")  # First 10
        print(f"         - Average duration: {durations.float().mean():.1f} frames")
    
    # Get model info
    info = model.get_model_info()
    print(f"Model parameters: {info['total_parameters']:,}")
    print(f"Memory efficient: {info['memory_efficient']}")
    
    return model, decoded_states


def demo_hsmm_duration_modeling():
    """Demonstrate HSMM with explicit duration modeling"""
    print("\n⏱️ Hidden Semi-Markov Model (HSMM) Demo")
    print("=" * 45)
    
    # Create HSMM with Gamma duration distribution
    hsmm = HSMMLayer(
        num_states=5,                    # 5 phoneme states
        feature_dim=40,                  # 40-dim MFCC features
        duration_distribution='gamma',   # Gamma distribution for durations
        max_duration=30,                 # Max 30 frames per state
        min_duration=3                   # Min 3 frames per state
    )
    
    print(f"Duration distribution: {hsmm.duration_distribution}")
    print(f"Expected durations: {hsmm.get_expected_durations()}")
    
    # Generate a sequence with explicit duration modeling
    seq_length = 150
    generated_states, generated_features = hsmm.generate_sequence(seq_length)
    
    print(f"Generated sequence length: {len(generated_states)}")
    print(f"Generated features shape: {generated_features.shape}")
    
    # Analyze the generated durations
    from pytorch_hmm.utils import compute_state_durations
    durations = compute_state_durations(generated_states)
    
    print(f"Number of state segments: {len(durations)}")
    print(f"Duration statistics:")
    print(f"  Mean: {durations.float().mean():.1f} frames")
    print(f"  Std:  {durations.float().std():.1f} frames")
    print(f"  Min:  {durations.min()} frames")
    print(f"  Max:  {durations.max()} frames")
    
    # Test HSMM decoding
    test_features = torch.randn(1, 80, 40)
    start_time = time.time()
    decoded_states, log_probs = hsmm(test_features)
    decode_time = time.time() - start_time
    
    print(f"HSMM decoding time: {decode_time:.3f}s for {test_features.shape[1]} frames")
    
    # Compare with different duration distributions
    print("\nComparing duration distributions:")
    distributions = ['gamma', 'poisson', 'weibull']
    
    for dist in distributions:
        model = HSMMLayer(num_states=3, feature_dim=20, duration_distribution=dist, max_duration=15)
        states, _ = model.generate_sequence(60)
        durations = compute_state_durations(states)
        
        print(f"  {dist:>8}: mean={durations.float().mean():.1f}, std={durations.float().std():.1f}")
    
    return hsmm, generated_states, generated_features


def demo_streaming_hmm():
    """Demonstrate real-time streaming HMM processor"""
    print("\n🚀 Real-time Streaming HMM Demo")
    print("=" * 35)
    
    # Create streaming processor
    processor = StreamingHMMProcessor(
        num_states=6,
        feature_dim=50,
        chunk_size=100,           # 100 frames per chunk (~10ms)
        lookahead_frames=5,       # 5 frames lookahead
        use_beam_search=True,     # Better accuracy
        beam_width=4
    )
    
    print(f"Chunk size: {processor.chunk_size} frames")
    print(f"Beam width: {processor.beam_width}")
    print(f"Lookahead: {processor.lookahead_frames} frames")
    
    # Simulate real-time audio stream
    total_chunks = 20
    chunk_times = []
    
    print("\nProcessing real-time audio stream...")
    
    for i in range(total_chunks):
        # Generate audio chunk (simulating microphone input)
        audio_chunk = torch.randn(100, 50)
        
        # Process chunk
        start_time = time.time()
        result = processor.process_chunk(audio_chunk)
        chunk_time = (time.time() - start_time) * 1000  # ms
        
        chunk_times.append(chunk_time)
        
        # Print results
        if result.decoded_states is not None:
            print(f"Chunk {i:2d}: {result.status:>10} | "
                  f"States: {len(result.decoded_states):2d} | "
                  f"Time: {result.processing_time_ms:5.1f}ms | "
                  f"Confidence: {result.confidence:.3f}")
        else:
            print(f"Chunk {i:2d}: {result.status:>10} | "
                  f"Buffer: {result.buffer_size:3d} frames")
        
        # Simulate real-time constraint (10ms per chunk)
        time.sleep(0.01)
    
    # Performance statistics
    stats = processor.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"  Total chunks processed: {stats['total_chunks_processed']}")
    print(f"  Average processing time: {stats['avg_processing_time_ms']:.2f} ms")
    print(f"  Real-time factor: {stats['real_time_factor']:.1f}x")
    print(f"  Throughput: {stats['throughput_fps']:.1f} frames/sec")
    
    # Test latency optimization
    print(f"\n🔧 Optimizing for low latency...")
    processor.optimize_for_latency(target_latency_ms=15.0)
    
    final_stats = processor.get_performance_stats()
    print(f"  After optimization:")
    print(f"    Beam search: {processor.use_beam_search}")
    print(f"    Beam width: {processor.beam_width}")
    print(f"    Chunk size: {processor.chunk_size}")
    
    return processor


def demo_advanced_transitions():
    """Demonstrate advanced transition matrix utilities"""
    print("\n🔄 Advanced Transition Matrix Demo")
    print("=" * 38)
    
    # 1. Skip-state transitions for fast speech
    print("1. Skip-state transitions for fast speech:")
    skip_matrix = create_skip_state_matrix(
        num_states=8,
        self_loop_prob=0.5,
        forward_prob=0.4, 
        skip_prob=0.1,
        max_skip=2
    )
    
    print(f"   Matrix shape: {skip_matrix.shape}")
    print(f"   Skip transitions (state 0): {skip_matrix[0, 2:4]}")
    
    # 2. Phoneme-aware transitions
    print("\n2. Phoneme-aware transitions:")
    korean_durations = [4, 6, 3, 8, 5, 7, 4, 6]  # Example Korean phoneme durations
    phoneme_matrix = create_phoneme_aware_transitions(korean_durations)
    
    print(f"   Matrix shape: {phoneme_matrix.shape}")
    for i, duration in enumerate(korean_durations[:4]):
        self_prob = phoneme_matrix[i, i].item()
        print(f"   Phoneme {i} (duration {duration}): self-loop = {self_prob:.3f}")
    
    # 3. Speech type transitions
    print("\n3. Different speech types:")
    speech_types = ['normal', 'fast', 'slow', 'emotional']
    
    for speech_type in speech_types:
        transitions = get_speech_transitions(6, speech_type)
        
        # Analyze transition patterns
        self_loops = torch.diag(transitions).mean().item()
        forward_trans = torch.diag(transitions, diagonal=1).mean().item()
        
        print(f"   {speech_type:>10}: self-loop={self_loops:.3f}, forward={forward_trans:.3f}")
    
    return skip_matrix, phoneme_matrix


def demo_korean_tts():
    """Demonstrate Korean TTS optimized HMM"""
    print("\n🇰🇷 Korean TTS Optimized HMM Demo")
    print("=" * 35)
    
    # Create Korean TTS model
    korean_model = create_korean_tts_hmm(
        model_type='mixture_gaussian',
        feature_dim=80,
        num_components=3
    )
    
    print(f"Korean model states: {korean_model.num_states}")
    print(f"Feature dimension: {korean_model.feature_dim}")
    print(f"Components per state: {korean_model.num_components}")
    
    # Simulate Korean speech features
    korean_text = "안녕하세요"  # "Hello" in Korean
    sequence_length = 120  # ~1.2 seconds at 100fps
    
    # Generate features that simulate Korean phoneme characteristics
    features = torch.randn(1, sequence_length, 80)
    
    # Add Korean-specific characteristics (example)
    # Korean has distinctive F0 patterns and vowel formants
    for t in range(sequence_length):
        # Simulate vowel regions with higher energy in certain frequency bands
        if t % 15 < 8:  # Vowel-like regions
            features[0, t, 10:25] += 0.5  # Boost mid frequencies
            features[0, t, 40:55] += 0.3  # Boost higher formants
    
    # Decode with Korean model
    start_time = time.time()
    korean_states, korean_log_probs = korean_model(features, return_log_probs=True)
    korean_time = time.time() - start_time
    
    print(f"Korean decoding time: {korean_time:.3f}s")
    print(f"Sequence log probability: {korean_log_probs.item():.2f}")
    
    # Analyze Korean phoneme durations
    from pytorch_hmm.utils import compute_state_durations
    korean_durations = compute_state_durations(korean_states[0])
    
    print(f"Korean phoneme analysis:")
    print(f"  Number of phonemes: {len(korean_durations)}")
    print(f"  Average duration: {korean_durations.float().mean():.1f} frames")
    print(f"  Duration range: {korean_durations.min()}-{korean_durations.max()} frames")
    
    return korean_model, korean_states


def demo_model_factory():
    """Demonstrate model factory for different use cases"""
    print("\n🏭 Model Factory Demo")
    print("=" * 25)
    
    # 1. ASR model
    asr_model = create_speech_hmm(
        num_states=30,              # Large vocabulary
        feature_dim=80,
        model_type='mixture_gaussian',
        num_components=4            # More components for ASR
    )
    
    print(f"ASR Model: {asr_model.num_states} states, {asr_model.num_components} components")
    
    # 2. TTS model with duration modeling
    tts_model = create_speech_hmm(
        num_states=25,
        feature_dim=80,
        model_type='hsmm',
        duration_distribution='gamma',
        max_duration=40
    )
    
    print(f"TTS Model: {tts_model.num_states} states, {tts_model.duration_distribution} duration")
    
    # 3. Real-time model
    realtime_model = create_speech_hmm(
        num_states=15,
        feature_dim=40,
        model_type='streaming',
        chunk_size=80,              # Small chunks for low latency
        use_beam_search=False       # Fast greedy decoding
    )
    
    print(f"Real-time Model: {realtime_model.num_states} states, {realtime_model.chunk_size} chunk size")
    
    return asr_model, tts_model, realtime_model


def demo_performance_comparison():
    """Compare performance of different model types"""
    print("\n📊 Performance Comparison")
    print("=" * 28)
    
    models = {
        'Basic HMM': create_speech_hmm(10, 40, 'mixture_gaussian', num_components=1),
        'Mixture GMM': create_speech_hmm(10, 40, 'mixture_gaussian', num_components=3),
        'HSMM': create_speech_hmm(10, 40, 'hsmm', max_duration=20)
    }
    
    # Test data
    test_data = torch.randn(4, 100, 40)
    
    print(f"{'Model':>12} | {'Time (ms)':>10} | {'Memory (MB)':>12} | {'Throughput':>12}")
    print("-" * 60)
    
    for name, model in models.items():
        # Measure processing time
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        start_time = time.time()
        if hasattr(model, 'viterbi_decode_hsmm'):
            # HSMM
            states, _ = model(test_data)
        else:
            # Regular HMM
            states, _ = model(test_data)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            memory_mb = 0
        
        # Throughput
        total_frames = test_data.shape[0] * test_data.shape[1]
        throughput = total_frames / (processing_time / 1000)
        
        print(f"{name:>12} | {processing_time:>8.1f} | {memory_mb:>10.1f} | {throughput:>9.1f} fps")


def main():
    """Run all demos"""
    print("🎉 PyTorch HMM v0.2.0 Feature Demonstration")
    print("=" * 50)
    print("New features showcase:")
    print("✅ Mixture Gaussian HMM")
    print("✅ Hidden Semi-Markov Model (HSMM)")  
    print("✅ Real-time Streaming HMM")
    print("✅ Advanced Transition Matrices")
    print("✅ Korean TTS Optimizations")
    print("✅ Model Factory & Utilities")
    print()
    
    # Configure for optimal performance
    Config.set_device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Configuration: {Config.get_info()}")
    
    try:
        # Run all demos
        demo_mixture_gaussian_hmm()
        demo_hsmm_duration_modeling()
        demo_streaming_hmm()
        demo_advanced_transitions()
        demo_korean_tts()
        demo_model_factory()
        demo_performance_comparison()
        
        print("\n🎊 All demos completed successfully!")
        print("PyTorch HMM v0.2.0 is ready for advanced speech processing tasks.")
        
        # Quick integration test
        print("\n🧪 Running quick integration test...")
        import pytorch_hmm
        success = pytorch_hmm.run_quick_test()
        
        if success:
            print("✅ Integration test passed!")
        else:
            print("❌ Integration test failed!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
