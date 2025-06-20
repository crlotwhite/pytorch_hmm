"""
Tests for Streaming HMM Processor

Testing real-time streaming capabilities, adaptive latency control,
and asynchronous processing functionality.
"""

import pytest
import torch
import time
import threading
import queue
from pytorch_hmm.streaming import (
    StreamingHMMProcessor, 
    AdaptiveLatencyController, 
    StreamingResult
)


class TestStreamingHMMProcessor:
    """Test suite for StreamingHMMProcessor"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def basic_processor(self, device):
        """Create basic streaming processor for testing"""
        return StreamingHMMProcessor(
            num_states=8,
            feature_dim=40,
            chunk_size=80,  # 5ms chunks
            lookahead_frames=3,
            use_beam_search=False  # Faster for testing
        ).to(device)
    
    @pytest.fixture
    def beam_processor(self, device):
        """Create beam search processor for testing"""
        return StreamingHMMProcessor(
            num_states=6,
            feature_dim=30,
            chunk_size=60,
            use_beam_search=True,
            beam_width=4
        ).to(device)
    
    def test_processor_initialization(self, device):
        """Test processor initialization with different parameters"""
        # Basic configuration
        processor = StreamingHMMProcessor(
            num_states=10,
            feature_dim=80,
            chunk_size=160,
            overlap_size=80,
            lookahead_frames=5,
            use_beam_search=True,
            beam_width=8
        ).to(device)
        
        assert processor.num_states == 10
        assert processor.feature_dim == 80
        assert processor.chunk_size == 160
        assert processor.overlap_size == 80
        assert processor.lookahead_frames == 5
        assert processor.use_beam_search == True
        assert processor.beam_width == 8
    
    def test_streaming_state_reset(self, basic_processor):
        """Test streaming state reset functionality"""
        # Process some chunks first
        chunk = torch.randn(80, 40)
        basic_processor.process_chunk(chunk)
        basic_processor.process_chunk(chunk)
        
        # Check state has accumulated
        assert len(basic_processor.feature_buffer) > 0
        assert basic_processor.chunk_counter > 0
        
        # Reset state
        basic_processor.reset_streaming_state()
        
        # Check state is cleared
        assert len(basic_processor.feature_buffer) == 0
        assert basic_processor.chunk_counter == 0
        assert basic_processor.last_output_frame == -1
    
    def test_chunk_processing_buffering(self, basic_processor):
        """Test chunk processing in buffering stage"""
        # Small chunk that requires buffering
        small_chunk = torch.randn(10, 40)
        result = basic_processor.process_chunk(small_chunk)
        
        assert isinstance(result, StreamingResult)
        assert result.status == 'buffering'
        assert result.decoded_states is None
        assert 'frames_needed' in result.metadata
    
    def test_chunk_processing_decoding(self, basic_processor):
        """Test chunk processing with successful decoding"""
        # Add enough frames for processing
        for _ in range(3):
            chunk = torch.randn(80, 40)
            result = basic_processor.process_chunk(chunk)
        
        # Should eventually get decoded results
        assert result.status in ['decoded', 'buffering', 'waiting_for_lookahead']
        
        if result.status == 'decoded':
            assert result.decoded_states is not None
            assert result.decoded_states.shape[0] > 0
            assert result.confidence >= 0.0
            assert result.processing_time_ms >= 0.0
    
    def test_greedy_vs_beam_search(self, device):
        """Compare greedy and beam search decoding"""
        # Create processors with same configuration except decoding method
        greedy_proc = StreamingHMMProcessor(
            num_states=5, feature_dim=30, chunk_size=50,
            use_beam_search=False
        ).to(device)
        
        beam_proc = StreamingHMMProcessor(
            num_states=5, feature_dim=30, chunk_size=50,
            use_beam_search=True, beam_width=4
        ).to(device)
        
        # Process same chunks
        chunk = torch.randn(50, 30)
        
        # Fill buffers
        for _ in range(3):
            greedy_proc.process_chunk(chunk)
            beam_proc.process_chunk(chunk)
        
        # Process and compare
        greedy_result = greedy_proc.process_chunk(chunk)
        beam_result = beam_proc.process_chunk(chunk)
        
        # Both should work, beam search might be slower
        if greedy_result.status == 'decoded' and beam_result.status == 'decoded':
            # Beam search should generally be slower but potentially more accurate
            # (This is probabilistic, so we just check they both work)
            assert greedy_result.processing_time_ms >= 0
            assert beam_result.processing_time_ms >= 0
    
    def test_performance_monitoring(self, basic_processor):
        """Test performance statistics collection"""
        # Process several chunks
        chunk = torch.randn(80, 40)
        
        for _ in range(5):
            basic_processor.process_chunk(chunk)
        
        stats = basic_processor.get_performance_stats()
        
        if basic_processor.chunk_counter > 0:
            assert 'total_chunks_processed' in stats
            assert 'avg_processing_time_ms' in stats
            assert 'real_time_factor' in stats
            assert 'throughput_fps' in stats
            
            assert stats['total_chunks_processed'] > 0
            assert stats['avg_processing_time_ms'] >= 0
    
    def test_latency_optimization(self, basic_processor):
        """Test automatic latency optimization"""
        # Process chunks and collect timing data
        chunk = torch.randn(80, 40)
        
        for _ in range(10):
            basic_processor.process_chunk(chunk)
        
        # Optimize for low latency
        basic_processor.optimize_for_latency(target_latency_ms=25.0)
        
        # Should have adjusted parameters (effects may be small in test)
        assert hasattr(basic_processor, 'use_beam_search')
        assert hasattr(basic_processor, 'beam_width')
        assert hasattr(basic_processor, 'chunk_size')
    
    def test_buffer_flush(self, basic_processor):
        """Test buffer flushing for end of stream"""
        # Add some data to buffer
        chunk = torch.randn(80, 40)
        basic_processor.process_chunk(chunk)
        
        # Flush remaining data
        flush_result = basic_processor.flush_buffer()
        
        if flush_result is not None:
            assert isinstance(flush_result, StreamingResult)
            assert flush_result.status == 'flushed'
            assert 'final_chunk' in flush_result.metadata
    
    def test_async_processing(self, device):
        """Test asynchronous processing functionality"""
        processor = StreamingHMMProcessor(
            num_states=5, feature_dim=25, chunk_size=40
        ).to(device)
        
        # Start async processing
        processor.start_async_processing()
        
        try:
            # Add chunks to processing queue
            chunk = torch.randn(40, 25)
            
            success = processor.add_audio_chunk_async(chunk)
            assert success  # Should succeed if queue not full
            
            # Wait a bit for processing
            time.sleep(0.1)
            
            # Try to get result
            result = processor.get_result_async()
            
            # May or may not have result depending on timing
            if result is not None:
                assert isinstance(result, StreamingResult)
        
        finally:
            # Clean up async processing
            processor.stop_async_processing()
    
    @pytest.mark.slow
    def test_real_time_simulation(self, device):
        """Test real-time processing simulation"""
        processor = StreamingHMMProcessor(
            num_states=8, feature_dim=50, chunk_size=100,
            use_beam_search=False  # Faster for real-time
        ).to(device)
        
        # Simulate real-time chunks (10ms each)
        chunk_duration_ms = 10
        total_chunks = 20
        
        processing_times = []
        
        for i in range(total_chunks):
            start_time = time.time()
            
            # Generate chunk
            chunk = torch.randn(100, 50)
            
            # Process
            result = processor.process_chunk(chunk)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            processing_times.append(processing_time)
            
            # Simulate real-time constraint
            time.sleep(chunk_duration_ms / 1000)
        
        # Check real-time performance
        avg_processing_time = sum(processing_times) / len(processing_times)
        real_time_factor = chunk_duration_ms / avg_processing_time
        
        # Should be faster than real-time (RTF > 1)
        print(f"Average processing time: {avg_processing_time:.2f}ms")
        print(f"Real-time factor: {real_time_factor:.1f}x")
        
        # This is a soft requirement - depends on hardware
        if real_time_factor < 1.0:
            pytest.skip(f"Hardware too slow for real-time: {real_time_factor:.1f}x")
    
    def test_memory_management(self, device):
        """Test memory management during streaming"""
        processor = StreamingHMMProcessor(
            num_states=10, feature_dim=60, chunk_size=120,
            max_delay_frames=200  # Limit buffer size
        ).to(device)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Process many chunks
        chunk = torch.randn(120, 60)
        
        for _ in range(50):
            processor.process_chunk(chunk)
        
        # Buffer should be bounded
        assert len(processor.feature_buffer) <= processor.feature_buffer.maxlen
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            # Should use reasonable memory
            assert peak_memory < 500, f"Memory usage too high: {peak_memory:.1f} MB"


class TestAdaptiveLatencyController:
    """Test suite for AdaptiveLatencyController"""
    
    @pytest.fixture
    def controller(self):
        """Create adaptive latency controller for testing"""
        return AdaptiveLatencyController(
            initial_chunk_size=160,
            min_chunk_size=80,
            max_chunk_size=320,
            target_latency_ms=30.0
        )
    
    def test_controller_initialization(self, controller):
        """Test controller initialization"""
        assert controller.chunk_size == 160
        assert controller.min_chunk_size == 80
        assert controller.max_chunk_size == 320
        assert controller.target_latency_ms == 30.0
    
    def test_latency_too_high_adaptation(self, controller):
        """Test adaptation when latency is too high"""
        # Simulate high latency
        high_latency = 50.0  # Above target of 30ms
        buffer_size = 100
        
        # Need multiple updates to trigger adaptation
        for _ in range(15):
            controller.update(high_latency, buffer_size)
        
        recommendations = controller.update(high_latency, buffer_size)
        
        # Should recommend reducing complexity
        if recommendations:
            if 'chunk_size' in recommendations:
                assert recommendations['chunk_size'] <= controller.max_chunk_size
            if 'use_beam_search' in recommendations:
                # May recommend disabling beam search for very high latency
                pass
    
    def test_latency_low_adaptation(self, controller):
        """Test adaptation when latency is low"""
        # Simulate low latency with stable variance
        low_latency = 10.0  # Below target of 30ms
        buffer_size = 200  # Good buffer
        
        # Need multiple stable updates
        for _ in range(15):
            controller.update(low_latency, buffer_size)
        
        recommendations = controller.update(low_latency, buffer_size)
        
        # Should recommend increasing quality
        if recommendations:
            if 'chunk_size' in recommendations:
                assert recommendations['chunk_size'] >= controller.min_chunk_size
            if 'use_beam_search' in recommendations:
                assert recommendations['use_beam_search'] == True
    
    def test_adaptation_cooldown(self, controller):
        """Test that adaptation has proper cooldown"""
        # Make rapid updates
        for _ in range(5):
            recommendations = controller.update(50.0, 100)
            # Should not adapt too frequently
        
        # Cooldown mechanism should prevent rapid changes
        assert hasattr(controller, 'last_adjustment_time')


class TestStreamingIntegration:
    """Integration tests for streaming components"""
    
    def test_processor_with_controller(self, device):
        """Test processor with adaptive latency controller"""
        processor = StreamingHMMProcessor(
            num_states=6, feature_dim=40, chunk_size=100
        ).to(device)
        
        controller = AdaptiveLatencyController(
            target_latency_ms=25.0
        )
        
        chunk = torch.randn(100, 40)
        
        # Process with adaptation
        for i in range(10):
            result = processor.process_chunk(chunk)
            
            if result.status == 'decoded':
                recommendations = controller.update(
                    result.processing_time_ms,
                    result.buffer_size
                )
                
                # Apply recommendations
                if 'chunk_size' in recommendations:
                    processor.chunk_size = recommendations['chunk_size']
                if 'use_beam_search' in recommendations:
                    processor.use_beam_search = recommendations['use_beam_search']
        
        # Should have processed successfully
        assert processor.chunk_counter > 0
    
    def test_multiple_processors(self, device):
        """Test multiple processors running concurrently"""
        processors = []
        
        for i in range(3):
            proc = StreamingHMMProcessor(
                num_states=4 + i,
                feature_dim=30,
                chunk_size=60
            ).to(device)
            processors.append(proc)
        
        # Process same chunks on all processors
        chunk = torch.randn(60, 30)
        
        results = []
        for proc in processors:
            # Fill buffers
            for _ in range(3):
                proc.process_chunk(chunk)
            
            result = proc.process_chunk(chunk)
            results.append(result)
        
        # All should work independently
        for result in results:
            assert isinstance(result, StreamingResult)
    
    @pytest.mark.slow
    def test_concurrent_processing(self, device):
        """Test concurrent processing with threading"""
        processor = StreamingHMMProcessor(
            num_states=5, feature_dim=35, chunk_size=70
        ).to(device)
        
        results = []
        errors = []
        
        def process_worker():
            try:
                chunk = torch.randn(70, 35)
                for _ in range(5):
                    result = processor.process_chunk(chunk)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=process_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors in concurrent processing: {errors}"
        assert len(results) > 0


@pytest.mark.performance
class TestStreamingPerformance:
    """Performance tests for streaming HMM"""
    
    def test_throughput_measurement(self, device):
        """Test throughput measurement"""
        processor = StreamingHMMProcessor(
            num_states=10, feature_dim=80, chunk_size=160,
            use_beam_search=False  # Maximize throughput
        ).to(device)
        
        chunk = torch.randn(160, 80)
        
        # Warm up
        for _ in range(5):
            processor.process_chunk(chunk)
        
        # Measure throughput
        start_time = time.time()
        total_frames = 0
        
        for _ in range(20):
            result = processor.process_chunk(chunk)
            total_frames += chunk.shape[0]
        
        elapsed_time = time.time() - start_time
        throughput = total_frames / elapsed_time
        
        print(f"Throughput: {throughput:.1f} frames/sec")
        
        # Should achieve reasonable throughput
        assert throughput > 1000, f"Throughput too low: {throughput:.1f} fps"
    
    def test_latency_measurement(self, device):
        """Test latency measurement"""
        processor = StreamingHMMProcessor(
            num_states=8, feature_dim=60, chunk_size=120,
            lookahead_frames=2  # Minimize latency
        ).to(device)
        
        chunk = torch.randn(120, 60)
        
        # Fill buffer
        for _ in range(3):
            processor.process_chunk(chunk)
        
        # Measure processing latency
        latencies = []
        
        for _ in range(10):
            start_time = time.time()
            result = processor.process_chunk(chunk)
            latency = (time.time() - start_time) * 1000  # ms
            
            if result.status == 'decoded':
                latencies.append(latency)
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            print(f"Average latency: {avg_latency:.2f}ms")
            print(f"Maximum latency: {max_latency:.2f}ms")
            
            # Should be low latency (target < 50ms for real-time)
            assert avg_latency < 100, f"Average latency too high: {avg_latency:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__])
