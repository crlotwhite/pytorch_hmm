"""
Real-time streaming HMM processor for low-latency speech processing
실시간 음성 처리를 위한 낮은 지연시간 HMM 프로세서

Author: Speech Synthesis Engineer  
Version: 0.2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import threading
import time
from typing import Optional, Tuple, List, Dict, Any
import warnings
import queue
from dataclasses import dataclass

from .hmm import HMMPyTorch


@dataclass
class StreamingResult:
    """Streaming processing result"""
    decoded_states: Optional[torch.Tensor]
    confidence: float
    processing_time_ms: float
    buffer_size: int
    chunk_id: int
    status: str
    metadata: Dict[str, Any]


class StreamingHMMProcessor(nn.Module):
    """
    실시간 음성 처리를 위한 스트리밍 HMM 프로세서
    낮은 지연시간과 온라인 처리 최적화
    
    Args:
        num_states: Number of HMM states
        feature_dim: Dimension of input features
        chunk_size: Size of each processing chunk (frames)
        overlap_size: Overlap between consecutive chunks
        lookahead_frames: Number of future frames to consider
        max_delay_frames: Maximum allowed delay in frames
        use_beam_search: Whether to use beam search decoding
        beam_width: Width of beam search
        buffer_size: Maximum buffer size for incoming audio
    """
    
    def __init__(
        self, 
        num_states: int, 
        feature_dim: int,
        chunk_size: int = 160,           # 10ms @ 16kHz
        overlap_size: int = 80,          # 5ms overlap
        lookahead_frames: int = 5,       # Future context
        max_delay_frames: int = 50,      # Maximum delay
        use_beam_search: bool = True,
        beam_width: int = 8,
        buffer_size: int = 1000
    ):
        super().__init__()
        
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.lookahead_frames = lookahead_frames
        self.max_delay_frames = max_delay_frames
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width
        self.buffer_size = buffer_size
        
        # HMM parameters
        self.transition_logits = nn.Parameter(
            torch.randn(num_states, num_states) * 0.1
        )
        self.emission_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_states),
            nn.LogSoftmax(dim=-1)
        )
        
        # Streaming state
        self.reset_streaming_state()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=1000)
        self.chunk_counter = 0
        self.total_frames_processed = 0
        
        # Threading for async processing
        self.processing_queue = queue.Queue(maxsize=self.buffer_size)
        self.result_queue = queue.Queue(maxsize=self.buffer_size)
        self.is_processing = False
        self.processing_thread = None
    
    def reset_streaming_state(self):
        """Reset streaming state for new audio stream"""
        self.feature_buffer = deque(maxlen=self.max_delay_frames + self.lookahead_frames)
        self.viterbi_states = []
        self.viterbi_scores = []
        self.beam_hypotheses = []
        self.last_output_frame = -1
        self.chunk_counter = 0
        self.total_frames_processed = 0
        
        # Initialize beam search hypotheses
        if self.use_beam_search:
            initial_score = -torch.log(torch.tensor(self.num_states, dtype=torch.float))
            self.beam_hypotheses = [
                (initial_score, [], s) for s in range(min(self.beam_width, self.num_states))
            ]
    
    def get_transition_matrix(self) -> torch.Tensor:
        """Get transition probability matrix"""
        return F.softmax(self.transition_logits, dim=-1)
    
    def start_async_processing(self):
        """Start asynchronous processing thread"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._async_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_async_processing(self):
        """Stop asynchronous processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def _async_processing_loop(self):
        """Main async processing loop"""
        while self.is_processing:
            try:
                # Get audio chunk from queue (with timeout)
                audio_chunk = self.processing_queue.get(timeout=0.1)
                
                # Process chunk
                result = self.process_chunk(audio_chunk)
                
                # Put result in output queue
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                warnings.warn(f"Error in async processing: {e}")
    
    def add_audio_chunk_async(self, audio_chunk: torch.Tensor) -> bool:
        """
        Add audio chunk for asynchronous processing
        
        Args:
            audio_chunk: (chunk_size, feature_dim) feature vectors
        
        Returns:
            True if successfully added, False if queue is full
        """
        try:
            self.processing_queue.put_nowait(audio_chunk)
            return True
        except queue.Full:
            return False
    
    def get_result_async(self) -> Optional[StreamingResult]:
        """Get processing result from async queue"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def process_chunk(self, audio_chunk: torch.Tensor) -> StreamingResult:
        """
        Process single audio chunk
        
        Args:
            audio_chunk: (chunk_size, feature_dim) feature vectors
        
        Returns:
            StreamingResult with decoding information
        """
        start_time = time.time()
        
        # Add to buffer
        for frame in audio_chunk:
            self.feature_buffer.append(frame)
        
        # Check if we have enough frames
        available_frames = len(self.feature_buffer)
        required_frames = self.chunk_size + self.lookahead_frames
        
        if available_frames < required_frames:
            processing_time = (time.time() - start_time) * 1000
            return StreamingResult(
                decoded_states=None,
                confidence=0.0,
                processing_time_ms=processing_time,
                buffer_size=available_frames,
                chunk_id=self.chunk_counter,
                status='buffering',
                metadata={'frames_needed': required_frames - available_frames}
            )
        
        # Determine processing range
        start_frame = max(0, self.last_output_frame + 1)
        end_frame = available_frames - self.lookahead_frames
        
        if end_frame <= start_frame:
            processing_time = (time.time() - start_time) * 1000
            return StreamingResult(
                decoded_states=None,
                confidence=0.0,
                processing_time_ms=processing_time,
                buffer_size=available_frames,
                chunk_id=self.chunk_counter,
                status='waiting_for_lookahead',
                metadata={}
            )
        
        # Extract features for processing
        features = torch.stack(list(self.feature_buffer)[start_frame:end_frame])
        
        # Decode
        if self.use_beam_search:
            decoded_states, confidence = self._beam_search_decode(features)
        else:
            decoded_states, confidence = self._greedy_decode(features)
        
        # Update state
        self.last_output_frame = end_frame - 1
        self.total_frames_processed += len(features)
        
        # Performance tracking
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        self.chunk_counter += 1
        
        # Calculate real-time factor
        frame_duration_ms = len(features) * 1000 / 100  # Assuming 100 fps
        rtf = frame_duration_ms / processing_time if processing_time > 0 else float('inf')
        
        return StreamingResult(
            decoded_states=decoded_states,
            confidence=confidence.mean().item() if confidence is not None else 0.0,
            processing_time_ms=processing_time,
            buffer_size=available_frames,
            chunk_id=self.chunk_counter,
            status='decoded',
            metadata={
                'frames_processed': len(features),
                'real_time_factor': rtf,
                'buffer_utilization': available_frames / self.feature_buffer.maxlen
            }
        )
    
    def _greedy_decode(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast greedy decoding for real-time processing
        
        Args:
            features: (seq_len, feature_dim)
        
        Returns:
            decoded_states: (seq_len,)
            confidence: (seq_len,) confidence scores
        """
        seq_len = features.shape[0]
        
        with torch.no_grad():
            # Get emission probabilities
            emission_log_probs = self.emission_net(features)  # (seq_len, num_states)
            
            # Get transition probabilities
            transition_matrix = self.get_transition_matrix()
            log_transitions = torch.log(transition_matrix + 1e-8)
            
            # Simple Viterbi for streaming
            states = torch.zeros(seq_len, dtype=torch.long, device=features.device)
            scores = torch.zeros(seq_len, device=features.device)
            
            if not self.viterbi_states:
                # First chunk - uniform initialization
                initial_scores = emission_log_probs[0] - torch.log(torch.tensor(self.num_states))
                states[0] = torch.argmax(initial_scores)
                scores[0] = initial_scores[states[0]]
            else:
                # Continue from previous state
                prev_state = self.viterbi_states[-1]
                transition_scores = log_transitions[prev_state] + emission_log_probs[0]
                states[0] = torch.argmax(transition_scores)
                scores[0] = transition_scores[states[0]]
            
            # Process remaining frames
            for t in range(1, seq_len):
                transition_scores = log_transitions[states[t-1]] + emission_log_probs[t]
                states[t] = torch.argmax(transition_scores)
                scores[t] = transition_scores[states[t]]
            
            # Update history
            self.viterbi_states.extend(states.tolist())
            self.viterbi_scores.extend(scores.tolist())
            
            # Keep history bounded
            if len(self.viterbi_states) > self.max_delay_frames:
                excess = len(self.viterbi_states) - self.max_delay_frames
                self.viterbi_states = self.viterbi_states[excess:]
                self.viterbi_scores = self.viterbi_scores[excess:]
        
        return states, torch.exp(scores)
    
    def _beam_search_decode(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search decoding for better accuracy
        
        Args:
            features: (seq_len, feature_dim)
        
        Returns:
            decoded_states: (seq_len,)
            confidence: (seq_len,) confidence scores  
        """
        seq_len = features.shape[0]
        
        with torch.no_grad():
            # Get emission probabilities
            emission_log_probs = self.emission_net(features)
            
            # Get transition probabilities
            transition_matrix = self.get_transition_matrix()
            log_transitions = torch.log(transition_matrix + 1e-8)
            
            # Beam search for each time step
            for t in range(seq_len):
                new_hypotheses = []
                
                for score, path, last_state in self.beam_hypotheses:
                    for next_state in range(self.num_states):
                        if t == 0 and not path:
                            # First frame
                            new_score = score + emission_log_probs[t, next_state]
                        else:
                            # Subsequent frames
                            transition_score = log_transitions[last_state, next_state]
                            emission_score = emission_log_probs[t, next_state]
                            new_score = score + transition_score + emission_score
                        
                        new_path = path + [next_state]
                        new_hypotheses.append((new_score, new_path, next_state))
                
                # Keep top-k hypotheses
                new_hypotheses.sort(key=lambda x: x[0], reverse=True)
                self.beam_hypotheses = new_hypotheses[:self.beam_width]
            
            # Extract best path
            best_score, best_path, _ = self.beam_hypotheses[0]
            
            # Get the last seq_len states
            if len(best_path) >= seq_len:
                states = torch.tensor(best_path[-seq_len:], dtype=torch.long, device=features.device)
            else:
                states = torch.tensor(best_path, dtype=torch.long, device=features.device)
            
            # Calculate confidence (normalized score)
            confidence = torch.full((seq_len,), torch.exp(best_score / len(best_path)), device=features.device)
        
        return states, confidence
    
    def flush_buffer(self) -> Optional[StreamingResult]:
        """
        Process remaining frames in buffer (for end of stream)
        
        Returns:
            Final processing result or None if buffer is empty
        """
        if len(self.feature_buffer) == 0:
            return None
        
        # Process all remaining frames
        remaining_features = torch.stack(list(self.feature_buffer))
        
        if self.use_beam_search:
            decoded_states, confidence = self._beam_search_decode(remaining_features)
        else:
            decoded_states, confidence = self._greedy_decode(remaining_features)
        
        self.chunk_counter += 1
        
        return StreamingResult(
            decoded_states=decoded_states,
            confidence=confidence.mean().item() if confidence is not None else 0.0,
            processing_time_ms=0.0,
            buffer_size=0,
            chunk_id=self.chunk_counter,
            status='flushed',
            metadata={'final_chunk': True}
        )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics"""
        if not self.processing_times:
            return {'message': 'No processing data available'}
        
        times_ms = list(self.processing_times)
        
        # Calculate statistics
        avg_time = sum(times_ms) / len(times_ms)
        max_time = max(times_ms)
        min_time = min(times_ms)
        
        # Real-time factor (assuming 100 fps feature extraction)
        frame_duration_ms = self.chunk_size * 1000 / 100
        avg_rtf = frame_duration_ms / avg_time if avg_time > 0 else float('inf')
        
        # Throughput
        throughput_fps = self.total_frames_processed / (sum(times_ms) / 1000) if times_ms else 0
        
        return {
            'total_chunks_processed': self.chunk_counter,
            'total_frames_processed': self.total_frames_processed,
            'avg_processing_time_ms': avg_time,
            'max_processing_time_ms': max_time,
            'min_processing_time_ms': min_time,
            'std_processing_time_ms': torch.tensor(times_ms).std().item(),
            'real_time_factor': avg_rtf,
            'throughput_fps': throughput_fps,
            'buffer_utilization': len(self.feature_buffer) / self.feature_buffer.maxlen,
            'chunk_size': self.chunk_size,
            'lookahead_frames': self.lookahead_frames,
            'beam_width': self.beam_width if self.use_beam_search else 1,
            'processing_mode': 'beam_search' if self.use_beam_search else 'greedy'
        }
    
    def optimize_for_latency(self, target_latency_ms: float = 50.0):
        """
        Automatically optimize parameters for target latency
        
        Args:
            target_latency_ms: Target processing latency in milliseconds
        """
        current_stats = self.get_performance_stats()
        
        if 'avg_processing_time_ms' not in current_stats:
            warnings.warn("No performance data available for optimization")
            return
        
        current_latency = current_stats['avg_processing_time_ms']
        
        if current_latency > target_latency_ms:
            # Reduce complexity to meet latency target
            if self.use_beam_search and self.beam_width > 2:
                self.beam_width = max(2, self.beam_width - 1)
                print(f"Reduced beam width to {self.beam_width}")
            
            elif self.use_beam_search:
                self.use_beam_search = False
                print("Switched to greedy decoding for lower latency")
            
            # Reduce chunk size if needed
            elif self.chunk_size > 80:
                self.chunk_size = max(80, int(self.chunk_size * 0.8))
                print(f"Reduced chunk size to {self.chunk_size}")
        
        elif current_latency < target_latency_ms * 0.5:
            # We have latency headroom, can increase quality
            if not self.use_beam_search:
                self.use_beam_search = True
                self.beam_width = 4
                print("Enabled beam search for better accuracy")
            
            elif self.beam_width < 8:
                self.beam_width += 1
                print(f"Increased beam width to {self.beam_width}")
    
    def get_latency_breakdown(self) -> Dict[str, float]:
        """Get detailed latency breakdown"""
        # This would require more detailed profiling in a real implementation
        stats = self.get_performance_stats()
        
        if 'avg_processing_time_ms' not in stats:
            return {}
        
        total_time = stats['avg_processing_time_ms']
        
        # Rough estimates (would need actual profiling)
        return {
            'feature_extraction': total_time * 0.1,  # 10%
            'emission_computation': total_time * 0.3,  # 30%
            'transition_computation': total_time * 0.1,  # 10%
            'viterbi_decoding': total_time * 0.4,  # 40%
            'bookkeeping': total_time * 0.1,  # 10%
            'total': total_time
        }


class AdaptiveLatencyController:
    """
    Adaptive latency controller for dynamic parameter adjustment
    네트워크 상황과 처리 성능에 따라 동적으로 매개변수 조정
    """
    
    def __init__(
        self, 
        initial_chunk_size: int = 160,
        min_chunk_size: int = 80,
        max_chunk_size: int = 320,
        target_latency_ms: float = 50.0,
        adaptation_rate: float = 0.1
    ):
        self.chunk_size = initial_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_latency_ms = target_latency_ms
        self.adaptation_rate = adaptation_rate
        
        self.latency_history = deque(maxlen=100)
        self.adjustment_cooldown = 0
        self.last_adjustment_time = 0
    
    def update(self, processing_time_ms: float, buffer_size: int) -> Dict[str, Any]:
        """
        Update controller with new performance data
        
        Args:
            processing_time_ms: Latest processing time
            buffer_size: Current buffer utilization
        
        Returns:
            Dictionary with recommended parameter changes
        """
        self.latency_history.append(processing_time_ms)
        current_time = time.time()
        
        # Don't adjust too frequently
        if current_time - self.last_adjustment_time < 1.0:  # 1 second cooldown
            return {}
        
        if len(self.latency_history) < 10:
            return {}
        
        # Calculate moving statistics
        recent_latencies = list(self.latency_history)[-20:]  # Last 20 measurements
        avg_latency = sum(recent_latencies) / len(recent_latencies)
        latency_variance = torch.tensor(recent_latencies).var().item()
        
        recommendations = {}
        
        # Latency too high - reduce complexity
        if avg_latency > self.target_latency_ms * 1.2:
            if self.chunk_size > self.min_chunk_size:
                new_chunk_size = max(
                    self.min_chunk_size,
                    int(self.chunk_size * (1 - self.adaptation_rate))
                )
                recommendations['chunk_size'] = new_chunk_size
                self.chunk_size = new_chunk_size
            
            recommendations['beam_width'] = max(1, int(4 * 0.8))  # Reduce beam width
            recommendations['use_beam_search'] = False if avg_latency > self.target_latency_ms * 2 else True
        
        # Latency acceptable with headroom - increase quality
        elif avg_latency < self.target_latency_ms * 0.6 and latency_variance < 10.0:
            if self.chunk_size < self.max_chunk_size and buffer_size > 100:
                new_chunk_size = min(
                    self.max_chunk_size,
                    int(self.chunk_size * (1 + self.adaptation_rate))
                )
                recommendations['chunk_size'] = new_chunk_size
                self.chunk_size = new_chunk_size
            
            recommendations['beam_width'] = min(8, 6)  # Increase beam width
            recommendations['use_beam_search'] = True
        
        # High variance - stabilize
        elif latency_variance > 25.0:
            recommendations['use_beam_search'] = False  # More predictable timing
            recommendations['chunk_size'] = max(self.min_chunk_size, int(self.chunk_size * 0.9))
        
        if recommendations:
            self.last_adjustment_time = current_time
        
        return recommendations


# Example usage and demo
def streaming_demo():
    """Demonstration of streaming HMM processor"""
    
    print("🎵 Real-time Streaming HMM Demo")
    print("=" * 40)
    
    # Create processor
    processor = StreamingHMMProcessor(
        num_states=10,
        feature_dim=80,
        chunk_size=160,  # 10ms chunks
        use_beam_search=True,
        beam_width=4
    )
    
    # Latency controller
    controller = AdaptiveLatencyController(target_latency_ms=30.0)
    
    # Simulate real-time processing
    print("Simulating real-time audio stream...")
    
    results = []
    total_chunks = 50
    
    for chunk_id in range(total_chunks):
        # Generate synthetic audio features
        audio_chunk = torch.randn(160, 80)  # 10ms of features
        
        # Process chunk
        result = processor.process_chunk(audio_chunk)
        
        if result.decoded_states is not None:
            results.append(result)
            
            # Adaptive control
            recommendations = controller.update(
                result.processing_time_ms,
                result.buffer_size
            )
            
            # Apply recommendations
            if 'chunk_size' in recommendations:
                processor.chunk_size = recommendations['chunk_size']
                print(f"Adapted chunk size: {processor.chunk_size}")
            
            if 'use_beam_search' in recommendations:
                processor.use_beam_search = recommendations['use_beam_search']
                print(f"Beam search: {processor.use_beam_search}")
        
        # Simulate real-time constraint
        time.sleep(0.01)  # 10ms
        
        if chunk_id % 10 == 0:
            stats = processor.get_performance_stats()
            print(f"Chunk {chunk_id}: "
                  f"Latency={stats.get('avg_processing_time_ms', 0):.1f}ms, "
                  f"RTF={stats.get('real_time_factor', 0):.1f}x")
    
    # Final statistics
    print(f"\n📊 Final Performance Stats:")
    final_stats = processor.get_performance_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n✅ Processed {len(results)} chunks successfully")
    print(f"Average confidence: {sum(r.confidence for r in results) / len(results):.3f}")


if __name__ == "__main__":
    streaming_demo()
