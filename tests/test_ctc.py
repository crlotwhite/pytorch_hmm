"""
Comprehensive tests for pytorch_hmm.alignment.ctc module
"""

import pytest
import torch
import numpy as np
from pytorch_hmm.alignment.ctc import (
    expand_targets_with_blank,
    ctc_forward_algorithm,
    ctc_backward_algorithm,
    ctc_alignment_path,
    CTCAligner,
    CTCSegmentationAligner,
    remove_ctc_blanks,
    collapse_repeated_tokens,
    ctc_decode_sequence
)


class TestCTCUtilities:
    """Test CTC utility functions"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_expand_targets_with_blank(self, device):
        """Test target expansion with blank tokens"""
        # Simple case
        targets = torch.tensor([[1, 2, 3]], device=device)
        blank_id = 0
        
        expanded = expand_targets_with_blank(targets, blank_id)
        expected = torch.tensor([[0, 1, 0, 2, 0, 3, 0]], device=device)
        
        assert torch.equal(expanded, expected)
        assert expanded.shape == (1, 7)  # 2*3+1
    
    def test_expand_targets_batch(self, device):
        """Test batch target expansion"""
        targets = torch.tensor([[1, 2], [3, 4]], device=device)
        blank_id = 0
        
        expanded = expand_targets_with_blank(targets, blank_id)
        expected = torch.tensor([[0, 1, 0, 2, 0], [0, 3, 0, 4, 0]], device=device)
        
        assert torch.equal(expanded, expected)
        assert expanded.shape == (2, 5)  # 2*2+1
    
    def test_remove_ctc_blanks(self, device):
        """Test blank token removal"""
        sequence = torch.tensor([0, 1, 0, 2, 0, 0, 3, 0], device=device)
        result = remove_ctc_blanks(sequence, blank_id=0)
        expected = torch.tensor([1, 2, 3], device=device)
        
        assert torch.equal(result, expected)
    
    def test_collapse_repeated_tokens(self, device):
        """Test repeated token collapse"""
        sequence = torch.tensor([1, 1, 2, 2, 2, 3, 1, 1], device=device)
        result = collapse_repeated_tokens(sequence)
        expected = torch.tensor([1, 2, 3, 1], device=device)
        
        assert torch.equal(result, expected)
    
    def test_ctc_decode_sequence(self, device):
        """Test full CTC decoding"""
        # Sequence with blanks and repeats
        sequence = torch.tensor([0, 1, 1, 0, 2, 2, 0, 3, 0], device=device)
        result = ctc_decode_sequence(sequence, blank_id=0)
        expected = torch.tensor([1, 2, 3], device=device)
        
        assert torch.equal(result, expected)
    
    def test_ctc_decode_empty_sequence(self, device):
        """Test decoding empty sequence"""
        sequence = torch.tensor([0, 0, 0], device=device)
        result = ctc_decode_sequence(sequence, blank_id=0)
        expected = torch.tensor([], device=device, dtype=torch.long)
        
        assert torch.equal(result, expected)


class TestCTCAlgorithms:
    """Test CTC forward/backward algorithms"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def simple_data(self, device):
        """Simple test data for CTC algorithms"""
        # 3 time steps, 1 batch, 4 classes (including blank=0)
        log_probs = torch.log_softmax(torch.randn(3, 1, 4, device=device), dim=-1)
        targets = torch.tensor([[1, 2]], device=device)  # target sequence [1, 2]
        input_lengths = torch.tensor([3], device=device)
        target_lengths = torch.tensor([2], device=device)
        
        return log_probs, targets, input_lengths, target_lengths
    
    def test_ctc_forward_algorithm(self, simple_data, device):
        """Test CTC forward algorithm"""
        log_probs, targets, input_lengths, target_lengths = simple_data
        
        log_likelihood = ctc_forward_algorithm(
            log_probs, targets, input_lengths, target_lengths, blank_id=0
        )
        
        assert log_likelihood.shape == (1,)
        assert torch.isfinite(log_likelihood).all()
        assert log_likelihood.item() <= 0  # log probability should be <= 0
    
    def test_ctc_backward_algorithm(self, simple_data, device):
        """Test CTC backward algorithm"""
        log_probs, targets, input_lengths, target_lengths = simple_data
        
        log_beta = ctc_backward_algorithm(
            log_probs, targets, input_lengths, target_lengths, blank_id=0
        )
        
        batch_size, max_time, max_expanded_length = log_beta.shape
        assert batch_size == 1
        assert max_time == 3
        assert max_expanded_length == 5  # 2*2+1
        assert torch.isfinite(log_beta[log_beta > float('-inf')]).all()
    
    def test_ctc_alignment_path(self, simple_data, device):
        """Test CTC alignment path computation"""
        log_probs, targets, input_lengths, target_lengths = simple_data
        
        paths = ctc_alignment_path(
            log_probs, targets, input_lengths, target_lengths, blank_id=0
        )
        
        assert len(paths) == 1  # batch size
        assert paths[0].shape[0] == 3  # sequence length
        assert torch.all(paths[0] >= 0)  # valid token indices
    
    def test_ctc_algorithms_batch(self, device):
        """Test CTC algorithms with batch data"""
        batch_size = 2
        max_time = 4
        num_classes = 5
        
        log_probs = torch.log_softmax(
            torch.randn(max_time, batch_size, num_classes, device=device), dim=-1
        )
        targets = torch.tensor([[1, 2], [3, 4]], device=device)
        input_lengths = torch.tensor([4, 3], device=device)
        target_lengths = torch.tensor([2, 2], device=device)
        
        # Forward
        log_likelihood = ctc_forward_algorithm(
            log_probs, targets, input_lengths, target_lengths, blank_id=0
        )
        assert log_likelihood.shape == (batch_size,)
        assert torch.isfinite(log_likelihood).all()
        
        # Backward
        log_beta = ctc_backward_algorithm(
            log_probs, targets, input_lengths, target_lengths, blank_id=0
        )
        assert log_beta.shape == (batch_size, max_time, 5)  # 2*2+1
        
        # Alignment
        paths = ctc_alignment_path(
            log_probs, targets, input_lengths, target_lengths, blank_id=0
        )
        assert len(paths) == batch_size


class TestCTCAligner:
    """Test CTCAligner class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def aligner(self, device):
        return CTCAligner(num_classes=10, blank_id=0).to(device)
    
    def test_ctc_aligner_init(self, aligner):
        """Test CTCAligner initialization"""
        assert aligner.num_classes == 10
        assert aligner.blank_id == 0
        assert aligner.reduction == 'mean'
    
    def test_ctc_aligner_forward(self, aligner, device):
        """Test CTCAligner forward pass"""
        batch_size = 2
        max_time = 5
        max_target_length = 3
        
        log_probs = torch.log_softmax(
            torch.randn(max_time, batch_size, 10, device=device), dim=-1
        )
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        input_lengths = torch.tensor([5, 4], device=device)
        target_lengths = torch.tensor([3, 3], device=device)
        
        loss = aligner(log_probs, targets, input_lengths, target_lengths)
        
        assert loss.shape == ()  # scalar
        assert torch.isfinite(loss)
        assert loss.item() >= 0  # CTC loss should be non-negative
    
    def test_ctc_aligner_decode_greedy(self, aligner, device):
        """Test greedy decoding"""
        max_time = 4
        batch_size = 1
        
        log_probs = torch.log_softmax(
            torch.randn(max_time, batch_size, 10, device=device), dim=-1
        )
        input_lengths = torch.tensor([4], device=device)
        
        decoded = aligner.decode(log_probs, input_lengths, beam_width=1)
        
        assert len(decoded) == batch_size
        assert decoded[0].dtype == torch.long
        assert decoded[0].device == device
    
    def test_ctc_aligner_decode_beam_search(self, aligner, device):
        """Test beam search decoding"""
        max_time = 4
        batch_size = 1
        
        log_probs = torch.log_softmax(
            torch.randn(max_time, batch_size, 10, device=device), dim=-1
        )
        input_lengths = torch.tensor([4], device=device)
        
        decoded = aligner.decode(log_probs, input_lengths, beam_width=3)
        
        assert len(decoded) == batch_size
        assert decoded[0].dtype == torch.long
        assert decoded[0].device == device
    
    def test_ctc_aligner_align(self, aligner, device):
        """Test alignment functionality"""
        batch_size = 1
        max_time = 5
        
        log_probs = torch.log_softmax(
            torch.randn(max_time, batch_size, 10, device=device), dim=-1
        )
        targets = torch.tensor([[1, 2]], device=device)
        input_lengths = torch.tensor([5], device=device)
        target_lengths = torch.tensor([2], device=device)
        
        alignments = aligner.align(log_probs, targets, input_lengths, target_lengths)
        
        assert len(alignments) == batch_size
        assert alignments[0].shape[0] == max_time
        assert torch.all(alignments[0] >= 0)


class TestCTCSegmentationAligner:
    """Test CTCSegmentationAligner class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def seg_aligner(self, device):
        return CTCSegmentationAligner(
            num_classes=10, 
            min_segment_length=10,
            max_segment_length=50,
            blank_id=0
        ).to(device)
    
    def test_segmentation_aligner_init(self, seg_aligner):
        """Test CTCSegmentationAligner initialization"""
        assert seg_aligner.num_classes == 10
        assert seg_aligner.min_segment_length == 10
        assert seg_aligner.max_segment_length == 50
        assert seg_aligner.blank_id == 0
    
    def test_segment_and_align(self, seg_aligner, device):
        """Test segmentation and alignment"""
        max_time = 30
        batch_size = 1
        
        log_probs = torch.log_softmax(
            torch.randn(max_time, batch_size, 10, device=device), dim=-1
        )
        full_transcript = torch.tensor([1, 2, 3, 4, 5], device=device)
        
        # Test with automatic boundary detection
        results = seg_aligner.segment_and_align(log_probs, full_transcript)
        
        assert isinstance(results, list)
        for segment_alignment, segment_text, start_frame, end_frame in results:
            assert isinstance(segment_alignment, torch.Tensor)
            assert isinstance(segment_text, torch.Tensor)
            assert isinstance(start_frame, int)
            assert isinstance(end_frame, int)
            assert start_frame < end_frame
            assert start_frame >= 0
            assert end_frame <= max_time
    
    def test_segment_with_boundaries(self, seg_aligner, device):
        """Test segmentation with provided boundaries"""
        max_time = 30
        batch_size = 1
        
        log_probs = torch.log_softmax(
            torch.randn(max_time, batch_size, 10, device=device), dim=-1
        )
        full_transcript = torch.tensor([1, 2, 3, 4, 5], device=device)
        segment_boundaries = torch.tensor([0, 15, 30], device=device)
        
        results = seg_aligner.segment_and_align(
            log_probs, full_transcript, segment_boundaries
        )
        
        assert isinstance(results, list)
        assert len(results) == 2  # 3 boundaries = 2 segments


class TestCTCEdgeCases:
    """Test CTC edge cases and error conditions"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_empty_target(self, device):
        """Test CTC with empty target sequence"""
        aligner = CTCAligner(num_classes=5, blank_id=0).to(device)
        
        log_probs = torch.log_softmax(
            torch.randn(3, 1, 5, device=device), dim=-1
        )
        targets = torch.tensor([[]], device=device, dtype=torch.long)
        input_lengths = torch.tensor([3], device=device)
        target_lengths = torch.tensor([0], device=device)
        
        loss = aligner(log_probs, targets, input_lengths, target_lengths)
        
        assert torch.isfinite(loss)
        assert loss.item() >= 0
    
    def test_single_token_target(self, device):
        """Test CTC with single token target"""
        aligner = CTCAligner(num_classes=5, blank_id=0).to(device)
        
        log_probs = torch.log_softmax(
            torch.randn(3, 1, 5, device=device), dim=-1
        )
        targets = torch.tensor([[1]], device=device)
        input_lengths = torch.tensor([3], device=device)
        target_lengths = torch.tensor([1], device=device)
        
        loss = aligner(log_probs, targets, input_lengths, target_lengths)
        
        assert torch.isfinite(loss)
        assert loss.item() >= 0
    
    def test_very_short_sequence(self, device):
        """Test CTC with very short input sequence"""
        aligner = CTCAligner(num_classes=5, blank_id=0).to(device)
        
        log_probs = torch.log_softmax(
            torch.randn(1, 1, 5, device=device), dim=-1
        )
        targets = torch.tensor([[1]], device=device)
        input_lengths = torch.tensor([1], device=device)
        target_lengths = torch.tensor([1], device=device)
        
        loss = aligner(log_probs, targets, input_lengths, target_lengths)
        
        assert torch.isfinite(loss)
    
    def test_different_blank_id(self, device):
        """Test CTC with non-zero blank ID"""
        blank_id = 4
        aligner = CTCAligner(num_classes=5, blank_id=blank_id).to(device)
        
        log_probs = torch.log_softmax(
            torch.randn(3, 1, 5, device=device), dim=-1
        )
        targets = torch.tensor([[1, 2]], device=device)
        input_lengths = torch.tensor([3], device=device)
        target_lengths = torch.tensor([2], device=device)
        
        loss = aligner(log_probs, targets, input_lengths, target_lengths)
        
        assert torch.isfinite(loss)
        assert loss.item() >= 0
        
        # Test utility functions with different blank ID
        expanded = expand_targets_with_blank(targets, blank_id)
        expected = torch.tensor([[4, 1, 4, 2, 4]], device=device)
        assert torch.equal(expanded, expected) 