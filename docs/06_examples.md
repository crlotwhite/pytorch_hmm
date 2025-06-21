# PyTorch HMM 예제 모음

이 문서는 PyTorch HMM 라이브러리의 다양한 사용 예제를 제공합니다. 각 예제는 실제 사용 시나리오에 기반하여 작성되었으며, 복사-붙여넣기로 바로 실행할 수 있습니다.

## 목차

1. [기본 예제](#1-기본-예제)
2. [음성 합성 예제](#2-음성-합성-예제)
3. [음성 인식 예제](#3-음성-인식-예제)
4. [실시간 처리 예제](#4-실시간-처리-예제)
5. [고급 모델 예제](#5-고급-모델-예제)
6. [평가 및 분석 예제](#6-평가-및-분석-예제)

---

## 1. 기본 예제

### 1.1 간단한 HMM 생성 및 추론

```python
import torch
from pytorch_hmm import HMMPyTorch, create_left_to_right_matrix

# HMM 설정
num_states = 5
transition_matrix = create_left_to_right_matrix(num_states, self_loop_prob=0.8)
hmm = HMMPyTorch(transition_matrix)

# 관측 데이터 (확률 형태)
batch_size, seq_len = 2, 50
observations = torch.softmax(torch.randn(batch_size, seq_len, num_states), dim=-1)

# Forward-backward 알고리즘
posteriors, forward, backward = hmm.forward_backward(observations)
print(f"Posterior 형태: {posteriors.shape}")

# Viterbi 디코딩
states, scores = hmm.viterbi_decode(observations)
print(f"최적 상태 시퀀스: {states[0, :10]}")
```

### 1.2 가우시안 관측 모델 HMM

```python
from pytorch_hmm import GaussianHMMLayer

# 가우시안 HMM 레이어
hmm_layer = GaussianHMMLayer(
    num_states=6,
    observation_dim=13,  # MFCC 차원
    transition_type="left_to_right"
)

# 연속 관측값 (MFCC 특징)
continuous_obs = torch.randn(batch_size, seq_len, 13)

# 추론
log_probs = hmm_layer(continuous_obs)
alignment = torch.argmax(log_probs, dim=-1)
print(f"정렬 결과: {alignment[0, :15]}")
```

---

## 2. 음성 합성 예제

### 2.1 기본 TTS 파이프라인

```python
import torch
import torch.nn as nn
from pytorch_hmm import HMMLayer, create_phoneme_aware_transitions

class BasicTTSModel(nn.Module):
    def __init__(self, num_phonemes=50, mel_dim=80):
        super().__init__()
        
        # 텍스트 인코더
        self.text_encoder = nn.Sequential(
            nn.Embedding(num_phonemes, 256),
            nn.LSTM(256, 128, batch_first=True),
        )
        
        # HMM 정렬
        self.hmm_layer = HMMLayer(
            num_states=num_phonemes * 3,  # 각 음소당 3상태
            learnable_transitions=True,
            transition_type="phoneme_aware"
        )
        
        # 음향 디코더
        self.acoustic_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, mel_dim)
        )
    
    def forward(self, phoneme_ids, mel_targets=None):
        # 텍스트 인코딩
        text_features, _ = self.text_encoder(phoneme_ids)
        
        # HMM 정렬
        if mel_targets is not None:
            # 훈련 시: 강제 정렬
            alignment_probs = self.hmm_layer(mel_targets)
            aligned_text = torch.matmul(alignment_probs, text_features)
        else:
            # 추론 시: 예측된 길이 사용
            seq_len = text_features.size(1) * 5  # 평균 확장 비율
            aligned_text = text_features.repeat_interleave(5, dim=1)[:, :seq_len]
        
        # 멜 스펙트로그램 생성
        mel_output = self.acoustic_decoder(aligned_text)
        return mel_output

# 모델 사용 예제
model = BasicTTSModel()
phoneme_seq = torch.randint(0, 50, (1, 20))  # 20개 음소
mel_spec = model(phoneme_seq)
print(f"생성된 멜 스펙트로그램: {mel_spec.shape}")
```

### 2.2 다중 화자 TTS

```python
from pytorch_hmm import AdaptiveTransitionMatrix

class MultiSpeakerTTS(nn.Module):
    def __init__(self, num_phonemes=50, num_speakers=10, mel_dim=80):
        super().__init__()
        
        # 화자 임베딩
        self.speaker_embedding = nn.Embedding(num_speakers, 64)
        
        # 적응적 전이 행렬
        self.adaptive_transitions = AdaptiveTransitionMatrix(
            num_states=num_phonemes * 3,
            context_dim=64
        )
        
        # 텍스트 인코더
        self.text_encoder = nn.LSTM(256 + 64, 128, batch_first=True)
        
        # HMM 레이어
        self.hmm_layer = HMMLayer(
            num_states=num_phonemes * 3,
            learnable_transitions=False  # 적응적 전이 사용
        )
        
        # 디코더
        self.decoder = nn.Linear(128, mel_dim)
    
    def forward(self, phoneme_ids, speaker_ids, mel_targets=None):
        batch_size, seq_len = phoneme_ids.shape
        
        # 화자 임베딩
        speaker_emb = self.speaker_embedding(speaker_ids)  # (B, 64)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 텍스트 + 화자 정보 결합
        phoneme_emb = nn.functional.embedding(phoneme_ids, 
                                            torch.randn(50, 256))
        text_input = torch.cat([phoneme_emb, speaker_emb], dim=-1)
        
        # 텍스트 인코딩
        text_features, _ = self.text_encoder(text_input)
        
        # 화자별 전이 행렬 생성
        speaker_context = speaker_emb[:, 0]  # (B, 64)
        transition_matrices = self.adaptive_transitions(speaker_context)
        
        # HMM 정렬 (화자별 전이 행렬 사용)
        self.hmm_layer.set_transition_matrix(transition_matrices)
        
        if mel_targets is not None:
            alignment_probs = self.hmm_layer(mel_targets)
            aligned_features = torch.matmul(alignment_probs, text_features)
        else:
            # 추론 로직
            aligned_features = text_features.repeat_interleave(4, dim=1)
        
        # 멜 스펙트로그램 생성
        mel_output = self.decoder(aligned_features)
        return mel_output

# 사용 예제
model = MultiSpeakerTTS()
phonemes = torch.randint(0, 50, (2, 15))
speakers = torch.tensor([0, 1])
output = model(phonemes, speakers)
print(f"다중 화자 TTS 출력: {output.shape}")
```

---

## 3. 음성 인식 예제

### 3.1 CTC 기반 ASR

```python
from pytorch_hmm import CTCAligner
import torch.nn.functional as F

class CTCASRModel(nn.Module):
    def __init__(self, input_dim=80, vocab_size=29):  # 26 letters + space + blank + eos
        super().__init__()
        
        # 음향 인코더
        self.encoder = nn.Sequential(
            nn.LSTM(input_dim, 256, num_layers=3, batch_first=True, 
                   dropout=0.1, bidirectional=True),
        )
        
        # CTC 출력층
        self.classifier = nn.Linear(512, vocab_size)  # bidirectional이므로 512
        
        # CTC 정렬기
        self.ctc_aligner = CTCAligner(vocab_size, blank_id=0)
    
    def forward(self, audio_features, targets=None, input_lengths=None, target_lengths=None):
        # 음향 인코딩
        encoder_out, _ = self.encoder(audio_features)
        
        # CTC 로그 확률
        log_probs = F.log_softmax(self.classifier(encoder_out), dim=-1)
        
        if targets is not None:
            # 훈련 시: CTC 손실 계산
            # log_probs: (T, B, V), targets: (B, S)
            log_probs_t = log_probs.transpose(0, 1)  # (T, B, V)
            loss = self.ctc_aligner(log_probs_t, targets, input_lengths, target_lengths)
            return loss, log_probs
        else:
            # 추론 시: 디코딩
            decoded = self.ctc_aligner.decode(log_probs.transpose(0, 1), input_lengths)
            return decoded

# 사용 예제
model = CTCASRModel()

# 훈련 데이터
audio = torch.randn(2, 100, 80)  # (batch, time, features)
targets = torch.tensor([[8, 5, 12, 12, 15], [23, 15, 18, 12, 4]])  # "HELLO", "WORLD"
input_lengths = torch.tensor([100, 100])
target_lengths = torch.tensor([5, 5])

# 훈련
loss, log_probs = model(audio, targets, input_lengths, target_lengths)
print(f"CTC 손실: {loss.item():.4f}")

# 추론
with torch.no_grad():
    decoded_sequences = model(audio, input_lengths=input_lengths)
    for i, seq in enumerate(decoded_sequences):
        chars = [chr(ord('A') + idx - 1) if idx > 0 else '_' for idx in seq.tolist()]
        print(f"디코딩 결과 {i}: {''.join(chars)}")
```

### 3.2 End-to-End ASR with Attention

```python
from pytorch_hmm import DTWAligner

class AttentionASRModel(nn.Module):
    def __init__(self, input_dim=80, vocab_size=29, hidden_dim=256):
        super().__init__()
        
        # 인코더
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                              batch_first=True, bidirectional=True)
        
        # 디코더
        self.decoder = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        
        # 어텐션
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 출력층
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # DTW 정렬 (후처리용)
        self.dtw_aligner = DTWAligner(distance_fn='cosine')
    
    def forward(self, audio_features, target_sequence=None):
        batch_size, seq_len, _ = audio_features.shape
        
        # 음향 인코딩
        encoder_out, _ = self.encoder(audio_features)  # (B, T, 2*hidden_dim)
        
        if target_sequence is not None:
            # 훈련 시: Teacher forcing
            target_embed = F.one_hot(target_sequence, num_classes=29).float()
            decoder_out, _ = self.decoder(target_embed)
            
            # 어텐션
            attended, attention_weights = self.attention(
                decoder_out, encoder_out, encoder_out
            )
            
            # 출력
            output = self.output_projection(attended)
            return output, attention_weights
        else:
            # 추론 시: Autoregressive 디코딩
            outputs = []
            hidden = None
            current_input = torch.zeros(batch_size, 1, 29)  # Start token
            
            for _ in range(seq_len):  # 최대 길이
                decoder_out, hidden = self.decoder(current_input, hidden)
                attended, _ = self.attention(decoder_out, encoder_out, encoder_out)
                output = self.output_projection(attended)
                
                outputs.append(output)
                current_input = F.one_hot(output.argmax(dim=-1), num_classes=29).float()
                
                # 종료 조건 (EOS 토큰)
                if output.argmax(dim=-1).item() == 28:  # EOS
                    break
            
            return torch.cat(outputs, dim=1)

# 사용 예제
model = AttentionASRModel()
audio = torch.randn(1, 80, 80)
target = torch.randint(0, 28, (1, 20))

# 훈련
output, attention = model(audio, target)
print(f"출력 형태: {output.shape}, 어텐션 형태: {attention.shape}")

# 추론
with torch.no_grad():
    predicted = model(audio)
    print(f"예측 시퀀스 형태: {predicted.shape}")
```

---

## 4. 실시간 처리 예제

### 4.1 스트리밍 HMM 처리

```python
from pytorch_hmm import StreamingHMMProcessor, AdaptiveLatencyController

class RealTimeASR:
    def __init__(self, model, chunk_size=160, overlap=80):
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # 스트리밍 HMM 프로세서
        self.streaming_processor = StreamingHMMProcessor(
            hmm_model=model,
            chunk_size=chunk_size,
            overlap_size=overlap,
            max_delay_frames=5
        )
        
        # 적응적 지연 제어
        self.latency_controller = AdaptiveLatencyController(
            target_latency_ms=200,
            max_latency_ms=500
        )
        
        self.buffer = []
        self.partial_results = []
    
    def process_chunk(self, audio_chunk):
        """실시간 오디오 청크 처리"""
        # 청크를 버퍼에 추가
        self.buffer.extend(audio_chunk)
        
        # 충분한 데이터가 쌓이면 처리
        if len(self.buffer) >= self.chunk_size:
            # 처리할 청크 추출
            process_chunk = torch.tensor(self.buffer[:self.chunk_size]).unsqueeze(0)
            
            # 스트리밍 처리
            result = self.streaming_processor.process_chunk(process_chunk)
            
            if result.is_complete:
                # 완전한 결과
                decoded_text = self.decode_result(result.states)
                self.partial_results.append(decoded_text)
                
                # 지연 시간 조정
                self.latency_controller.update_latency(result.processing_time)
                
                return decoded_text, True  # (결과, 완료 여부)
            else:
                # 부분 결과
                partial_text = self.decode_result(result.states, partial=True)
                return partial_text, False
            
            # 버퍼 업데이트 (overlap 고려)
            self.buffer = self.buffer[self.chunk_size - self.overlap:]
        
        return None, False
    
    def decode_result(self, states, partial=False):
        """상태 시퀀스를 텍스트로 변환"""
        # 간단한 디코딩 (실제로는 더 복잡한 후처리 필요)
        chars = [chr(ord('A') + s % 26) for s in states.tolist()]
        text = ''.join(chars)
        
        if partial:
            text += "..."
        
        return text
    
    def reset(self):
        """스트림 리셋"""
        self.buffer = []
        self.partial_results = []
        self.streaming_processor.reset()

# 사용 예제
class DummyASRModel(nn.Module):
    def forward(self, x):
        return torch.randint(0, 26, (x.size(0), x.size(1)))

real_time_asr = RealTimeASR(DummyASRModel())

# 실시간 처리 시뮬레이션
print("실시간 ASR 시뮬레이션:")
for i in range(10):
    # 새 오디오 청크 (실제로는 마이크에서 입력)
    audio_chunk = torch.randn(80).tolist()
    
    result, is_complete = real_time_asr.process_chunk(audio_chunk)
    
    if result:
        status = "완료" if is_complete else "부분"
        print(f"청크 {i}: [{status}] {result}")
```

### 4.2 저지연 음성 합성

```python
from pytorch_hmm import StreamingHMMProcessor

class LowLatencyTTS:
    def __init__(self, model, target_latency_ms=100):
        self.model = model
        self.target_latency_ms = target_latency_ms
        
        # 스트리밍 설정
        self.chunk_size = 10  # 작은 청크로 저지연 달성
        self.lookahead_frames = 5
        
        self.streaming_processor = StreamingHMMProcessor(
            hmm_model=model,
            chunk_size=self.chunk_size,
            overlap_size=2,
            max_delay_frames=2  # 매우 낮은 지연
        )
    
    def synthesize_streaming(self, phoneme_sequence):
        """스트리밍 음성 합성"""
        results = []
        
        # 음소 시퀀스를 청크로 분할
        for i in range(0, len(phoneme_sequence), self.chunk_size):
            chunk = phoneme_sequence[i:i + self.chunk_size]
            
            # 룩어헤드 추가 (자연스러운 합성을 위해)
            if i + self.chunk_size < len(phoneme_sequence):
                lookahead = phoneme_sequence[i + self.chunk_size:
                                           i + self.chunk_size + self.lookahead_frames]
                extended_chunk = torch.cat([chunk, lookahead])
            else:
                extended_chunk = chunk
            
            # 청크 처리
            start_time = time.time()
            mel_chunk = self.model(extended_chunk.unsqueeze(0))
            
            # 룩어헤드 제거
            if len(lookahead) > 0:
                mel_chunk = mel_chunk[:, :len(chunk) * 4]  # 평균 확장 비율 4
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            results.append({
                'mel_spectrogram': mel_chunk,
                'processing_time_ms': processing_time,
                'chunk_id': i // self.chunk_size
            })
            
            # 지연 시간 체크
            if processing_time > self.target_latency_ms:
                print(f"경고: 청크 {i//self.chunk_size} 처리 시간 초과 "
                      f"({processing_time:.1f}ms > {self.target_latency_ms}ms)")
        
        return results
    
    def get_latency_stats(self, results):
        """지연 시간 통계"""
        processing_times = [r['processing_time_ms'] for r in results]
        
        return {
            'mean_latency_ms': np.mean(processing_times),
            'max_latency_ms': np.max(processing_times),
            'min_latency_ms': np.min(processing_times),
            'target_latency_ms': self.target_latency_ms,
            'latency_violations': sum(1 for t in processing_times 
                                    if t > self.target_latency_ms)
        }

# 사용 예제
import time
import numpy as np

class DummyTTSModel(nn.Module):
    def forward(self, phonemes):
        time.sleep(0.05)  # 처리 시간 시뮬레이션
        return torch.randn(phonemes.size(0), phonemes.size(1) * 4, 80)

low_latency_tts = LowLatencyTTS(DummyTTSModel())

# 테스트 음소 시퀀스
phonemes = torch.randint(0, 50, (50,))  # 50개 음소

print("저지연 TTS 테스트:")
results = low_latency_tts.synthesize_streaming(phonemes)

# 통계 출력
stats = low_latency_tts.get_latency_stats(results)
print(f"평균 지연: {stats['mean_latency_ms']:.1f}ms")
print(f"최대 지연: {stats['max_latency_ms']:.1f}ms")
print(f"지연 위반: {stats['latency_violations']}/{len(results)} 청크")
```

---

## 5. 고급 모델 예제

### 5.1 Semi-Markov HMM (HSMM)

```python
from pytorch_hmm import SemiMarkovHMM, DurationModel

class PhonemeDurationTTS(nn.Module):
    def __init__(self, num_phonemes=50, mel_dim=80):
        super().__init__()
        
        # 지속시간 모델
        self.duration_model = DurationModel(
            num_states=num_phonemes,
            max_duration=20,
            distribution_type='gamma'  # 감마 분포로 지속시간 모델링
        )
        
        # Semi-Markov HMM
        self.hsmm = SemiMarkovHMM(
            num_states=num_phonemes,
            observation_dim=mel_dim,
            duration_model=self.duration_model,
            observation_type='mixture_gaussian'
        )
        
        # 텍스트 인코더
        self.text_encoder = nn.Sequential(
            nn.Embedding(num_phonemes, 128),
            nn.LSTM(128, 256, batch_first=True)
        )
    
    def forward(self, phoneme_ids, mel_targets=None, durations=None):
        # 텍스트 인코딩
        text_features, _ = self.text_encoder(phoneme_ids)
        
        if mel_targets is not None and durations is not None:
            # 훈련 시: 지속시간 정보 사용
            # 지속시간 모델 훈련
            duration_loss = self.duration_model.compute_loss(phoneme_ids, durations)
            
            # HSMM 정렬
            alignment_probs, state_durations = self.hsmm.forward_backward_with_duration(
                mel_targets, phoneme_ids, durations
            )
            
            # 정렬된 텍스트 특징
            aligned_features = torch.matmul(alignment_probs, text_features)
            
            return aligned_features, duration_loss, state_durations
        else:
            # 추론 시: 지속시간 예측
            predicted_durations = self.duration_model.predict_durations(phoneme_ids)
            
            # 예측된 지속시간으로 확장
            expanded_features = []
            for i, duration in enumerate(predicted_durations[0]):
                phoneme_feature = text_features[0, i:i+1]  # (1, feature_dim)
                expanded = phoneme_feature.repeat(int(duration.item()), 1)
                expanded_features.append(expanded)
            
            aligned_features = torch.cat(expanded_features, dim=0).unsqueeze(0)
            return aligned_features, predicted_durations

# 사용 예제
model = PhonemeDurationTTS()

# 훈련 데이터
phonemes = torch.randint(0, 50, (1, 10))
mel_spec = torch.randn(1, 80, 80)  # (batch, time, mel_dim)
durations = torch.randint(5, 15, (1, 10))  # 각 음소의 지속시간

# 훈련
aligned_features, duration_loss, predicted_durations = model(
    phonemes, mel_spec, durations
)
print(f"정렬된 특징: {aligned_features.shape}")
print(f"지속시간 손실: {duration_loss.item():.4f}")

# 추론
with torch.no_grad():
    synthesized_features, pred_durations = model(phonemes)
    print(f"합성된 특징: {synthesized_features.shape}")
    print(f"예측 지속시간: {pred_durations[0][:5]}")
```

### 5.2 Neural HMM with Contextual Information

```python
from pytorch_hmm import ContextualNeuralHMM

class ContextualTTS(nn.Module):
    def __init__(self, num_phonemes=50, mel_dim=80):
        super().__init__()
        
        # 컨텍스트 인코더들
        self.linguistic_encoder = nn.Linear(20, 64)  # 언어적 특징
        self.prosodic_encoder = nn.Linear(10, 32)    # 운율 특징
        self.speaker_encoder = nn.Embedding(100, 32)  # 화자 특징
        
        # Contextual Neural HMM
        self.contextual_hmm = ContextualNeuralHMM(
            num_states=num_phonemes * 3,
            observation_dim=mel_dim,
            linguistic_context_dim=64,
            prosodic_context_dim=32,
            speaker_context_dim=32,
            hidden_dim=256
        )
        
        # 음향 디코더
        self.acoustic_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, mel_dim)
        )
    
    def forward(self, phoneme_ids, linguistic_features, prosodic_features, 
                speaker_ids, mel_targets=None):
        batch_size, seq_len = phoneme_ids.shape
        
        # 컨텍스트 인코딩
        ling_context = self.linguistic_encoder(linguistic_features)
        pros_context = self.prosodic_encoder(prosodic_features)
        spk_context = self.speaker_encoder(speaker_ids).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Contextual HMM 추론
        if mel_targets is not None:
            # 훈련 시
            posteriors, context_weights = self.contextual_hmm(
                observations=mel_targets,
                linguistic_context=ling_context,
                prosodic_context=pros_context,
                speaker_context=spk_context
            )
            
            # 컨텍스트 가중 특징
            weighted_features = (
                context_weights['linguistic'].unsqueeze(-1) * ling_context +
                context_weights['prosodic'].unsqueeze(-1) * pros_context +
                context_weights['speaker'].unsqueeze(-1) * spk_context
            )
            
            # 음향 특징 생성
            mel_output = self.acoustic_decoder(weighted_features)
            
            return mel_output, posteriors, context_weights
        else:
            # 추론 시 - 예측된 길이로 확장
            expanded_seq_len = seq_len * 5  # 평균 확장 비율
            
            # 컨텍스트 확장
            ling_expanded = ling_context.repeat_interleave(5, dim=1)[:, :expanded_seq_len]
            pros_expanded = pros_context.repeat_interleave(5, dim=1)[:, :expanded_seq_len]
            spk_expanded = spk_context.repeat_interleave(5, dim=1)[:, :expanded_seq_len]
            
            # 기본 가중치 사용 (훈련된 평균값)
            combined_features = (
                0.4 * ling_expanded + 
                0.3 * pros_expanded + 
                0.3 * spk_expanded
            )
            
            mel_output = self.acoustic_decoder(combined_features)
            return mel_output

# 사용 예제
model = ContextualTTS()

# 입력 데이터
phonemes = torch.randint(0, 50, (2, 15))
linguistic_feat = torch.randn(2, 15, 20)  # 언어적 특징
prosodic_feat = torch.randn(2, 15, 10)    # 운율 특징
speaker_ids = torch.tensor([0, 1])
mel_targets = torch.randn(2, 75, 80)      # 타겟 멜 스펙트로그램

# 훈련
mel_output, posteriors, context_weights = model(
    phonemes, linguistic_feat, prosodic_feat, speaker_ids, mel_targets
)
print(f"멜 출력: {mel_output.shape}")
print(f"언어적 가중치 평균: {context_weights['linguistic'].mean():.3f}")
print(f"운율 가중치 평균: {context_weights['prosodic'].mean():.3f}")

# 추론
with torch.no_grad():
    synthesized_mel = model(phonemes, linguistic_feat, prosodic_feat, speaker_ids)
    print(f"합성된 멜: {synthesized_mel.shape}")
```

---

## 6. 평가 및 분석 예제

### 6.1 종합적인 음성 품질 평가

```python
from pytorch_hmm import (
    comprehensive_speech_evaluation,
    print_evaluation_summary,
    save_evaluation_results
)

def evaluate_tts_system(model, test_dataset):
    """TTS 시스템의 종합적인 평가"""
    
    results = {
        'mcd_scores': [],
        'f0_rmse_scores': [],
        'alignment_accuracies': [],
        'processing_times': []
    }
    
    model.eval()
    with torch.no_grad():
        for i, (phonemes, mel_target, f0_target, alignment_target) in enumerate(test_dataset):
            start_time = time.time()
            
            # 음성 합성
            mel_pred = model(phonemes)
            
            processing_time = time.time() - start_time
            results['processing_times'].append(processing_time)
            
            # 종합 평가
            evaluation = comprehensive_speech_evaluation(
                predicted_mel=mel_pred,
                target_mel=mel_target,
                predicted_f0=None,  # F0는 별도 모델에서 추출
                target_f0=f0_target,
                predicted_alignment=model.get_alignment(),
                target_alignment=alignment_target
            )
            
            results['mcd_scores'].append(evaluation['mel_cepstral_distortion'])
            results['f0_rmse_scores'].append(evaluation['f0_rmse'])
            results['alignment_accuracies'].append(evaluation['alignment_accuracy'])
            
            # 진행상황 출력
            if (i + 1) % 10 == 0:
                print(f"평가 진행: {i+1}/{len(test_dataset)} 샘플 완료")
    
    # 최종 통계
    final_results = {
        'mean_mcd': np.mean(results['mcd_scores']),
        'std_mcd': np.std(results['mcd_scores']),
        'mean_f0_rmse': np.mean(results['f0_rmse_scores']),
        'std_f0_rmse': np.std(results['f0_rmse_scores']),
        'mean_alignment_acc': np.mean(results['alignment_accuracies']),
        'std_alignment_acc': np.std(results['alignment_accuracies']),
        'mean_processing_time': np.mean(results['processing_times']),
        'real_time_factor': np.mean(results['processing_times']) / 0.1  # 100ms 오디오 기준
    }
    
    # 결과 출력
    print_evaluation_summary(final_results)
    
    # 결과 저장
    save_evaluation_results(final_results, 'tts_evaluation_results.json')
    
    return final_results

# 더미 데이터셋으로 예제 실행
class DummyTestDataset:
    def __init__(self, size=50):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return (
            torch.randint(0, 50, (20,)),      # phonemes
            torch.randn(80, 80),              # mel_target
            torch.randn(80),                  # f0_target
            torch.randint(0, 50, (80,))       # alignment_target
        )

# 평가 실행
class DummyTTSModel(nn.Module):
    def forward(self, phonemes):
        return torch.randn(phonemes.size(0) * 4, 80)
    
    def get_alignment(self):
        return torch.randint(0, 50, (80,))

test_dataset = DummyTestDataset()
model = DummyTTSModel()

print("TTS 시스템 종합 평가 시작...")
evaluation_results = evaluate_tts_system(model, test_dataset)
```

### 6.2 HMM 정렬 품질 분석

```python
from pytorch_hmm import alignment_accuracy, boundary_accuracy, duration_accuracy
import matplotlib.pyplot as plt

def analyze_alignment_quality(hmm_model, test_data):
    """HMM 정렬 품질 상세 분석"""
    
    alignment_metrics = {
        'frame_accuracies': [],
        'boundary_accuracies': [],
        'duration_accuracies': [],
        'state_distributions': [],
        'transition_patterns': []
    }
    
    for phonemes, mel_features, true_alignment in test_data:
        # HMM 정렬 수행
        predicted_alignment = hmm_model.align(mel_features, phonemes)
        
        # 프레임 단위 정확도
        frame_acc = alignment_accuracy(predicted_alignment, true_alignment)
        alignment_metrics['frame_accuracies'].append(frame_acc)
        
        # 경계 정확도 (±20ms 허용)
        boundary_acc = boundary_accuracy(
            predicted_alignment, true_alignment, 
            tolerance_frames=2  # 20ms at 10ms frame rate
        )
        alignment_metrics['boundary_accuracies'].append(boundary_acc)
        
        # 지속시간 정확도
        duration_acc = duration_accuracy(predicted_alignment, true_alignment)
        alignment_metrics['duration_accuracies'].append(duration_acc)
        
        # 상태 분포 분석
        state_dist = torch.bincount(predicted_alignment, minlength=hmm_model.num_states)
        alignment_metrics['state_distributions'].append(state_dist)
        
        # 전이 패턴 분석
        transitions = []
        for i in range(len(predicted_alignment) - 1):
            if predicted_alignment[i] != predicted_alignment[i + 1]:
                transitions.append((predicted_alignment[i].item(), 
                                 predicted_alignment[i + 1].item()))
        alignment_metrics['transition_patterns'].extend(transitions)
    
    # 통계 계산
    stats = {
        'mean_frame_accuracy': np.mean(alignment_metrics['frame_accuracies']),
        'mean_boundary_accuracy': np.mean(alignment_metrics['boundary_accuracies']),
        'mean_duration_accuracy': np.mean(alignment_metrics['duration_accuracies']),
        'std_frame_accuracy': np.std(alignment_metrics['frame_accuracies']),
        'std_boundary_accuracy': np.std(alignment_metrics['boundary_accuracies']),
        'std_duration_accuracy': np.std(alignment_metrics['duration_accuracies'])
    }
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 정확도 분포
    axes[0, 0].hist(alignment_metrics['frame_accuracies'], bins=20, alpha=0.7, label='Frame')
    axes[0, 0].hist(alignment_metrics['boundary_accuracies'], bins=20, alpha=0.7, label='Boundary')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Alignment Accuracy Distribution')
    axes[0, 0].legend()
    
    # 상태 사용 빈도
    avg_state_dist = torch.stack(alignment_metrics['state_distributions']).float().mean(0)
    axes[0, 1].bar(range(len(avg_state_dist)), avg_state_dist.numpy())
    axes[0, 1].set_xlabel('State ID')
    axes[0, 1].set_ylabel('Average Usage')
    axes[0, 1].set_title('State Usage Distribution')
    
    # 전이 패턴 히트맵
    transition_matrix = torch.zeros(hmm_model.num_states, hmm_model.num_states)
    for from_state, to_state in alignment_metrics['transition_patterns']:
        transition_matrix[from_state, to_state] += 1
    
    im = axes[1, 0].imshow(transition_matrix.numpy(), cmap='Blues')
    axes[1, 0].set_xlabel('To State')
    axes[1, 0].set_ylabel('From State')
    axes[1, 0].set_title('Transition Pattern Heatmap')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 정확도 vs 시퀀스 길이
    seq_lengths = [len(alignment) for _, _, alignment in test_data]
    axes[1, 1].scatter(seq_lengths, alignment_metrics['frame_accuracies'], alpha=0.6)
    axes[1, 1].set_xlabel('Sequence Length')
    axes[1, 1].set_ylabel('Frame Accuracy')
    axes[1, 1].set_title('Accuracy vs Sequence Length')
    
    plt.tight_layout()
    plt.savefig('alignment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats, alignment_metrics

# 사용 예제
class DummyHMMModel:
    def __init__(self):
        self.num_states = 50
    
    def align(self, mel_features, phonemes):
        # 더미 정렬 결과
        seq_len = mel_features.size(0)
        return torch.randint(0, self.num_states, (seq_len,))

# 더미 테스트 데이터
test_data = [
    (torch.randint(0, 50, (10,)), torch.randn(40, 80), torch.randint(0, 50, (40,)))
    for _ in range(20)
]

hmm_model = DummyHMMModel()
stats, metrics = analyze_alignment_quality(hmm_model, test_data)

print("정렬 품질 분석 결과:")
for key, value in stats.items():
    print(f"{key}: {value:.4f}")
```

---

## 마무리

이 예제 모음은 PyTorch HMM 라이브러리의 다양한 활용 방법을 보여줍니다. 각 예제는 실제 프로젝트에서 바로 사용할 수 있도록 작성되었으며, 필요에 따라 수정하여 사용하시기 바랍니다.

### 추가 학습 자료

- [기초 이론](01_hmm_theory.md): HMM의 수학적 배경
- [기본 사용법](02_basic_usage.md): 라이브러리 기본 사용법
- [고급 기능](03_advanced_features.md): Neural HMM, Semi-Markov HMM 등
- [실제 응용](04_real_world_applications.md): TTS, ASR 시스템 구축
- [성능 최적화](05_performance_optimization.md): 메모리 및 속도 최적화

### 지원 및 기여

- GitHub Issues: 버그 리포트 및 기능 요청
- Pull Requests: 코드 기여 환영
- Documentation: 문서 개선 제안 