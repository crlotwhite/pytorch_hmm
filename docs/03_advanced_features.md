# PyTorch HMM 라이브러리 고급 기능

이 문서에서는 PyTorch HMM 라이브러리의 고급 기능들을 다룹니다. 기본적인 HMM 사용법을 이해한 후에 이 문서를 읽어보시기 바랍니다.

## 목차

1. [Neural HMM](#neural-hmm)
2. [Semi-Markov HMM (HSMM)](#semi-markov-hmm-hsmm)
3. [정렬 알고리즘](#정렬-알고리즘)
   - [Dynamic Time Warping (DTW)](#dynamic-time-warping-dtw)
   - [Connectionist Temporal Classification (CTC)](#connectionist-temporal-classification-ctc)
4. [고급 평가 메트릭](#고급-평가-메트릭)
5. [성능 최적화 기법](#성능-최적화-기법)
6. [실제 응용 예제](#실제-응용-예제)

## Neural HMM

Neural HMM은 전통적인 HMM에 신경망을 결합하여 더 복잡한 패턴을 학습할 수 있게 합니다.

### 기본 Neural HMM

```python
import torch
from pytorch_hmm import NeuralHMM

# Neural HMM 초기화
num_states = 10
observation_dim = 80  # 멜 스펙트로그램 차원
context_dim = 64      # 컨텍스트 특징 차원

neural_hmm = NeuralHMM(
    num_states=num_states,
    observation_dim=observation_dim,
    context_dim=context_dim,
    hidden_dim=128,
    transition_type='rnn',      # RNN 기반 전이 모델
    observation_type='mixture'  # Mixture Gaussian 관측 모델
)

# 입력 데이터 준비
batch_size, seq_length = 2, 200
context = torch.randn(batch_size, seq_length, context_dim)
observations = torch.randn(batch_size, seq_length, observation_dim)

# 추론 수행
posteriors, forward, backward = neural_hmm(observations, context)
print(f"State posteriors shape: {posteriors.shape}")
```

### Contextual Neural HMM

컨텍스트 정보를 더 효과적으로 활용하는 향상된 버전입니다.

```python
from pytorch_hmm import ContextualNeuralHMM

# Contextual Neural HMM 초기화
contextual_hmm = ContextualNeuralHMM(
    num_states=num_states,
    observation_dim=observation_dim,
    context_dim=context_dim,
    hidden_dim=128,
    num_context_layers=2,       # 컨텍스트 처리 레이어 수
    attention_dim=64,           # 어텐션 메커니즘 차원
    use_attention=True          # 어텐션 사용 여부
)

# 언어적 특징 (음소 정보, 운율 등)
linguistic_features = torch.randn(batch_size, seq_length, 32)
prosodic_features = torch.randn(batch_size, seq_length, 16)
speaker_embedding = torch.randn(batch_size, 16)

# 컨텍스트 결합
context = torch.cat([linguistic_features, prosodic_features], dim=-1)

# 화자 임베딩을 포함한 추론
posteriors = contextual_hmm(
    observations, 
    context, 
    speaker_embedding=speaker_embedding
)
```

### Neural HMM 학습

```python
import torch.optim as optim
from torch.nn import MSELoss

# 옵티마이저 설정
optimizer = optim.Adam(neural_hmm.parameters(), lr=0.001)
criterion = MSELoss()

# 학습 루프
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    posteriors, _, _ = neural_hmm(observations, context)
    
    # 타겟 상태 시퀀스 (예: 강제 정렬로부터)
    target_states = torch.randint(0, num_states, (batch_size, seq_length))
    target_posteriors = torch.zeros_like(posteriors)
    target_posteriors.scatter_(2, target_states.unsqueeze(2), 1.0)
    
    # Loss 계산
    loss = criterion(posteriors, target_posteriors)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Semi-Markov HMM (HSMM)

Semi-Markov HMM은 각 상태의 지속 시간을 명시적으로 모델링합니다.

### 기본 HSMM

```python
from pytorch_hmm import SemiMarkovHMM, DurationModel

# 지속 시간 모델 정의
duration_model = DurationModel(
    num_states=num_states,
    distribution_type='gamma',  # 감마 분포 사용
    max_duration=50            # 최대 지속 시간
)

# Semi-Markov HMM 초기화
hsmm = SemiMarkovHMM(
    num_states=num_states,
    observation_dim=observation_dim,
    duration_model=duration_model,
    transition_type='standard'  # 표준 전이 행렬
)

# 관측 시퀀스
observations = torch.randn(batch_size, seq_length, observation_dim)

# Forward-backward 알고리즘
log_likelihood, posteriors = hsmm.forward_backward(observations)
print(f"Log-likelihood: {log_likelihood.item():.4f}")

# Viterbi 디코딩 (최적 상태 시퀀스와 지속 시간)
best_path, best_durations = hsmm.viterbi_decode(observations)
print(f"Best path: {best_path}")
print(f"State durations: {best_durations}")
```

### 지속 시간 모델 사용자 정의

```python
import torch.nn as nn

class CustomDurationModel(nn.Module):
    def __init__(self, num_states, hidden_dim=64):
        super().__init__()
        self.num_states = num_states
        self.duration_net = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_id, duration):
        # 상태별 지속 시간 확률 계산
        state_one_hot = torch.zeros(self.num_states)
        state_one_hot[state_id] = 1.0
        
        # 신경망을 통한 지속 시간 확률 예측
        log_prob = self.duration_net(state_one_hot.unsqueeze(0))
        return log_prob.squeeze()

# 사용자 정의 지속 시간 모델 사용
custom_duration = CustomDurationModel(num_states)
hsmm_custom = SemiMarkovHMM(
    num_states=num_states,
    observation_dim=observation_dim,
    duration_model=custom_duration
)
```

## 정렬 알고리즘

### Dynamic Time Warping (DTW)

DTW는 두 시계열 간의 최적 정렬을 찾는 알고리즘입니다.

```python
from pytorch_hmm import DTWAligner

# DTW 정렬기 초기화
dtw_aligner = DTWAligner(
    distance_fn='cosine',        # 거리 함수: 'euclidean', 'cosine', 'manhattan'
    step_pattern='symmetric',    # 스텝 패턴: 'symmetric', 'asymmetric'
    window_type='sakoe_chiba',   # 윈도우 타입: None, 'sakoe_chiba', 'itakura'
    window_size=10               # 윈도우 크기
)

# 음소 특징과 음성 특징
phoneme_features = torch.randn(5, 12)   # 5개 음소, 12차원 특징
audio_features = torch.randn(100, 12)   # 100 프레임, 12차원 특징

# DTW 정렬 수행
path_i, path_j, total_cost = dtw_aligner(phoneme_features, audio_features)

print(f"Alignment path length: {len(path_i)}")
print(f"Total DTW cost: {total_cost:.4f}")

# 음소 경계 추출
def extract_phoneme_boundaries(path_i, path_j):
    boundaries = []
    current_phoneme = path_i[0].item()
    current_start = 0
    
    for i, phoneme_idx in enumerate(path_i):
        if phoneme_idx != current_phoneme:
            boundaries.append((current_phoneme, current_start, path_j[i-1].item()))
            current_phoneme = phoneme_idx.item()
            current_start = path_j[i].item()
    
    # 마지막 음소
    boundaries.append((current_phoneme, current_start, path_j[-1].item()))
    return boundaries

boundaries = extract_phoneme_boundaries(path_i, path_j)
for phoneme_id, start, end in boundaries:
    print(f"Phoneme {phoneme_id}: frames {start}-{end}")
```

### Connectionist Temporal Classification (CTC)

CTC는 입력과 출력 시퀀스의 길이가 다른 경우에 사용되는 정렬 알고리즘입니다.

```python
from pytorch_hmm import CTCAligner

# CTC 정렬기 초기화
vocab_size = 28  # 26개 문자 + blank + space
ctc_aligner = CTCAligner(
    num_classes=vocab_size,
    blank_id=0,              # blank 토큰 ID
    reduction='mean'         # loss reduction 방법
)

# 음향 모델 출력 (로그 확률)
sequence_length, batch_size = 80, 2
log_probs = torch.log_softmax(
    torch.randn(sequence_length, batch_size, vocab_size), 
    dim=-1
)

# 타겟 텍스트
targets = torch.tensor([
    [8, 5, 12, 12, 15],   # "HELLO"
    [23, 15, 18, 12, 4]   # "WORLD"
])

input_lengths = torch.full((batch_size,), sequence_length)
target_lengths = torch.tensor([5, 5])

# CTC Loss 계산
ctc_loss = ctc_aligner(log_probs, targets, input_lengths, target_lengths)
print(f"CTC Loss: {ctc_loss.item():.4f}")

# Greedy 디코딩
decoded_sequences = ctc_aligner.decode(log_probs, input_lengths)
print("Decoded sequences:", decoded_sequences)

# 강제 정렬
alignments = ctc_aligner.align(log_probs, targets, input_lengths, target_lengths)
print("Forced alignments:", [len(align) for align in alignments])
```

### Beam Search 디코딩

```python
# Beam search 디코딩 (더 정확한 결과)
beam_width = 5
beam_decoded = ctc_aligner.beam_search_decode(
    log_probs, 
    input_lengths, 
    beam_width=beam_width
)

print("Beam search results:")
for i, candidates in enumerate(beam_decoded):
    print(f"Sequence {i}:")
    for j, (sequence, score) in enumerate(candidates[:3]):  # 상위 3개
        chars = [chr(ord('A') + idx - 1) if idx > 0 else '_' for idx in sequence]
        print(f"  {j+1}. {''.join(chars)} (score: {score:.4f})")
```

## 고급 평가 메트릭

### 음성 품질 평가

```python
from pytorch_hmm import (
    mel_cepstral_distortion,
    f0_root_mean_square_error,
    comprehensive_speech_evaluation
)

# 예측된 음성 특징과 실제 음성 특징
predicted_mel = torch.randn(100, 80)  # 100 프레임, 80차원 멜 스펙트로그램
target_mel = torch.randn(100, 80)

predicted_f0 = torch.randn(100) * 100 + 150  # F0 값 (Hz)
target_f0 = torch.randn(100) * 100 + 150

# 멜 켑스트럼 왜곡 (MCD)
mcd = mel_cepstral_distortion(predicted_mel, target_mel)
print(f"Mel-Cepstral Distortion: {mcd:.4f} dB")

# F0 RMSE
f0_rmse = f0_root_mean_square_error(predicted_f0, target_f0)
print(f"F0 RMSE: {f0_rmse:.4f} Hz")

# 종합 평가
evaluation_results = comprehensive_speech_evaluation(
    predicted_mel, target_mel,
    predicted_f0, target_f0,
    sample_rate=22050
)

print("Comprehensive evaluation:")
for metric, value in evaluation_results.items():
    print(f"  {metric}: {value:.4f}")
```

### 정렬 정확도 평가

```python
from pytorch_hmm import alignment_accuracy

# 예측된 정렬과 실제 정렬
predicted_alignment = torch.randint(0, 5, (100,))  # 100 프레임
target_alignment = torch.randint(0, 5, (100,))

# 정렬 정확도 계산
accuracy = alignment_accuracy(predicted_alignment, target_alignment)
print(f"Alignment accuracy: {accuracy:.4f}")

# 프레임별 허용 오차를 고려한 정확도
tolerance_accuracy = alignment_accuracy(
    predicted_alignment, 
    target_alignment, 
    tolerance=2  # ±2 프레임 허용
)
print(f"Alignment accuracy (±2 frames): {tolerance_accuracy:.4f}")
```

## 성능 최적화 기법

### 배치 처리 최적화

```python
import torch.nn.functional as F

class OptimizedHMM(nn.Module):
    def __init__(self, num_states, observation_dim):
        super().__init__()
        self.num_states = num_states
        self.observation_dim = observation_dim
        
        # 파라미터 초기화
        self.log_transition = nn.Parameter(torch.randn(num_states, num_states))
        self.log_initial = nn.Parameter(torch.randn(num_states))
        
        # 관측 모델 (효율적인 구현)
        self.observation_net = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_states)
        )
    
    def forward(self, observations):
        batch_size, seq_length, _ = observations.shape
        
        # 관측 확률 계산 (배치 처리)
        log_obs_probs = self.observation_net(observations.view(-1, self.observation_dim))
        log_obs_probs = log_obs_probs.view(batch_size, seq_length, self.num_states)
        
        # Forward algorithm (벡터화된 구현)
        log_alpha = self.log_initial.unsqueeze(0) + log_obs_probs[:, 0]
        
        for t in range(1, seq_length):
            log_alpha = torch.logsumexp(
                log_alpha.unsqueeze(2) + self.log_transition.unsqueeze(0),
                dim=1
            ) + log_obs_probs[:, t]
        
        # 로그 우도 계산
        log_likelihood = torch.logsumexp(log_alpha, dim=1)
        return log_likelihood

# 사용 예제
optimized_hmm = OptimizedHMM(num_states=10, observation_dim=80)
observations = torch.randn(32, 100, 80)  # 큰 배치 크기

# 성능 측정
import time
start_time = time.time()
log_likelihood = optimized_hmm(observations)
end_time = time.time()

print(f"Batch processing time: {end_time - start_time:.4f}s")
print(f"Average log-likelihood: {log_likelihood.mean().item():.4f}")
```

### GPU 가속화

```python
# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 모델과 데이터를 GPU로 이동
optimized_hmm = optimized_hmm.to(device)
observations = observations.to(device)

# GPU에서 추론
with torch.no_grad():
    start_time = time.time()
    log_likelihood = optimized_hmm(observations)
    torch.cuda.synchronize()  # GPU 동기화
    end_time = time.time()

print(f"GPU processing time: {end_time - start_time:.4f}s")
```

### 메모리 효율적인 구현

```python
import torch.utils.checkpoint as checkpoint

class MemoryEfficientHMM(nn.Module):
    def __init__(self, num_states, observation_dim):
        super().__init__()
        self.num_states = num_states
        self.hmm_layer = HMMLayer(num_states, observation_dim)
    
    def forward(self, observations):
        # Gradient checkpointing으로 메모리 사용량 줄이기
        return checkpoint.checkpoint(self.hmm_layer, observations)

# 긴 시퀀스에 대한 청크 처리
def process_long_sequence(hmm_model, long_observations, chunk_size=1000):
    results = []
    seq_length = long_observations.shape[1]
    
    for start in range(0, seq_length, chunk_size):
        end = min(start + chunk_size, seq_length)
        chunk = long_observations[:, start:end]
        
        with torch.no_grad():
            result = hmm_model(chunk)
            results.append(result)
    
    return torch.cat(results, dim=1)

# 사용 예제
memory_efficient_hmm = MemoryEfficientHMM(num_states=10, observation_dim=80)
long_observations = torch.randn(1, 10000, 80)  # 매우 긴 시퀀스

processed_results = process_long_sequence(
    memory_efficient_hmm, 
    long_observations, 
    chunk_size=1000
)
```

## 실제 응용 예제

### 음성 합성 (TTS) 파이프라인

```python
class TTSPipeline:
    def __init__(self):
        # 각 구성 요소 초기화
        self.text_encoder = self._build_text_encoder()
        self.duration_predictor = self._build_duration_predictor()
        self.acoustic_model = self._build_acoustic_model()
        self.vocoder = self._build_vocoder()
    
    def _build_text_encoder(self):
        # 텍스트 -> 언어적 특징
        return nn.Sequential(
            nn.Embedding(256, 128),  # 문자 임베딩
            nn.LSTM(128, 64, batch_first=True)
        )
    
    def _build_duration_predictor(self):
        # 지속 시간 예측
        return SemiMarkovHMM(
            num_states=50,  # 음소 수
            observation_dim=64,
            duration_model=DurationModel(50, 'gamma')
        )
    
    def _build_acoustic_model(self):
        # 음향 특징 생성
        return ContextualNeuralHMM(
            num_states=150,  # 음소 × 상태
            observation_dim=80,  # 멜 스펙트로그램
            context_dim=64,
            hidden_dim=256
        )
    
    def _build_vocoder(self):
        # 멜 스펙트로그램 -> 음성 파형
        return nn.Sequential(
            nn.ConvTranspose1d(80, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, 4, 2, 1),
            nn.Tanh()
        )
    
    def synthesize(self, text):
        # 1. 텍스트 인코딩
        text_features, _ = self.text_encoder(text)
        
        # 2. 지속 시간 예측
        durations = self.duration_predictor.predict_durations(text_features)
        
        # 3. 음향 특징 생성
        mel_spectrogram = self.acoustic_model.generate(text_features, durations)
        
        # 4. 음성 파형 생성
        waveform = self.vocoder(mel_spectrogram.transpose(1, 2))
        
        return waveform.squeeze()

# TTS 파이프라인 사용
tts = TTSPipeline()
text_input = torch.randint(0, 256, (1, 20))  # 텍스트 시퀀스
synthesized_audio = tts.synthesize(text_input)
print(f"Synthesized audio shape: {synthesized_audio.shape}")
```

### 음성 인식 (ASR) 파이프라인

```python
class ASRPipeline:
    def __init__(self):
        self.feature_extractor = self._build_feature_extractor()
        self.acoustic_model = self._build_acoustic_model()
        self.language_model = self._build_language_model()
        self.ctc_decoder = CTCAligner(vocab_size=1000, blank_id=0)
    
    def _build_feature_extractor(self):
        # 음성 -> 음향 특징
        return nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv1d(128, 80, 3, 2, 1)
        )
    
    def _build_acoustic_model(self):
        # 음향 특징 -> 음소 확률
        return NeuralHMM(
            num_states=200,  # 음소 상태 수
            observation_dim=80,
            context_dim=0,  # 컨텍스트 없음
            hidden_dim=256
        )
    
    def _build_language_model(self):
        # 언어 모델 (간단한 LSTM)
        return nn.LSTM(1000, 256, batch_first=True)
    
    def recognize(self, audio_waveform):
        # 1. 특징 추출
        features = self.feature_extractor(audio_waveform.unsqueeze(1))
        features = features.transpose(1, 2)  # (B, T, F)
        
        # 2. 음향 모델링
        phone_posteriors, _, _ = self.acoustic_model(features)
        
        # 3. CTC 디코딩
        input_lengths = torch.full((features.shape[0],), features.shape[1])
        decoded_phones = self.ctc_decoder.decode(
            torch.log_softmax(phone_posteriors, dim=-1).transpose(0, 1),
            input_lengths
        )
        
        # 4. 언어 모델 적용 (선택적)
        # lm_scores, _ = self.language_model(decoded_phones)
        
        return decoded_phones

# ASR 파이프라인 사용
asr = ASRPipeline()
audio_input = torch.randn(1, 16000)  # 1초 음성 (16kHz)
recognized_text = asr.recognize(audio_input)
print(f"Recognized phones: {recognized_text}")
```

## 마무리

이 문서에서는 PyTorch HMM 라이브러리의 고급 기능들을 살펴보았습니다:

- **Neural HMM**: 신경망과 HMM의 결합으로 복잡한 패턴 학습
- **Semi-Markov HMM**: 상태 지속 시간의 명시적 모델링
- **정렬 알고리즘**: DTW와 CTC를 통한 시퀀스 정렬
- **고급 평가 메트릭**: 음성 품질과 정렬 정확도 평가
- **성능 최적화**: 배치 처리, GPU 가속, 메모리 효율성
- **실제 응용**: TTS와 ASR 파이프라인 구현

이러한 고급 기능들을 활용하면 더욱 정교하고 실용적인 음성 처리 시스템을 구축할 수 있습니다.

다음 단계로는 [실제 응용 사례](04_real_world_applications.md)를 살펴보거나 [성능 최적화 가이드](05_performance_optimization.md)를 참고하시기 바랍니다.
