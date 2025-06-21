# 히든 마르코프 모델(HMM) 완전 가이드

## 목차
1. [HMM 기초 이론](#1-hmm-기초-이론)
2. [PyTorch HMM 라이브러리 소개](#2-pytorch-hmm-라이브러리-소개)
3. [기본 사용법](#3-기본-사용법)
4. [고급 기능](#4-고급-기능)
5. [실제 응용 예제](#5-실제-응용-예제)
6. [성능 최적화](#6-성능-최적화)
7. [문제 해결](#7-문제-해결)

---

## 1. HMM 기초 이론

### 1.1 히든 마르코프 모델이란?

히든 마르코프 모델(Hidden Markov Model, HMM)은 시간에 따라 변화하는 시스템을 모델링하는 확률적 방법입니다. 특히 **관측할 수 없는 숨겨진 상태(hidden state)**가 **관측 가능한 출력(observation)**을 생성하는 상황에서 사용됩니다.

#### 핵심 구성 요소

1. **상태 집합 (States)**: S = {s₁, s₂, ..., sₙ}
2. **관측 집합 (Observations)**: O = {o₁, o₂, ..., oₘ}
3. **전이 확률 (Transition Probabilities)**: A = {aᵢⱼ = P(qₜ₊₁ = sⱼ | qₜ = sᵢ)}
4. **방출 확률 (Emission Probabilities)**: B = {bⱼ(oₖ) = P(oₜ = oₖ | qₜ = sⱼ)}
5. **초기 확률 (Initial Probabilities)**: π = {πᵢ = P(q₁ = sᵢ)}

### 1.2 HMM의 세 가지 기본 문제

#### 문제 1: 평가 (Evaluation)
주어진 모델 λ = (A, B, π)와 관측 시퀀스 O에 대해 P(O|λ)를 계산
- **해결법**: Forward-Backward 알고리즘

#### 문제 2: 디코딩 (Decoding)
주어진 모델 λ와 관측 시퀀스 O에 대해 가장 가능성 높은 상태 시퀀스 찾기
- **해결법**: Viterbi 알고리즘

#### 문제 3: 학습 (Learning)
관측 시퀀스 O가 주어졌을 때 모델 파라미터 λ 추정
- **해결법**: Baum-Welch 알고리즘 (EM 알고리즘의 특수한 경우)

### 1.3 음성 처리에서의 HMM

음성 신호는 시간에 따라 변화하는 특성을 가지며, HMM은 이러한 특성을 모델링하는 데 매우 적합합니다:

- **음소(Phoneme)**: 각 음소를 여러 상태로 나누어 모델링
- **음성 합성**: 텍스트에서 음성으로 변환 시 정렬 문제 해결
- **음성 인식**: 음향 신호에서 언어적 단위 추출

---

## 2. PyTorch HMM 라이브러리 소개

### 2.1 주요 특징

- **PyTorch 네이티브**: 자동 미분과 GPU 가속 지원
- **모듈식 설계**: 다양한 HMM 변형들을 쉽게 조합
- **음성 처리 최적화**: TTS, ASR 등 음성 애플리케이션에 특화
- **실시간 처리**: 스트리밍 HMM 프로세서 제공

### 2.2 핵심 클래스들

```python
from pytorch_hmm import (
    HMMPyTorch,           # 기본 HMM 구현
    HMMLayer,             # nn.Module 래퍼
    GaussianHMMLayer,     # 가우시안 관측 모델
    MixtureGaussianHMMLayer,  # 혼합 가우시안 모델
    HSMMLayer,            # 반마르코프 모델
    StreamingHMMProcessor # 실시간 처리
)
```

### 2.3 설치 및 설정

```bash
# 개발 버전 설치
pip install -e .

# 의존성 확인
python -c "import pytorch_hmm; pytorch_hmm.run_quick_test()"
```

---

## 3. 기본 사용법

### 3.1 첫 번째 HMM 모델

```python
import torch
from pytorch_hmm import HMMPyTorch, create_left_to_right_matrix

# 1. 전이 행렬 생성
num_states = 5
transition_matrix = create_left_to_right_matrix(
    num_states, 
    self_loop_prob=0.7  # 자기 자신으로의 전이 확률
)

# 2. HMM 모델 생성
hmm = HMMPyTorch(transition_matrix)

# 3. 관측 데이터 준비
batch_size, seq_len = 2, 50
observations = torch.softmax(
    torch.randn(batch_size, seq_len, num_states), 
    dim=-1
)

# 4. Forward-backward 알고리즘
posteriors, forward, backward = hmm.forward_backward(observations)
print(f"Posterior shape: {posteriors.shape}")  # [batch, seq_len, num_states]

# 5. Viterbi 디코딩
states, scores = hmm.viterbi_decode(observations)
print(f"Optimal states: {states[0, :10]}")  # 첫 10개 상태
```

### 3.2 Forward-Backward vs Viterbi

```python
def compare_algorithms(hmm, observations):
    """Forward-backward와 Viterbi 알고리즘 비교"""
    
    # Forward-backward: 확률적 정렬
    posteriors, _, _ = hmm.forward_backward(observations)
    soft_states = torch.argmax(posteriors, dim=-1)
    
    # Viterbi: 결정적 정렬
    hard_states, scores = hmm.viterbi_decode(observations)
    
    # 일치도 계산
    agreement = (soft_states == hard_states).float().mean()
    print(f"State agreement: {agreement:.3f}")
    
    return soft_states, hard_states

# 사용 예시
soft_alignment, hard_alignment = compare_algorithms(hmm, observations)
```

### 3.3 신경망과의 통합

```python
import torch.nn as nn
from pytorch_hmm import HMMLayer

class TextToSpeechModel(nn.Module):
    def __init__(self, text_dim, num_phonemes, mel_dim):
        super().__init__()
        
        # 텍스트 인코더
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        
        # HMM 정렬 레이어
        self.hmm_layer = HMMLayer(
            num_states=num_phonemes,
            learnable_transitions=True,
            transition_type="left_to_right"
        )
        
        # 음향 디코더
        self.acoustic_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, mel_dim)
        )
    
    def forward(self, text_features, mel_features=None):
        # 텍스트 인코딩
        encoded = self.text_encoder(text_features)
        
        if mel_features is not None:
            # 훈련 시: 정렬 학습
            alignment = self.hmm_layer(encoded, mel_features)
            mel_pred = self.acoustic_decoder(encoded)
            return mel_pred, alignment
        else:
            # 추론 시: 멜 스펙트로그램 생성
            mel_pred = self.acoustic_decoder(encoded)
            return mel_pred

# 모델 사용
model = TextToSpeechModel(text_dim=128, num_phonemes=50, mel_dim=80)
text_input = torch.randn(1, 100, 128)  # [batch, seq_len, text_dim]
mel_output = model(text_input)
```

---

## 4. 고급 기능

### 4.1 혼합 가우시안 HMM

복잡한 음향 특성을 모델링하기 위해 혼합 가우시안 관측 모델을 사용할 수 있습니다.

```python
from pytorch_hmm import MixtureGaussianHMMLayer

# 혼합 가우시안 HMM 생성
mixture_hmm = MixtureGaussianHMMLayer(
    num_states=20,
    observation_dim=80,  # 멜 스펙트로그램 차원
    num_mixtures=4,      # 각 상태당 가우시안 개수
    covariance_type='diagonal'
)

# 음향 특징으로 훈련
mel_features = torch.randn(10, 200, 80)  # [batch, time, mel_dim]
log_likelihood = mixture_hmm(mel_features)
print(f"Log-likelihood: {log_likelihood}")
```

### 4.2 반마르코프 모델 (HSMM)

명시적 지속시간 모델링이 필요한 경우 Hidden Semi-Markov Model을 사용합니다.

```python
from pytorch_hmm import HSMMLayer, DurationModel

# 지속시간 모델 정의
duration_model = DurationModel(
    num_states=10,
    max_duration=50,
    distribution_type='gamma'  # 감마 분포 사용
)

# HSMM 레이어 생성
hsmm = HSMMLayer(
    num_states=10,
    duration_model=duration_model,
    observation_dim=80
)

# 사용 예시
observations = torch.randn(2, 150, 80)
states, durations = hsmm.decode_with_duration(observations)
print(f"State sequence: {states[0, :20]}")
print(f"Durations: {durations[0, :10]}")
```

### 4.3 신경 HMM

컨텍스트 정보를 활용한 고급 HMM 모델입니다.

```python
from pytorch_hmm import NeuralHMM, ContextualNeuralHMM

# 기본 신경 HMM
neural_hmm = NeuralHMM(
    num_states=30,
    observation_dim=80,
    context_dim=64,
    hidden_dim=256,
    transition_type='rnn',
    observation_type='mixture'
)

# 컨텍스트 정보 포함 모델
contextual_hmm = ContextualNeuralHMM(
    num_states=30,
    observation_dim=80,
    context_dim=128,
    linguistic_dim=64,   # 언어적 특징
    prosodic_dim=32,     # 운율 특징
    speaker_dim=16       # 화자 특징
)

# 사용 예시
context = torch.randn(2, 100, 128)
linguistic = torch.randn(2, 100, 64)
prosodic = torch.randn(2, 100, 32)
speaker = torch.randn(2, 16)

posteriors = contextual_hmm(
    observations, context, linguistic, prosodic, speaker
)
```

### 4.4 정렬 알고리즘

#### Dynamic Time Warping (DTW)

```python
from pytorch_hmm import DTWAligner

# DTW 정렬기 생성
dtw_aligner = DTWAligner(
    distance_fn='cosine',
    step_pattern='symmetric'
)

# 음소와 음성 특징 정렬
phoneme_features = torch.randn(10, 40)   # 10개 음소
audio_features = torch.randn(200, 40)    # 200 프레임

path_i, path_j, cost = dtw_aligner(phoneme_features, audio_features)
print(f"Alignment path length: {len(path_i)}")
print(f"DTW cost: {cost:.4f}")
```

#### CTC 정렬

```python
from pytorch_hmm import CTCAligner

# CTC 정렬기 생성
ctc_aligner = CTCAligner(num_classes=28, blank_id=0)

# 음성 인식 시나리오
log_probs = torch.log_softmax(torch.randn(100, 2, 28), dim=-1)
targets = torch.tensor([[8, 5, 12, 12, 15], [23, 15, 18, 12, 4]])
input_lengths = torch.tensor([100, 100])
target_lengths = torch.tensor([5, 5])

# CTC 손실 계산
loss = ctc_aligner(log_probs, targets, input_lengths, target_lengths)
print(f"CTC Loss: {loss:.4f}")

# 디코딩
decoded = ctc_aligner.decode(log_probs, input_lengths)
```

---

## 5. 실제 응용 예제

### 5.1 음성 합성 (TTS) 시스템

```python
import torch
import torch.nn as nn
from pytorch_hmm import HMMLayer, create_left_to_right_matrix

class CompleteTTSSystem(nn.Module):
    def __init__(self, vocab_size, num_phonemes, mel_dim):
        super().__init__()
        
        # 텍스트 전처리
        self.embedding = nn.Embedding(vocab_size, 128)
        self.text_encoder = nn.LSTM(128, 256, batch_first=True)
        
        # 음소 HMM 정렬
        self.phoneme_hmm = HMMLayer(
            num_states=num_phonemes * 3,  # 각 음소당 3상태
            learnable_transitions=True,
            transition_type="left_to_right"
        )
        
        # 음향 모델
        self.acoustic_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, mel_dim)
        )
        
        # 보코더 (선택사항)
        self.vocoder = nn.Sequential(
            nn.ConvTranspose1d(mel_dim, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, text_ids, mel_targets=None):
        # 텍스트 인코딩
        embedded = self.embedding(text_ids)
        encoded, _ = self.text_encoder(embedded)
        
        if mel_targets is not None:
            # 훈련 모드: 정렬 학습
            alignment = self.phoneme_hmm(encoded, mel_targets)
            
            # 정렬된 텍스트 특징으로 멜 생성
            aligned_features = torch.bmm(alignment.transpose(1, 2), encoded)
            mel_pred = self.acoustic_model(aligned_features)
            
            return mel_pred, alignment
        else:
            # 추론 모드: 자동 회귀 생성
            mel_pred = self.acoustic_model(encoded)
            return mel_pred
    
    def synthesize(self, text_ids, max_length=1000):
        """음성 합성 수행"""
        with torch.no_grad():
            mel_pred = self.forward(text_ids)
            
            # 보코더로 파형 생성
            mel_transposed = mel_pred.transpose(1, 2)
            waveform = self.vocoder(mel_transposed)
            
            return waveform.squeeze(1)

# 사용 예시
tts_model = CompleteTTSSystem(vocab_size=100, num_phonemes=50, mel_dim=80)

# 훈련 데이터
text_input = torch.randint(0, 100, (2, 20))  # 배치 크기 2, 시퀀스 길이 20
mel_target = torch.randn(2, 200, 80)         # 멜 스펙트로그램

# 훈련
mel_pred, alignment = tts_model(text_input, mel_target)
loss = nn.MSELoss()(mel_pred, mel_target)

# 추론
synthesized_audio = tts_model.synthesize(text_input[:1])
print(f"Synthesized audio shape: {synthesized_audio.shape}")
```

### 5.2 음성 인식 (ASR) 시스템

```python
from pytorch_hmm import CTCAligner, HMMLayer

class HybridASRSystem(nn.Module):
    def __init__(self, input_dim, vocab_size, num_phonemes):
        super().__init__()
        
        # 음향 인코더
        self.acoustic_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.LSTM(512, 256, 2, batch_first=True, bidirectional=True)
        )
        
        # HMM 기반 음소 모델링
        self.phoneme_hmm = HMMLayer(
            num_states=num_phonemes,
            learnable_transitions=True
        )
        
        # CTC 출력 레이어
        self.ctc_output = nn.Linear(512, vocab_size)
        self.ctc_aligner = CTCAligner(vocab_size, blank_id=0)
        
        # 언어 모델 (선택사항)
        self.language_model = nn.LSTM(vocab_size, 256, batch_first=True)
        self.lm_output = nn.Linear(256, vocab_size)
    
    def forward(self, audio_features, targets=None, input_lengths=None, target_lengths=None):
        # 음향 특징 인코딩
        encoded, _ = self.acoustic_encoder(audio_features)
        
        # CTC 출력
        ctc_logits = self.ctc_output(encoded)
        ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)
        
        if targets is not None:
            # 훈련 모드
            ctc_loss = self.ctc_aligner(
                ctc_log_probs.transpose(0, 1), targets, input_lengths, target_lengths
            )
            return ctc_loss
        else:
            # 추론 모드
            decoded = self.ctc_aligner.decode(ctc_log_probs.transpose(0, 1), input_lengths)
            return decoded

# ASR 시스템 사용
asr_model = HybridASRSystem(input_dim=80, vocab_size=28, num_phonemes=50)

# 음향 특징 (MFCC, 멜 스펙트로그램 등)
audio_input = torch.randn(2, 200, 80)  # [batch, time, features]
targets = torch.tensor([[8, 5, 12, 12, 15], [23, 15, 18, 12, 4]])
input_lengths = torch.tensor([200, 200])
target_lengths = torch.tensor([5, 5])

# 훈련
loss = asr_model(audio_input, targets, input_lengths, target_lengths)
print(f"CTC Loss: {loss:.4f}")

# 추론
recognized_text = asr_model(audio_input, input_lengths=input_lengths)
print(f"Recognized: {recognized_text}")
```

### 5.3 실시간 스트리밍 처리

```python
from pytorch_hmm import StreamingHMMProcessor, AdaptiveLatencyController

class RealTimeASR:
    def __init__(self, model, chunk_size=160):
        self.model = model
        self.chunk_size = chunk_size
        
        # 스트리밍 HMM 프로세서
        self.streaming_processor = StreamingHMMProcessor(
            model=model,
            chunk_size=chunk_size,
            overlap_size=40,
            max_delay=500  # 최대 지연 시간 (ms)
        )
        
        # 적응적 지연 제어
        self.latency_controller = AdaptiveLatencyController(
            target_latency=200,  # 목표 지연 시간
            quality_threshold=0.8
        )
    
    def process_audio_stream(self, audio_stream):
        """실시간 오디오 스트림 처리"""
        results = []
        
        for chunk in audio_stream:
            # 청크 단위 처리
            chunk_result = self.streaming_processor.process_chunk(chunk)
            
            if chunk_result.is_complete:
                # 지연 시간 조정
                adjusted_result = self.latency_controller.adjust_output(chunk_result)
                results.append(adjusted_result)
                
                # 실시간 출력
                print(f"Recognized: {adjusted_result.text}")
                print(f"Confidence: {adjusted_result.confidence:.3f}")
                print(f"Latency: {adjusted_result.latency_ms}ms")
        
        return results

# 실시간 ASR 사용 예시
real_time_asr = RealTimeASR(asr_model)

# 시뮬레이션된 오디오 스트림
def audio_stream_generator():
    """오디오 스트림 시뮬레이션"""
    for i in range(100):
        yield torch.randn(160, 80)  # 10ms 청크 (16kHz 기준)

# 실시간 처리
stream_results = real_time_asr.process_audio_stream(audio_stream_generator())
```

---

## 6. 성능 최적화

### 6.1 GPU 가속

```python
import torch
from pytorch_hmm import HMMPyTorch, Config

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Config.set_device(device)

# 모델을 GPU로 이동
hmm = HMMPyTorch(transition_matrix).to(device)
observations = observations.to(device)

# 혼합 정밀도 사용
Config.set_precision(use_mixed_precision=True)

with torch.cuda.amp.autocast():
    posteriors, _, _ = hmm.forward_backward(observations)
```

### 6.2 배치 처리 최적화

```python
def efficient_batch_processing(hmm, observations_list, batch_size=32):
    """효율적인 배치 처리"""
    results = []
    
    for i in range(0, len(observations_list), batch_size):
        batch = observations_list[i:i+batch_size]
        
        # 패딩으로 길이 통일
        max_len = max(obs.size(1) for obs in batch)
        padded_batch = []
        lengths = []
        
        for obs in batch:
            seq_len = obs.size(1)
            if seq_len < max_len:
                padding = torch.zeros(obs.size(0), max_len - seq_len, obs.size(2))
                obs = torch.cat([obs, padding], dim=1)
            padded_batch.append(obs)
            lengths.append(seq_len)
        
        # 배치 처리
        batch_obs = torch.stack(padded_batch)
        batch_posteriors, _, _ = hmm.forward_backward(batch_obs)
        
        # 원래 길이로 자르기
        for j, length in enumerate(lengths):
            results.append(batch_posteriors[j, :length])
    
    return results
```

### 6.3 메모리 효율성

```python
from pytorch_hmm import Config

# 청킹으로 메모리 사용량 제어
Config.DEFAULT_CHUNK_SIZE = 500  # 시퀀스를 500 프레임씩 나누어 처리

def memory_efficient_processing(hmm, long_sequence):
    """긴 시퀀스의 메모리 효율적 처리"""
    chunk_size = Config.DEFAULT_CHUNK_SIZE
    seq_len = long_sequence.size(1)
    
    all_posteriors = []
    
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = long_sequence[:, start:end]
        
        with torch.no_grad():  # 그래디언트 계산 비활성화
            chunk_posteriors, _, _ = hmm.forward_backward(chunk)
            all_posteriors.append(chunk_posteriors.cpu())  # CPU로 이동하여 메모리 절약
    
    return torch.cat(all_posteriors, dim=1)
```

---

## 7. 문제 해결

### 7.1 일반적인 오류들

#### 차원 불일치 오류
```python
# 문제: 관측 데이터와 모델 차원이 맞지 않음
observations = torch.randn(2, 100, 40)  # 40차원
hmm = HMMPyTorch(transition_matrix)     # 5상태 모델

# 해결: 차원 확인 및 조정
print(f"Observation dim: {observations.size(-1)}")
print(f"Model states: {hmm.K}")

# 관측 모델 사용
from pytorch_hmm import GaussianHMMLayer
gaussian_hmm = GaussianHMMLayer(num_states=5, observation_dim=40)
```

#### 수치적 불안정성
```python
from pytorch_hmm import Config

# 수치적 안정성을 위한 설정
Config.EPS = 1e-8  # 작은 값 임계치
Config.LOG_EPS = -18.42  # 로그 공간에서의 작은 값

# 로그 공간에서 계산 수행
def stable_forward_backward(hmm, observations):
    """수치적으로 안정한 forward-backward"""
    # 관측 확률을 로그 공간으로 변환
    log_observations = torch.log(observations + Config.EPS)
    
    # 로그 공간에서 계산
    log_posteriors, log_forward, log_backward = hmm.forward_backward(
        torch.exp(log_observations)
    )
    
    return log_posteriors, log_forward, log_backward
```

### 7.2 성능 디버깅

```python
import time
import torch.profiler

def profile_hmm_performance(hmm, observations):
    """HMM 성능 프로파일링"""
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, 
                   torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        # Forward-backward 프로파일링
        start_time = time.time()
        posteriors, _, _ = hmm.forward_backward(observations)
        fb_time = time.time() - start_time
        
        # Viterbi 프로파일링
        start_time = time.time()
        states, _ = hmm.viterbi_decode(observations)
        viterbi_time = time.time() - start_time
    
    # 결과 출력
    print(f"Forward-backward time: {fb_time:.4f}s")
    print(f"Viterbi time: {viterbi_time:.4f}s")
    print(f"Speed ratio: {viterbi_time/fb_time:.2f}x")
    
    # 프로파일 결과 저장
    prof.export_chrome_trace("hmm_profile.json")
    print("Profile saved to hmm_profile.json")
```

### 7.3 모델 검증

```python
def validate_hmm_model(hmm, test_observations):
    """HMM 모델 검증"""
    
    # 1. 전이 행렬 검증
    transition_matrix = hmm.P
    row_sums = transition_matrix.sum(dim=1)
    print(f"Transition matrix row sums: {row_sums}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    
    # 2. 확률 일관성 검증
    posteriors, _, _ = hmm.forward_backward(test_observations)
    posterior_sums = posteriors.sum(dim=-1)
    print(f"Posterior sums (should be ~1.0): {posterior_sums[0, :5]}")
    
    # 3. Viterbi vs Forward-backward 일관성
    viterbi_states, viterbi_scores = hmm.viterbi_decode(test_observations)
    fb_states = torch.argmax(posteriors, dim=-1)
    
    agreement = (viterbi_states == fb_states).float().mean()
    print(f"Viterbi-FB agreement: {agreement:.3f}")
    
    # 4. 로그 우도 계산
    log_likelihood = hmm.compute_likelihood(test_observations)
    print(f"Log-likelihood: {log_likelihood}")
    
    return {
        'transition_valid': torch.allclose(row_sums, torch.ones_like(row_sums)),
        'posterior_valid': torch.allclose(posterior_sums, torch.ones_like(posterior_sums)),
        'agreement_rate': agreement.item(),
        'log_likelihood': log_likelihood.item()
    }

# 모델 검증 실행
validation_results = validate_hmm_model(hmm, observations)
print("Validation results:", validation_results)
```

---

## 마무리

이 튜토리얼에서는 히든 마르코프 모델의 기초 이론부터 PyTorch HMM 라이브러리를 사용한 실제 구현까지 포괄적으로 다뤘습니다. 

### 주요 포인트

1. **이론적 기반**: HMM의 세 가지 기본 문제와 알고리즘 이해
2. **실용적 구현**: PyTorch를 활용한 효율적인 HMM 구현
3. **고급 기능**: 혼합 가우시안, 반마르코프, 신경 HMM 등
4. **실제 응용**: TTS, ASR 시스템 구축 예제
5. **최적화**: 성능과 메모리 효율성 개선 방법

### 다음 단계

- [고급 예제 모음](advanced_features_demo.py)에서 더 복잡한 사용 사례 학습
- [벤치마크 스크립트](benchmark.py)로 성능 측정 및 비교
- 실제 음성 데이터로 모델 훈련 및 평가

### 추가 자료

- [공식 문서](../README.md)
- [API 레퍼런스](../pytorch_hmm/__init__.py)
- [로드맵](../ROADMAP.md)

문제가 발생하거나 질문이 있으시면 GitHub Issues를 통해 문의해 주세요! 