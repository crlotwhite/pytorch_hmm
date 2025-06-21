# 📚 기본 사용 예제

PyTorch HMM의 다양한 기본 사용 예제들을 모았습니다.

## 🎯 기본 HMM 예제

### 1. 간단한 HMM 생성 및 사용

```python
import torch
from pytorch_hmm import HMMPyTorch

# 기본 HMM 생성
hmm = HMMPyTorch(num_states=5)

# 관측 데이터 생성
observations = torch.randn(2, 100, 5)  # [batch, time, features]

# Forward-backward 알고리즘
posteriors, log_likelihood = hmm.forward_backward(observations)
print(f"Posteriors shape: {posteriors.shape}")
print(f"Log-likelihood: {log_likelihood}")
```

### 2. Viterbi 디코딩

```python
# 최적 상태 시퀀스 찾기
best_states, scores = hmm.viterbi_decode(observations)
print(f"Best states shape: {best_states.shape}")
print(f"Best scores: {scores}")
```

## 🔥 고급 모델 예제

### 1. MixtureGaussianHMM

```python
from pytorch_hmm import MixtureGaussianHMM

# 가우시안 혼합 HMM
mixture_hmm = MixtureGaussianHMM(
    num_states=10,
    obs_dim=80,
    num_mixtures=4
)

# 멜 스펙트로그램 데이터
mel_data = torch.randn(4, 200, 80)

# Forward pass
log_probs = mixture_hmm.forward(mel_data)
print(f"Log probabilities: {log_probs.shape}")
```

### 2. Semi-Markov HMM

```python
from pytorch_hmm import SemiMarkovHMM

# Semi-Markov HMM (지속시간 모델링)
shmm = SemiMarkovHMM(
    num_states=8,
    obs_dim=80,
    max_duration=10
)

# Forward pass with duration modeling
duration_probs = shmm.forward(mel_data)
print(f"Duration probabilities: {duration_probs.shape}")
```

## 🎵 음성 처리 예제

### 1. 음성 정렬

```python
from pytorch_hmm.alignment import DTWAlignment, CTCAlignment

# DTW 정렬
dtw = DTWAlignment()
text_features = torch.randn(1, 50, 256)  # 텍스트 특징
audio_features = torch.randn(1, 200, 256)  # 오디오 특징

alignment_path = dtw.align(text_features, audio_features)
print(f"Alignment path: {alignment_path.shape}")

# CTC 정렬
ctc = CTCAlignment()
ctc_alignment = ctc.align(text_features, audio_features)
print(f"CTC alignment: {ctc_alignment.shape}")
```

### 2. 실시간 스트리밍

```python
from pytorch_hmm import StreamingHMMProcessor

# 스트리밍 프로세서 설정
processor = StreamingHMMProcessor(
    hmm_model=hmm,
    chunk_size=160,  # 10ms chunks
    overlap=80
)

# 실시간 처리 시뮬레이션
def process_audio_stream():
    for i in range(10):  # 10개 청크
        chunk = torch.randn(160, 80)  # 오디오 청크
        result = processor.process_chunk(chunk)
        if result is not None:
            print(f"Chunk {i}: {result.shape}")

process_audio_stream()
```

## 📊 성능 측정 예제

### 1. 벤치마크 테스트

```python
import time

def benchmark_hmm(model, data, num_runs=100):
    """HMM 모델 성능 벤치마크"""
    
    # 워밍업
    for _ in range(10):
        _ = model.forward(data)
    
    # 성능 측정
    start_time = time.time()
    for _ in range(num_runs):
        result = model.forward(data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = data.numel() / avg_time
    
    return {
        'avg_time': avg_time,
        'throughput': throughput,
        'realtime_factor': throughput / 16000
    }

# 벤치마크 실행
test_data = torch.randn(32, 1000, 80, device='cuda')
results = benchmark_hmm(mixture_hmm, test_data)
print(f"Results: {results}")
```

## 🔧 유틸리티 예제

### 1. 모델 저장/로드

```python
# 모델 저장
torch.save({
    'model_state_dict': hmm.state_dict(),
    'config': {
        'num_states': hmm.num_states,
        'obs_dim': getattr(hmm, 'obs_dim', None)
    }
}, 'hmm_model.pth')

# 모델 로드
checkpoint = torch.load('hmm_model.pth')
new_hmm = HMMPyTorch(**checkpoint['config'])
new_hmm.load_state_dict(checkpoint['model_state_dict'])
```

### 2. 시각화

```python
import matplotlib.pyplot as plt

def plot_alignment(alignment_path):
    """정렬 경로 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(alignment_path[:, 0], alignment_path[:, 1])
    plt.xlabel('Text Position')
    plt.ylabel('Audio Position')
    plt.title('Alignment Path')
    plt.grid(True)
    plt.show()

def plot_posteriors(posteriors):
    """사후 확률 시각화"""
    plt.figure(figsize=(12, 8))
    plt.imshow(posteriors[0].T, aspect='auto', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('States')
    plt.title('Posterior Probabilities')
    plt.colorbar()
    plt.show()
```

이러한 예제들을 통해 PyTorch HMM의 다양한 기능을 쉽게 익힐 수 있습니다. 