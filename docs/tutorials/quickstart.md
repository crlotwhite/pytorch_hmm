# 🚀 빠른 시작 가이드

PyTorch HMM을 5분 만에 시작하는 가이드입니다.

## 📦 설치

### 요구사항
- Python 3.8+
- PyTorch 1.9+
- CUDA (GPU 가속용, 선택사항)

### 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/crlotwhite/pytorch_hmm.git
cd pytorch_hmm

# 2. 의존성 설치 (uv 권장)
pip install uv
uv sync

# 또는 pip 사용
pip install -e .
```

## 🎯 첫 번째 HMM 모델

### 1. 기본 HMM 사용

```python
import torch
from pytorch_hmm import HMMPyTorch

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# HMM 모델 생성 (5개 상태)
num_states = 5
hmm = HMMPyTorch(num_states=num_states).to(device)

# 관측 데이터 생성 (배치 크기: 2, 시퀀스 길이: 100)
batch_size, seq_len = 2, 100
observations = torch.randn(batch_size, seq_len, num_states, device=device)

# Forward-backward 알고리즘 실행
posteriors, log_likelihood = hmm.forward_backward(observations)

print(f"Log-likelihood: {log_likelihood}")
print(f"Posterior shape: {posteriors.shape}")  # [batch_size, seq_len, num_states]
```

### 2. Viterbi 디코딩

```python
# 최적 상태 시퀀스 찾기
best_states, best_scores = hmm.viterbi_decode(observations)

print(f"최적 상태 시퀀스 (첫 10개): {best_states[0, :10]}")
print(f"최적 점수: {best_scores[0]}")
```

## 🔥 고급 모델 사용

### 1. MixtureGaussianHMM

```python
from pytorch_hmm import MixtureGaussianHMM

# 가우시안 혼합 HMM 생성
mixture_hmm = MixtureGaussianHMM(
    num_states=5,
    obs_dim=80,        # 음성 특징 차원 (예: 멜 스펙트로그램)
    num_mixtures=3     # 가우시안 혼합 수
).to(device)

# 음성 특징 데이터 (예: 멜 스펙트로그램)
mel_features = torch.randn(batch_size, seq_len, 80, device=device)

# Forward pass
log_probs = mixture_hmm.forward(mel_features)
print(f"Log probabilities shape: {log_probs.shape}")
```

### 2. 실시간 스트리밍

```python
from pytorch_hmm import StreamingHMMProcessor

# 실시간 스트리밍 프로세서 생성
streaming_processor = StreamingHMMProcessor(
    hmm_model=hmm,
    chunk_size=160,    # 10ms @ 16kHz
    overlap=80         # 50% 오버랩
)

# 실시간 처리 시뮬레이션
def simulate_real_time_processing():
    """실시간 오디오 처리 시뮬레이션"""
    total_frames = 1000
    chunk_size = 160
    
    for i in range(0, total_frames, chunk_size):
        # 오디오 청크 시뮬레이션
        audio_chunk = torch.randn(chunk_size, 80, device=device)
        
        # 실시간 처리
        result = streaming_processor.process_chunk(audio_chunk)
        
        if result is not None:
            print(f"청크 {i//chunk_size + 1} 처리 완료, 결과 shape: {result.shape}")

simulate_real_time_processing()
```

## 🎵 음성 합성 예제

### TTS 시스템 통합

```python
import torch.nn as nn
from pytorch_hmm import HMMLayer

class SimpleTTSModel(nn.Module):
    """간단한 TTS 모델 예제"""
    
    def __init__(self, text_dim=256, num_phonemes=50, mel_dim=80):
        super().__init__()
        
        # 텍스트 인코더
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # HMM 정렬 레이어
        self.hmm_layer = HMMLayer(
            num_states=num_phonemes,
            learnable_transitions=True
        )
        
        # 멜 스펙트로그램 디코더
        self.mel_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, mel_dim)
        )
    
    def forward(self, text_features):
        # 텍스트 인코딩
        encoded = self.text_encoder(text_features)
        
        # HMM 정렬
        aligned_features, posteriors = self.hmm_layer(encoded)
        
        # 멜 스펙트로그램 생성
        mel_output = self.mel_decoder(aligned_features)
        
        return mel_output, posteriors

# 모델 생성 및 테스트
tts_model = SimpleTTSModel().to(device)

# 텍스트 특징 (예: 음소 임베딩)
text_features = torch.randn(batch_size, seq_len, 256, device=device)

# TTS 실행
mel_output, posteriors = tts_model(text_features)

print(f"멜 스펙트로그램 shape: {mel_output.shape}")
print(f"정렬 확률 shape: {posteriors.shape}")
```

## 📊 성능 벤치마크

### 빠른 성능 테스트

```python
import time

def quick_benchmark():
    """빠른 성능 벤치마크"""
    
    # 테스트 설정
    batch_size, seq_len, obs_dim = 32, 1000, 80
    num_iterations = 100
    
    # 모델과 데이터 준비
    hmm = MixtureGaussianHMM(num_states=10, obs_dim=obs_dim).to(device)
    observations = torch.randn(batch_size, seq_len, obs_dim, device=device)
    
    # 워밍업
    for _ in range(10):
        _ = hmm.forward(observations)
    
    # 성능 측정
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        log_probs = hmm.forward(observations)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    
    # 결과 계산
    avg_time = (end_time - start_time) / num_iterations
    throughput = (batch_size * seq_len) / avg_time
    realtime_factor = throughput / 16000  # 16kHz 기준
    
    print(f"📊 성능 결과:")
    print(f"   ⏱️  평균 처리 시간: {avg_time*1000:.2f}ms")
    print(f"   🚀 처리량: {throughput:.0f} frames/sec")
    print(f"   ⚡ 실시간 배수: {realtime_factor:.1f}x")
    
    if realtime_factor > 100:
        print(f"   ✅ 실시간 처리 가능! ({realtime_factor:.0f}x 실시간)")
    else:
        print(f"   ⚠️ 실시간 처리 어려움 ({realtime_factor:.1f}x)")

# 벤치마크 실행
quick_benchmark()
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. GPU 메모리 부족
```python
# 배치 크기 줄이기
batch_size = 8  # 32 대신 8 사용

# 또는 그래디언트 누적 사용
with torch.no_grad():
    result = hmm.forward(observations)
```

#### 2. 느린 처리 속도
```python
# Mixed precision 사용
with torch.cuda.amp.autocast():
    result = hmm.forward(observations)

# JIT 컴파일 (가능한 경우)
try:
    jit_hmm = torch.jit.script(hmm)
    print("✅ JIT 컴파일 성공")
except:
    print("❌ JIT 컴파일 실패, 일반 모델 사용")
    jit_hmm = hmm
```

#### 3. 설치 문제
```bash
# CUDA 버전 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 의존성 재설치
pip uninstall pytorch-hmm
pip install -e . --force-reinstall
```

## 📚 다음 단계

### 더 자세한 학습
1. **[기본 HMM 사용법](basic_hmm.md)** - HMM 이론과 상세 사용법
2. **[고급 모델 활용](advanced_models.md)** - Neural HMM, HSMM 등
3. **[성능 최적화](optimization.md)** - GPU 가속, 메모리 최적화
4. **[실제 응용](../examples/integration_examples.md)** - 실제 프로젝트 통합

### 예제 코드
- **[기본 예제](../examples/basic_examples.md)** - 다양한 기본 사용 예제
- **[벤치마크 예제](../examples/benchmark_examples.md)** - 성능 측정 코드

### 도움말
- **[FAQ](../troubleshooting/faq.md)** - 자주 묻는 질문
- **[GitHub Issues](https://github.com/crlotwhite/pytorch_hmm/issues)** - 버그 신고 및 질문

---

**축하합니다! 🎉 PyTorch HMM 기본 사용법을 익혔습니다. 이제 고급 기능들을 탐험해보세요!** 