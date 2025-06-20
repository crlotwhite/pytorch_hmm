# 🎯 PyTorch HMM v0.2.0 - 음성 합성에 특화된 Hidden Markov Model
🎯 **Advanced PyTorch Hidden Markov Model Library for Speech Synthesis**

[![CI](https://github.com/crlotwhite/pytorch_hmm/workflows/CI/badge.svg)](https://github.com/crlotwhite/pytorch_hmm/actions)
[![codecov](https://codecov.io/gh/crlotwhite/pytorch_hmm/branch/main/graph/badge.svg)](https://codecov.io/gh/crlotwhite/pytorch_hmm)
[![PyPI version](https://badge.fury.io/py/pytorch-hmm.svg)](https://badge.fury.io/py/pytorch-hmm)
[![Python versions](https://img.shields.io/pypi/pyversions/pytorch-hmm.svg)](https://pypi.org/project/pytorch-hmm/)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/crlotwhite/pytorch_hmm)

PyTorch 기반 Hidden Markov Model 구현체로, **음성 합성(TTS)과 음성 처리**에 최적화되어 있습니다. Forward-backward와 Viterbi 알고리즘을 지원하며, autograd와 GPU 가속을 완벽하게 지원합니다.

## 🚀 v0.2.0 주요 기능

### ✨ 새로운 모델들
- 🎨 **MixtureGaussianHMM**: 복잡한 음향 모델링을 위한 GMM-HMM
- ⏰ **Semi-Markov Model (HSMM)**: 명시적 지속시간 모델링
- 📡 **StreamingHMM**: 실시간 낮은 지연시간 처리
- 🔄 **AdaptiveTransitions**: 컨텍스트 기반 동적 전이

### 🎯 음성 처리 특화 기능
- 🇰🇷 **한국어 TTS 지원**: 음소 정렬과 지속시간 제어
- 🎵 **운율 인식 전이**: F0와 에너지 기반 전이 행렬
- ⚡ **Skip-state 전이**: 빠른 발화 처리
- 📊 **성능 벤치마킹**: 종합적인 성능 분석 도구

### 💻 프로덕션 준비
- 🏭 **메모리 효율성**: 대용량 배치 처리 최적화
- 🔧 **TorchScript 지원**: 배포 최적화
- 📈 **실시간 모니터링**: 성능 통계 및 적응형 제어
- 🧪 **종합 테스트**: 95%+ 코드 커버리지

PyTorch 기반의 종합적인 Hidden Markov Model 라이브러리입니다. **음성 합성(TTS), 음성 인식(ASR), 시퀀스 모델링**에 최적화되어 있으며, 최신 딥러닝 기법과 전통적인 HMM을 결합한 고급 기능들을 제공합니다.

## ✨ 새로운 v0.2.0 주요 기능

### 🧠 **Neural HMM with Contextual Modeling**
- **Context-aware HMM**: 언어적 컨텍스트와 운율 정보를 활용
- **RNN/Transformer 기반 전이 모델**: 동적 전이 확률 계산
- **Mixture Gaussian 관측 모델**: 복잡한 음향 특징 모델링

### ⏱️ **Hidden Semi-Markov Model (HSMM)**
- **명시적 지속시간 모델링**: Gamma, Poisson, Neural 분포 지원
- **자연스러운 음소 지속시간**: 실제 음성 특성 반영
- **적응형 지속시간**: 컨텍스트에 따른 지속시간 조절

### 🎯 **Advanced Alignment Algorithms**
- **Dynamic Time Warping (DTW)**: 유연한 시퀀스 정렬, Soft-DTW 지원
- **CTC Alignment**: End-to-end 학습 가능한 정렬
- **Constrained alignment**: Bandwidth 제약, Monotonic 정렬

### 📊 **Comprehensive Evaluation Metrics**
- **Mel-Cepstral Distortion (MCD)**: 음성 품질의 객관적 평가
- **F0 RMSE**: 운율 모델링 성능 평가
- **Alignment Accuracy**: 정렬 정확도 및 경계 탐지 성능
- **Duration Modeling**: 지속시간 예측 정확도

### 🚀 **Production-Ready Features**
- **GPU 가속**: CUDA 지원으로 실시간 처리 가능 (>80fps)
- **배치 처리**: 효율적인 대용량 데이터 처리
- **JIT 호환**: `torch.jit.script`로 최적화
- **메모리 효율**: Log-space 계산으로 수치 안정성

## 📦 설치

### uv를 사용한 설치 (권장)

```bash
# 기본 설치 (CPU 버전)
uv add pytorch-hmm[cpu]

# GPU 지원 (CUDA 12.4)
uv add pytorch-hmm[cuda]

# 개발 환경 설정
git clone https://github.com/crlotwhite/pytorch_hmm.git
cd pytorch_hmm
uv sync --extra dev

# 특정 기능 그룹 설치
uv sync --extra audio      # 음성 처리
uv sync --extra visualization  # 시각화
uv sync --extra benchmarks # 성능 벤치마크
```

### PyTorch 버전 선택

이 라이브러리는 플랫폼별로 최적화된 PyTorch 버전을 자동으로 선택합니다:

- **macOS**: CPU 전용 버전
- **Linux/Windows**: CUDA 12.4 지원 버전 (GPU 가용시)

수동으로 PyTorch 버전을 선택하려면:

```bash
# CPU 전용 강제 설치
uv sync --extra cpu

# CUDA 버전 강제 설치
uv sync --extra cuda
```

### pip를 사용한 설치

```bash
# 기본 설치
pip install pytorch-hmm

# 음성 처리 기능 포함
pip install pytorch-hmm[audio]

# 전체 기능 (개발용)
pip install pytorch-hmm[all]

# 개발 버전
git clone https://github.com/crlotwhite/pytorch_hmm.git
cd pytorch_hmm
pip install -e .[dev]
```

### 선택적 의존성

```bash
# 오디오 처리
pip install pytorch-hmm[audio]

# 시각화
pip install pytorch-hmm[visualization]

# 개발 도구
pip install pytorch-hmm[dev]

# 벤치마크 도구
pip install pytorch-hmm[benchmarks]

```

## 🚀 빠른 시작

### 1. 기본 HMM 사용법

```python
import torch
from pytorch_hmm import create_speech_hmm, get_speech_transitions

# 음성용 HMM 생성
model = create_speech_hmm(
    num_states=10,      # 음소 개수
    feature_dim=80,     # 멜 스펙트로그램 차원
    model_type="mixture_gaussian"
)

# 음성 특징 데이터
features = torch.randn(4, 100, 80)  # (batch, time, features)

# 디코딩
decoded_states, log_probs = model(features, return_log_probs=True)
print(f"디코딩된 상태: {decoded_states.shape}")  # (4, 100)
```

### 🎨 Mixture Gaussian HMM

```python
from pytorch_hmm import MixtureGaussianHMMLayer

# 복잡한 음향 모델링
model = MixtureGaussianHMMLayer(
    num_states=20,          # 20개 음소
    feature_dim=80,         # 80차원 멜 스펙트로그램
    num_components=4,       # 4개 가우시안 믹스처
    covariance_type='diag'  # 대각 공분산
)

# 음향 특징 처리
audio_features = torch.randn(2, 150, 80)
states, confidence = model(audio_features, return_log_probs=True)

print(f"모델 정보: {model.get_model_info()}")
```

### ⏰ Semi-Markov Model (HSMM)

```python
from pytorch_hmm import HSMMLayer

# 지속시간 모델링
hsmm = HSMMLayer(
    num_states=15,
    feature_dim=80,
    duration_distribution='gamma',  # Gamma 분포
    max_duration=50                 # 최대 50프레임
)

# 자연스러운 음성 시퀀스 생성
states, observations = hsmm.generate_sequence(length=200)

# 지속시간 분석
expected_durations = hsmm.get_expected_durations()
print(f"각 상태의 예상 지속시간: {expected_durations}")
```

### 📡 실시간 스트리밍

```python
from pytorch_hmm import StreamingHMMProcessor, AdaptiveLatencyController

# 실시간 프로세서
processor = StreamingHMMProcessor(
    num_states=10,
    feature_dim=80,
    chunk_size=160,         # 10ms 청크
    use_beam_search=True,
    beam_width=4
)

# 적응형 지연시간 제어
controller = AdaptiveLatencyController(target_latency_ms=30.0)

# 실시간 처리 시뮬레이션
for i in range(100):
    # 10ms 오디오 청크
    audio_chunk = torch.randn(160, 80)

    # 처리
    result = processor.process_chunk(audio_chunk)

    if result.status == 'decoded':
        print(f"상태: {result.decoded_states}")
        print(f"처리 시간: {result.processing_time_ms:.1f}ms")

        # 성능 적응
        recommendations = controller.update(
            result.processing_time_ms,
            result.buffer_size
        )
```

### 🇰🇷 한국어 TTS 예시

```python
from pytorch_hmm import create_korean_tts_hmm

# 한국어 음소 집합으로 HMM 생성
korean_model = create_korean_tts_hmm(
    feature_dim=80,
    model_type="hsmm"  # 지속시간 모델링 포함
)

# 음소 시퀀스: "안녕하세요"
phoneme_sequence = ['sil', 'a', 'n', 'n', 'eo', 'ng', 'h', 'a', 's', 'e', 'j', 'o', 'sil']

print(f"한국어 HMM 정보:")
print(f"  상태 수: {korean_model.num_states}")
print(f"  특징 차원: {korean_model.feature_dim}")
```

## 🔧 고급 사용법

### 전이 행렬 커스터마이징

```python
from pytorch_hmm.utils import (
    create_skip_state_matrix,
    create_phoneme_aware_transitions,
    create_prosody_aware_transitions
)

# 빠른 발화용 Skip-state 전이
skip_transitions = create_skip_state_matrix(
    num_states=20,
    self_loop_prob=0.5,    # 낮은 유지 확률
    forward_prob=0.4,      # 높은 전진 확률
    skip_prob=0.1,         # 건너뛰기 허용
    max_skip=2
)

# 음소 지속시간 기반 전이
phoneme_durations = [5, 8, 6, 12, 4, 9, 7, 11]  # 각 음소별 평균 지속시간
duration_transitions = create_phoneme_aware_transitions(
    phoneme_durations,
    duration_variance=0.3
)

# 운율 정보 기반 동적 전이
f0_contour = torch.randn(100)      # F0 윤곽
energy_contour = torch.randn(100)  # 에너지 윤곽

prosody_transitions = create_prosody_aware_transitions(
    f0_contour, energy_contour, num_states=10
)
print(f"운율 기반 전이 행렬: {prosody_transitions.shape}")  # (100, 10, 10)
```

### 성능 최적화

```python
import pytorch_hmm

# 하드웨어 자동 설정
config_info = pytorch_hmm.auto_configure()
print(f"자동 설정: {config_info}")

# 벤치마크 실행
from pytorch_hmm.utils import benchmark_transition_operations

benchmark_results = benchmark_transition_operations(
    num_states_list=[5, 10, 20, 50],
    num_trials=100
)

for operation, results in benchmark_results.items():
    print(f"{operation}:")
    for num_states, time_ms in results.items():
        print(f"  {num_states} states: {time_ms:.2f}ms")

### 2. Neural HMM으로 고급 모델링

```python
from pytorch_hmm import NeuralHMM, ContextualNeuralHMM

# 컨텍스트 인식 Neural HMM
contextual_hmm = ContextualNeuralHMM(
    num_states=10,
    observation_dim=80,          # 멜 스펙트로그램 차원
    phoneme_vocab_size=50,       # 음소 개수
    linguistic_context_dim=32,   # 언어적 컨텍스트
    prosody_dim=8               # 운율 특징
)

# 입력 데이터 준비
batch_size, seq_len = 2, 100
observations = torch.randn(batch_size, seq_len, 80)           # 음향 특징
phoneme_sequence = torch.randint(0, 50, (batch_size, seq_len)) # 음소 시퀀스
prosody_features = torch.randn(batch_size, seq_len, 8)        # 운율 특징

# 컨텍스트 기반 정렬
posteriors, forward, backward = contextual_hmm.forward_with_context(
    observations, phoneme_sequence, prosody_features)

print(f"Context-aware posteriors: {posteriors.shape}")
```

### 3. Semi-Markov HMM으로 지속시간 모델링

```python
from pytorch_hmm import SemiMarkovHMM

# 지속시간 모델링이 가능한 HSMM
hsmm = SemiMarkovHMM(
    num_states=8,
    observation_dim=40,
    max_duration=20,
    duration_distribution='gamma',  # 감마 분포로 지속시간 모델링
    observation_model='gaussian'
)

# 시퀀스 샘플링
state_seq, duration_seq, observations = hsmm.sample(
    num_states=6, max_length=100)

print(f"Sampled states: {state_seq}")
print(f"State durations: {duration_seq}")
print(f"Total frames: {duration_seq.sum()} frames")

# Viterbi 디코딩으로 최적 상태-지속시간 시퀀스 찾기
optimal_states, optimal_durations, log_prob = hsmm.viterbi_decode(observations)
```

### 4. DTW 정렬

```python
from pytorch_hmm import DTWAligner

# DTW 정렬기 생성
aligner = DTWAligner(
    distance_fn='cosine',           # 코사인 거리
    step_pattern='symmetric',       # 대칭 스텝 패턴
    soft_dtw=True                  # 미분 가능한 Soft DTW
)

# 음소 특징과 음성 특징 정렬
phoneme_features = torch.randn(5, 12)   # 5개 음소
audio_features = torch.randn(100, 12)   # 100 프레임

# DTW 정렬 수행
path_i, path_j, total_cost = aligner(phoneme_features, audio_features)

print(f"Alignment path length: {len(path_i)}")
print(f"DTW cost: {total_cost:.4f}")
```

### 5. CTC 정렬

```python
from pytorch_hmm import CTCAligner

# CTC 정렬기 (음성 인식용)
ctc_aligner = CTCAligner(
    num_classes=28,  # 26 letters + blank + space
    blank_id=0
)

# 음성 인식 시뮬레이션
sequence_length, batch_size, vocab_size = 80, 2, 28
log_probs = torch.log_softmax(
    torch.randn(sequence_length, batch_size, vocab_size), dim=-1)

targets = torch.tensor([[8, 5, 12, 12, 15],   # "HELLO"
                       [23, 15, 18, 12, 4]])  # "WORLD"

input_lengths = torch.full((batch_size,), sequence_length)
target_lengths = torch.tensor([5, 5])

# CTC loss 계산
loss = ctc_aligner(log_probs, targets, input_lengths, target_lengths)

# Greedy 디코딩
decoded = ctc_aligner.decode(log_probs, input_lengths)
print(f"Decoded sequences: {decoded}")
```

## 📊 성능 평가

### 음성 품질 메트릭

```python
from pytorch_hmm import (
    mel_cepstral_distortion, f0_root_mean_square_error,
    comprehensive_speech_evaluation
)

# 시뮬레이션된 TTS 결과
seq_len, mfcc_dim = 200, 13
ground_truth_mfcc = torch.randn(seq_len, mfcc_dim)
predicted_mfcc = ground_truth_mfcc + 0.1 * torch.randn(seq_len, mfcc_dim)

# MCD 계산
mcd = mel_cepstral_distortion(ground_truth_mfcc, predicted_mfcc)
print(f"Mel-Cepstral Distortion: {mcd:.2f} dB")

# F0 평가
gt_f0 = torch.abs(torch.randn(seq_len)) * 100 + 120  # 120-220 Hz
pred_f0 = gt_f0 + 5 * torch.randn(seq_len)

f0_rmse = f0_root_mean_square_error(gt_f0, pred_f0)
print(f"F0 RMSE: {f0_rmse:.2f} Hz")

# 종합 평가
predicted_features = {
    'mfcc': predicted_mfcc.unsqueeze(0),
    'f0': pred_f0.unsqueeze(0)
}
ground_truth_features = {
    'mfcc': ground_truth_mfcc.unsqueeze(0),
    'f0': gt_f0.unsqueeze(0)
}

metrics = comprehensive_speech_evaluation(predicted_features, ground_truth_features)
print(f"Comprehensive metrics: {metrics}")
```

## 🎯 실제 응용 예제

### 완전한 TTS 파이프라인

```python
import torch.nn as nn
from pytorch_hmm import HMMLayer, DurationModel, DTWAligner

class AdvancedTTSModel(nn.Module):
    def __init__(self, vocab_size, num_phonemes, acoustic_dim):
        super().__init__()

        # 텍스트 인코더
        self.text_encoder = nn.Embedding(vocab_size, 256)

        # 지속시간 예측기 (HSMM 기반)
        self.duration_predictor = DurationModel(
            num_states=num_phonemes,
            max_duration=30,
            distribution_type='neural'
        )

        # HMM 정렬 레이어
        self.alignment_layer = HMMLayer(
            num_states=num_phonemes,
            learnable_transitions=True,
            transition_type="left_to_right"
        )

        # 음향 디코더
        self.acoustic_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, acoustic_dim)
        )

        # DTW 후처리
        self.dtw_aligner = DTWAligner(distance_fn='cosine')

    def forward(self, text_sequence, target_length=None):
        # 텍스트 인코딩
        text_emb = self.text_encoder(text_sequence)

        # 지속시간 예측
        phoneme_ids = text_sequence  # 간단화
        predicted_durations = self.duration_predictor.sample(phoneme_ids)

        # HMM 정렬
        aligned_features = self.alignment_layer(text_emb)

        # 음향 특징 생성
        acoustic_features = self.acoustic_decoder(aligned_features)

        # DTW로 길이 조정 (옵션)
        if target_length is not None:
            target_features = torch.randn(target_length, acoustic_features.shape[-1])
            path_i, path_j, _ = self.dtw_aligner(acoustic_features[0], target_features)
            # 길이 조정 로직...

        return acoustic_features, predicted_durations

# 사용 예제
model = AdvancedTTSModel(vocab_size=1000, num_phonemes=50, acoustic_dim=80)
text = torch.randint(0, 1000, (1, 20))  # 20개 토큰
acoustic_output, durations = model(text)

print(f"Generated acoustic features: {acoustic_output.shape}")
print(f"Predicted durations: {durations}")
```

## 🔧 고급 기능 및 설정

### 사용자 정의 전이 행렬

```python
from pytorch_hmm import create_transition_matrix

# 다양한 전이 패턴
transition_types = [
    "ergodic",              # 완전 연결
    "left_to_right",        # 순차 진행
    "left_to_right_skip",   # 건너뛰기 허용
    "circular"              # 순환 구조
]

for t_type in transition_types:
    P = create_transition_matrix(num_states=8, transition_type=t_type)
    print(f"{t_type}: {P.shape}")
```

### GPU 가속 및 최적화

```python
# GPU 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hmm = HMMPyTorch(transition_matrix).to(device)
observations = observations.to(device)

# JIT 컴파일로 속도 향상
@torch.jit.script
def fast_viterbi(hmm_model, obs):
    return hmm_model.viterbi_decode(obs)

# 배치 처리 최적화
def process_large_batch(hmm, data_loader):
    results = []
    for batch in data_loader:
        with torch.no_grad():
            batch = batch.to(device)
            posteriors, _, _ = hmm.forward_backward(batch)
            results.append(posteriors.cpu())
    return torch.cat(results, dim=0)

```

## 📈 성능 벤치마크

### 처리 속도 (RTX 3080 기준)
- **MixtureGaussianHMM**: ~18,000 frames/sec
- **HSMM**: ~12,000 frames/sec
- **StreamingHMM**: ~25,000 frames/sec
- **실시간 배율**: 150-400x (실시간 80fps 기준)

### 메모리 효율성
- **대용량 배치**: 32 sequences × 2000 frames 처리 가능
- **GPU 메모리**: 2GB 이하 (대부분의 작업)
- **스트리밍**: 일정한 메모리 사용량 (<100MB)

## 🧪 테스트 실행

```bash
# 빠른 기능 테스트
pytest tests/ -m "not slow"

# 전체 테스트 (시간 소요)
pytest tests/ --cov=pytorch_hmm

# GPU 테스트
pytest tests/ -m gpu

# 성능 벤치마크
pytest tests/ -m performance --benchmark-only

# 패키지 무결성 확인
python -c "import pytorch_hmm; pytorch_hmm.run_quick_test()"

### 빠른 벤치마크 실행

```bash
# 기본 성능 테스트
python examples/benchmark.py --quick

# 전체 벤치마크 (GPU 포함)
python examples/benchmark.py --save-results benchmark_results.json

# CPU만 테스트
python examples/benchmark.py --no-gpu
```

### 일반적인 성능 (RTX 3080 GPU 기준)

| 알고리즘 | 성능 (fps) | 실시간 처리 |
|---------|-----------|------------|
| **Basic HMM Forward-Backward** | ~15,000 | ✅ (188x faster) |
| **Basic HMM Viterbi** | ~25,000 | ✅ (312x faster) |
| **Neural HMM** | ~8,000 | ✅ (100x faster) |
| **DTW Alignment** | ~5,000 | ✅ (62x faster) |
| **CTC Decode** | ~12,000 | ✅ (150x faster) |

*실시간 처리 기준: 80fps (12.5ms/frame)*

## 🧪 테스트 및 검증

### 통합 테스트 실행

```bash
# 모든 새 기능 테스트
python tests/test_integration.py

# 기본 기능 테스트
python -m pytest tests/test_hmm.py -v

# 성능 회귀 테스트
python -m pytest tests/test_performance.py --benchmark-only
```

### 코드 품질 검사

```bash
# 코드 포맷팅
black pytorch_hmm/
isort pytorch_hmm/

# 린팅
flake8 pytorch_hmm/
mypy pytorch_hmm/
```

## 📖 학습 자료 및 예제

### 단계별 튜토리얼

```bash
# 1. 기본 사용법 학습
python examples/basic_tutorial.py

# 2. 고급 기능 데모
python examples/advanced_features_demo.py

# 3. 성능 벤치마크
python examples/benchmark.py
```

### 커맨드라인 도구

```bash
# 빠른 데모 실행
pytorch-hmm-demo

# 통합 테스트 실행
pytorch-hmm-test

```

## 📚 문서 및 예시

- 📖 **API 문서**: [pytorch-hmm.readthedocs.io](https://pytorch-hmm.readthedocs.io)
- 💡 **튜토리얼**: [examples/](examples/) 디렉토리
- 🎵 **음성 처리 예시**: [examples/speech_synthesis_examples.py](examples/speech_synthesis_examples.py)
- 🇰🇷 **한국어 TTS 데모**: [examples/korean_tts_demo.py](examples/korean_tts_demo.py)
- ⚡ **실시간 처리**: [examples/real_time_processing.py](examples/real_time_processing.py)

## 🛠️ 개발 참여

### 개발 환경 설정

```bash
# 개발 환경 설정
git clone https://github.com/crlotwhite/pytorch_hmm.git
cd pytorch_hmm
pip install -e .[dev]

# 코드 품질 검사
pre-commit install
black pytorch_hmm tests
isort pytorch_hmm tests
flake8 pytorch_hmm tests
mypy pytorch_hmm

# 개발 의존성 설치
pip install -e ".[dev]"

# pre-commit 훅 설정
pre-commit install

# 테스트 실행
pytest tests/ -v
```

## 📈 로드맵

### v0.3.0 (예정)
- 🎭 **감정 음성 합성**: 감정 상태 기반 HMM
- 🌐 **다국어 지원**: 영어, 중국어, 일본어 음소 집합
- 🔊 **화자 적응**: 다중 화자 모델링
- 🎯 **고급 정렬**: DTW와 HMM 결합 정렬

### v1.0.0 (목표)
- 🏭 **프로덕션 안정성**: API 고정 및 하위 호환성
- 📦 **패키지 생태계**: Hugging Face, PyTorch Lightning 통합
- 🚀 **배포 최적화**: ONNX, TensorRT 지원
- 📚 **완전한 문서화**: 종합 가이드 및 튜토리얼

## 🤝 기여하기

PyTorch HMM 프로젝트에 기여해주세요!

- 🐛 **버그 리포트**: [Issues](https://github.com/crlotwhite/pytorch_hmm/issues)
- 💡 **기능 제안**: [Discussions](https://github.com/crlotwhite/pytorch_hmm/discussions)
- 🔧 **Pull Request**: [Contributing Guide](CONTRIBUTING.md)

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- **PyTorch 팀**: 훌륭한 딥러닝 프레임워크
- **음성 처리 커뮤니티**: 귀중한 피드백과 제안
- **모든 기여자들**: 코드, 문서, 테스트 기여

### 기여 가이드라인

1. **이슈 생성**: 새 기능이나 버그 리포트
2. **Fork & Branch**: `feature/your-feature-name`
3. **테스트 작성**: 새 기능에 대한 테스트 추가
4. **문서화**: docstring과 README 업데이트
5. **Pull Request**: 상세한 설명과 함께 제출

## 📊 버전 히스토리 및 로드맵

### ✅ v0.2.0 (현재) - Advanced Features
- ✅ Neural HMM with contextual modeling
- ✅ Hidden Semi-Markov Model (HSMM)
- ✅ DTW and CTC alignment algorithms
- ✅ Comprehensive evaluation metrics
- ✅ Performance optimization and GPU acceleration
- ✅ Production-ready features

### 🔄 v0.3.0 (계획) - Real-world Integration
- [ ] LibriSpeech/KSS 데이터셋 지원
- [ ] ONNX 내보내기 및 양자화
- [ ] 실시간 마이크 입력 처리
- [ ] Attention-based alignment
- [ ] 멀티화자 모델링

### 🎯 v1.0.0 (목표) - Production Ready
- [ ] End-to-end TTS 파이프라인
- [ ] C++ 추론 엔진
- [ ] 자동 하이퍼파라미터 튜닝
- [ ] 완전한 문서화 및 API 안정화

## 📚 참고 자료

### 핵심 논문
- Rabiner, L. R. (1989). "A tutorial on hidden Markov models"
- Yu, K., et al. (2010). "Semi-Markov models for speech synthesis"
- Cuturi, M., & Blondel, M. (2017). "Soft-DTW: a Differentiable Loss Function for Time-Series"
- Graves, A., et al. (2006). "Connectionist temporal classification"

### 실무 활용 사례
- 음성 합성 시스템에서의 음소 정렬
- 음성 인식에서의 CTC 디코딩
- 화자 적응을 위한 DTW 정렬
- 지속시간 모델링을 통한 자연스러운 TTS

## 🙋‍♂️ 지원 및 커뮤니티

- **GitHub Issues**: [버그 리포트 및 기능 요청](https://github.com/crlotwhite/pytorch_hmm/issues)
- **GitHub Discussions**: [일반 질문 및 토론](https://github.com/crlotwhite/pytorch_hmm/discussions)
- **Documentation**: [상세 문서](https://github.com/crlotwhite/pytorch_hmm/wiki)

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🎉 v0.2.0 업데이트 하이라이트

이번 **v0.2.0** 업데이트는 PyTorch HMM 라이브러리를 **연구용 도구에서 프로덕션 레디 솔루션**으로 크게 발전시켰습니다:

🧠 **Neural HMM**: 컨텍스트 인식 모델링으로 기존 HMM의 한계 극복
⏱️ **Semi-Markov HMM**: 명시적 지속시간 모델링으로 자연스러운 음성 합성
🎯 **고급 정렬**: DTW와 CTC로 다양한 정렬 needs 지원
📊 **종합 평가**: MCD, F0 RMSE 등 표준 음성 평가 메트릭
🚀 **실시간 성능**: GPU 가속으로 실시간 음성 처리 가능

⭐ **이 프로젝트가 도움이 되었다면 Star를 눌러주세요!** ⭐
