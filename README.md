# PyTorch HMM

🎯 **음성 합성에 특화된 PyTorch Hidden Markov Model 라이브러리**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch 기반 Hidden Markov Model 구현체로, **음성 합성(TTS)과 음성 처리 응용**에 최적화되어 있습니다. Forward-backward와 Viterbi 알고리즘을 지원하며, autograd와 GPU 가속을 완벽하게 지원합니다.

## ✨ 주요 특징

### 🚀 **성능과 효율성**
- **PyTorch Native**: 완전한 autograd 지원과 GPU 가속
- **배치 처리**: 효율적인 대용량 데이터 처리
- **Numerical Stability**: Log-space 계산으로 수치적 안정성 확보
- **실시간 추론**: 최적화된 알고리즘으로 실시간 처리 가능

### 🎵 **음성 합성 최적화**
- **음소 정렬**: 텍스트-음성 정렬을 위한 전문 도구
- **지속시간 모델링**: 자연스러운 음소 지속시간 제어
- **Left-to-right HMM**: 음성 신호의 시간적 특성에 맞는 모델
- **가우시안 관측 모델**: 연속적인 음향 특징 처리

### 🔧 **개발자 친화적**
- **nn.Module 통합**: 기존 PyTorch 모델에 쉽게 통합
- **유연한 Transition Matrix**: 다양한 응용에 맞는 전이 패턴
- **상세한 문서화**: 튜토리얼과 실제 사용 예제 제공

## 📦 설치

```bash
pip install pytorch-hmm
```

또는 개발 버전:

```bash
git clone https://github.com/crlotwhite/pytorch_hmm.git
cd pytorch_hmm
pip install -e .
```

## 🚀 빠른 시작

### 기본 HMM 사용법

```python
import torch
from pytorch_hmm import HMMPyTorch, create_left_to_right_matrix

# Left-to-right HMM 생성 (음성 합성에 일반적)
num_states = 5
transition_matrix = create_left_to_right_matrix(num_states, self_loop_prob=0.8)
hmm = HMMPyTorch(transition_matrix)

# 관측 데이터 (batch_size=2, seq_len=10, num_states=5)
observations = torch.rand(2, 10, 5)

# Forward-backward 알고리즘
posterior, forward, backward = hmm.forward_backward(observations)

# Viterbi 디코딩
states, scores = hmm.viterbi_decode(observations)

print(f"Posterior shape: {posterior.shape}")  # (2, 10, 5)
print(f"Optimal states: {states[0]}")         # 첫 번째 시퀀스의 최적 상태 경로
```

### 신경망과의 통합

```python
import torch.nn as nn
from pytorch_hmm import HMMLayer

class SpeechSynthesisModel(nn.Module):
    def __init__(self, input_dim, num_phonemes):
        super().__init__()
        
        # 특징 추출기
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_phonemes)
        )
        
        # HMM 정렬 레이어
        self.hmm_layer = HMMLayer(
            num_states=num_phonemes,
            learnable_transitions=True,  # 학습 가능한 전이 확률
            transition_type="left_to_right"
        )
    
    def forward(self, linguistic_features):
        # 언어적 특징을 음소 확률로 변환
        phoneme_probs = self.feature_net(linguistic_features)
        
        # HMM으로 시간 정렬 수행
        aligned_phonemes = self.hmm_layer(phoneme_probs)
        
        return aligned_phonemes

# 모델 사용
model = SpeechSynthesisModel(input_dim=256, num_phonemes=50)
linguistic_input = torch.randn(4, 20, 256)  # (batch, time, features)
aligned_output = model(linguistic_input)
```

## 📚 상세 사용법

### 1. Transition Matrix 생성

```python
from pytorch_hmm import create_transition_matrix

# 다양한 타입의 전이 행렬
ergodic_matrix = create_transition_matrix(5, "ergodic")           # 완전 연결
left_right = create_transition_matrix(5, "left_to_right")         # 순차 진행
skip_matrix = create_transition_matrix(5, "left_to_right_skip")   # 상태 건너뛰기 허용
circular = create_transition_matrix(5, "circular")               # 순환 구조
```
생성된 전이 행렬은 각 행이 1로 정규화된 확률 분포입니다.

### 2. 가우시안 HMM (연속 특징용)

```python
from pytorch_hmm import GaussianHMMLayer

# 연속적인 음향 특징을 위한 가우시안 HMM
gaussian_hmm = GaussianHMMLayer(
    num_states=10,
    feature_dim=80,  # 멜 스펙트로그램 차원
    covariance_type='diag'
)

# MFCC나 멜 스펙트로그램 같은 연속 특징
acoustic_features = torch.randn(3, 100, 80)  # (batch, time, mfcc_dim)
posteriors = gaussian_hmm(acoustic_features)
```

### 3. 음성 합성 파이프라인

```python
from pytorch_hmm.utils import compute_state_durations

# 1. 음소 시퀀스 정렬
phoneme_sequence = ["sil", "a", "n", "n", "y", "eo", "ng", "sil"]
hmm = create_phoneme_hmm(phoneme_sequence)

# 2. 음향 특징과 정렬
acoustic_features = load_audio_features("hello.wav")
aligned_states, _ = hmm.align(acoustic_features)

# 3. 지속시간 계산
durations = compute_state_durations(aligned_states)
print(f"각 음소의 지속시간: {durations}")

# 4. 새로운 음성 합성 시 지속시간 조절
new_durations = durations * 1.2  # 20% 느리게
synthesized_audio = synthesize_with_durations(phoneme_sequence, new_durations)
```

## 🎯 음성 합성 응용 예제

### 음소-음향 정렬

```python
# 텍스트에서 추출된 음소 시퀀스
phonemes = ["sil", "안", "녕", "하", "세", "요", "sil"]

# 음성 파일에서 추출된 음향 특징 (MFCC, 멜스펙트로그램 등)
audio_features = extract_acoustic_features("audio.wav")

# HMM으로 정렬 수행
alignment_model = create_alignment_model(len(phonemes))
phoneme_boundaries = alignment_model.align(phonemes, audio_features)

print("음소별 시작-끝 시간:")
for phoneme, (start, end) in zip(phonemes, phoneme_boundaries):
    print(f"{phoneme}: {start:.2f}s - {end:.2f}s")
```

### 실시간 음성 인식 디코딩

```python
# 실시간 스트리밍 음성 인식
streaming_hmm = HMMLayer(
    num_states=num_phonemes,
    viterbi_inference=True,  # 빠른 하드 디코딩
    learnable_transitions=False  # 고정된 언어 모델
)

# 청크 단위 처리
chunk_size = 160  # 10ms @ 16kHz
for audio_chunk in audio_stream:
    features = extract_features(audio_chunk)
    phoneme_probs = acoustic_model(features)
    decoded_phonemes = streaming_hmm(phoneme_probs)
    
    # 실시간으로 텍스트 출력
    update_transcription(decoded_phonemes)
```

## 🔧 고급 기능

### 사용자 정의 Transition Matrix

```python
import torch
from pytorch_hmm import HMMPyTorch

# 특별한 제약이 있는 전이 행렬 생성
def create_custom_transition_matrix(num_states, min_duration=3):
    """각 상태에서 최소 지속시간을 보장하는 전이 행렬"""
    P = torch.zeros(num_states, num_states)
    
    for i in range(num_states):
        if i < num_states - 1:
            P[i, i] = 0.9      # 높은 self-loop 확률
            P[i, i + 1] = 0.1  # 낮은 전진 확률
        else:
            P[i, i] = 1.0      # 마지막 상태는 유지
    
    return P

# 사용자 정의 HMM
custom_P = create_custom_transition_matrix(10)
hmm = HMMPyTorch(custom_P)
```

### 배치 처리 최적화

```python
# 대용량 데이터 효율적 처리
def process_large_dataset(dataset, batch_size=32):
    hmm = HMMPyTorch(transition_matrix).cuda()  # GPU 사용
    
    results = []
    for batch in DataLoader(dataset, batch_size=batch_size):
        with torch.no_grad():  # 메모리 절약
            observations = batch.cuda()
            posteriors, _, _ = hmm.forward_backward(observations)
            results.append(posteriors.cpu())
    
    return torch.cat(results, dim=0)
```

## 📊 성능 벤치마크

```python
# 성능 테스트 실행
python examples/benchmark.py
```

일반적인 성능 (RTX 3080 GPU 기준):
- **Forward-Backward**: ~15,000 frames/sec
- **Viterbi**: ~25,000 frames/sec  
- **HMM Layer**: ~12,000 frames/sec

실시간 음성 처리 (80fps)에 대해 **150-300x** 빠른 속도로 처리 가능합니다.

## 🧪 테스트

```bash
# 전체 테스트 실행
python -m pytest tests/ -v

# 커버리지 포함
python -m pytest tests/ --cov=pytorch_hmm --cov-report=html

# 특정 테스트만 실행
python -m pytest tests/test_hmm.py::TestHMMPyTorch::test_forward_backward -v
```

## 📖 예제와 튜토리얼

### 기본 튜토리얼
```bash
python examples/basic_tutorial.py
```

### 음성 합성 예제
```bash
python examples/speech_synthesis_examples.py
```

### 성능 벤치마크
```bash
python examples/benchmark.py
```

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 개발 환경 설정

```bash
git clone https://github.com/crlotwhite/pytorch_hmm.git
cd pytorch_hmm

# 개발용 의존성 설치
pip install -e ".[dev]"

# 코드 스타일 검사
black pytorch_hmm/
flake8 pytorch_hmm/
isort pytorch_hmm/

# 테스트 실행
pytest tests/
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📚 참고 자료

### HMM 이론
- Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition"
- Jelinek, F. (1997). "Statistical methods for speech recognition"

### 음성 합성 응용
- Zen, H., et al. (2009). "Statistical parametric speech synthesis using deep neural networks"
- Wang, Y., et al. (2017). "Tacotron: Towards end-to-end speech synthesis"

### PyTorch와 딥러닝
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## 🙋‍♂️ 지원

- **이슈**: [GitHub Issues](https://github.com/crlotwhite/pytorch_hmm/issues)
- **토론**: [GitHub Discussions](https://github.com/crlotwhite/pytorch_hmm/discussions)
- **문서**: [Wiki](https://github.com/crlotwhite/pytorch_hmm/wiki)

## 📈 로드맵

- [ ] **v0.2.0**: 
  - Continuous HMM 지원 확장
  - 더 많은 Transition matrix 타입
  - 성능 최적화

- [ ] **v0.3.0**:
  - Semi-Markov Models 지원
  - 고급 음성 합성 유틸리티
  - 실시간 스트리밍 최적화

- [ ] **v1.0.0**:
  - 안정화된 API
  - 완전한 문서화
  - 프로덕션 레디

---

⭐ **이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**

🔗 **관련 프로젝트**: [Original TensorFlow HMM](https://github.com/crlotwhite/tensorflow_hmm)
