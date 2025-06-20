# 🎯 PyTorch HMM v0.2.0 - 음성 합성에 특화된 Hidden Markov Model

[![CI](https://github.com/crlotwhite/pytorch_hmm/workflows/CI/badge.svg)](https://github.com/crlotwhite/pytorch_hmm/actions)
[![codecov](https://codecov.io/gh/crlotwhite/pytorch_hmm/branch/main/graph/badge.svg)](https://codecov.io/gh/crlotwhite/pytorch_hmm)
[![PyPI version](https://badge.fury.io/py/pytorch-hmm.svg)](https://badge.fury.io/py/pytorch-hmm)
[![Python versions](https://img.shields.io/pypi/pyversions/pytorch-hmm.svg)](https://pypi.org/project/pytorch-hmm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## 📦 설치

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

## 🚀 빠른 시작

### 기본 HMM 사용법

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
```

## 📊 성능 벤치마크

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
```

## 📚 문서 및 예시

- 📖 **API 문서**: [pytorch-hmm.readthedocs.io](https://pytorch-hmm.readthedocs.io)
- 💡 **튜토리얼**: [examples/](examples/) 디렉토리
- 🎵 **음성 처리 예시**: [examples/speech_synthesis_examples.py](examples/speech_synthesis_examples.py)
- 🇰🇷 **한국어 TTS 데모**: [examples/korean_tts_demo.py](examples/korean_tts_demo.py)
- ⚡ **실시간 처리**: [examples/real_time_processing.py](examples/real_time_processing.py)

## 🛠️ 개발 참여

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

---

⭐ **이 프로젝트가 도움이 되었다면 Star를 눌러주세요!** ⭐
