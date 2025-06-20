# PyTorch HMM

🎯 **Advanced PyTorch Hidden Markov Model Library for Speech Synthesis**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/crlotwhite/pytorch_hmm)

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

```bash
# 기본 설치
pip install pytorch-hmm

# 모든 기능 포함
pip install pytorch-hmm[all]

# 개발 버전
git clone https://github.com/crlotwhite/pytorch_hmm.git
cd pytorch_hmm
pip install -e ".[all]"
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

## 🤝 기여하기

### 개발 환경 설정

```bash
git clone https://github.com/crlotwhite/pytorch_hmm.git
cd pytorch_hmm

# 개발 의존성 설치
pip install -e ".[dev]"

# pre-commit 훅 설정
pre-commit install

# 테스트 실행
pytest tests/ -v
```

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

⭐ **이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**

🔗 **관련 프로젝트**: [Original TensorFlow HMM](https://github.com/crlotwhite/tensorflow_hmm)
