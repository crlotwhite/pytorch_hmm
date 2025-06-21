# ❓ 자주 묻는 질문 (FAQ)

PyTorch HMM 사용 시 자주 묻는 질문들과 해답을 정리했습니다.

## 📦 설치 및 환경 설정

### Q1. PyTorch HMM 설치 시 오류가 발생합니다.

**A:** 다음 단계를 순서대로 시도해보세요:

```bash
# 1. Python 버전 확인 (3.8+ 필요)
python --version

# 2. PyTorch 버전 확인 (1.9+ 필요)
python -c "import torch; print(torch.__version__)"

# 3. 캐시 정리 후 재설치
pip cache purge
pip uninstall pytorch-hmm
pip install -e . --force-reinstall

# 4. uv 사용 (권장)
pip install uv
uv sync
```

### Q2. CUDA를 사용할 수 없다고 나옵니다.

**A:** CUDA 설정을 확인해보세요:

```python
import torch

print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"GPU 개수: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"현재 GPU: {torch.cuda.get_device_name()}")
else:
    print("CUDA 드라이버나 PyTorch CUDA 버전을 확인하세요")
```

**해결 방법:**
- NVIDIA 드라이버 최신 버전 설치
- PyTorch CUDA 버전과 시스템 CUDA 버전 일치 확인
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Q3. 메모리 부족 오류가 발생합니다.

**A:** 메모리 사용량을 줄여보세요:

```python
# 배치 크기 줄이기
batch_size = 8  # 32 대신

# 그래디언트 비활성화
with torch.no_grad():
    result = hmm.forward(observations)

# GPU 메모리 정리
torch.cuda.empty_cache()

# 메모리 사용량 확인
if torch.cuda.is_available():
    print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
```

## 🔧 모델 사용

### Q4. HMM 모델의 상태 수를 어떻게 결정하나요?

**A:** 상태 수는 데이터의 특성에 따라 결정됩니다:

- **음성 합성**: 음소 수 (보통 40-60개)
- **음성 인식**: 단어나 음소 단위 (데이터에 따라)
- **일반 시계열**: 데이터의 패턴 복잡도에 따라

```python
# 음성 합성 예제
num_phonemes = 50  # 영어 음소 수
hmm = HMMPyTorch(num_states=num_phonemes)

# 교차 검증으로 최적 상태 수 찾기
def find_optimal_states(data, state_range=(5, 20)):
    best_score = float('-inf')
    best_states = 5
    
    for num_states in range(*state_range):
        hmm = HMMPyTorch(num_states=num_states)
        score = hmm.score(data)  # 로그 우도
        
        if score > best_score:
            best_score = score
            best_states = num_states
    
    return best_states
```

### Q5. Forward-backward와 Viterbi의 차이점은?

**A:** 두 알고리즘의 목적이 다릅니다:

| 알고리즘 | 목적 | 출력 | 사용 시기 |
|----------|------|------|-----------|
| **Forward-backward** | 각 시점의 상태 확률 계산 | 사후 확률 분포 | 학습, 확률적 추론 |
| **Viterbi** | 최적 상태 시퀀스 찾기 | 단일 상태 시퀀스 | 디코딩, 정렬 |

```python
# Forward-backward: 확률 분포
posteriors, log_likelihood = hmm.forward_backward(observations)
print(f"각 시점의 상태 확률: {posteriors.shape}")  # [batch, time, states]

# Viterbi: 최적 경로
best_path, score = hmm.viterbi_decode(observations)
print(f"최적 상태 시퀀스: {best_path.shape}")  # [batch, time]
```

### Q6. MixtureGaussianHMM의 mixture 수는 어떻게 정하나요?

**A:** Mixture 수는 데이터 복잡도에 따라 결정됩니다:

```python
# 일반적인 가이드라인
data_complexity_guide = {
    "간단한 데이터": 1,      # 단일 가우시안
    "중간 복잡도": 2-4,      # 일반적인 음성 데이터
    "복잡한 데이터": 5-8,    # 노이즈가 많은 환경
    "매우 복잡한 데이터": 8+  # 다중 화자, 다양한 환경
}

# BIC(Bayesian Information Criterion)로 최적값 찾기
def find_optimal_mixtures(data, max_mixtures=8):
    bic_scores = []
    
    for n_mix in range(1, max_mixtures + 1):
        hmm = MixtureGaussianHMM(
            num_states=5, 
            obs_dim=data.size(-1), 
            num_mixtures=n_mix
        )
        
        # 간단한 학습
        for _ in range(10):
            hmm.forward(data)
        
        # BIC 계산 (간소화된 버전)
        log_likelihood = hmm.forward(data).sum()
        n_params = n_mix * data.size(-1) * 2  # 평균, 분산
        bic = -2 * log_likelihood + n_params * torch.log(torch.tensor(data.numel()))
        bic_scores.append(bic.item())
    
    optimal_mixtures = bic_scores.index(min(bic_scores)) + 1
    return optimal_mixtures
```

## 🚀 성능 최적화

### Q7. 실시간 처리가 너무 느립니다.

**A:** 다음 최적화 방법들을 시도해보세요:

```python
# 1. 배치 크기 최적화
def find_optimal_batch_size():
    for batch_size in [1, 2, 4, 8, 16, 32]:
        try:
            data = torch.randn(batch_size, 1000, 80, device='cuda')
            start_time = time.time()
            _ = hmm.forward(data)
            end_time = time.time()
            
            latency = (end_time - start_time) / batch_size * 1000
            print(f"배치 크기 {batch_size}: {latency:.2f}ms/sample")
            
            if latency < 10:  # 10ms 이하면 실시간 가능
                return batch_size
        except RuntimeError:
            continue
    return 1

# 2. Mixed precision 사용
with torch.cuda.amp.autocast():
    result = hmm.forward(observations)

# 3. 청크 크기 조정
streaming_processor = StreamingHMMProcessor(
    hmm_model=hmm,
    chunk_size=80,    # 더 작은 청크 (5ms)
    overlap=40        # 오버랩 줄이기
)
```

### Q8. GPU 메모리를 더 효율적으로 사용하려면?

**A:** 메모리 최적화 기법들을 적용하세요:

```python
# 1. 인플레이스 연산 사용
def memory_efficient_forward(hmm, observations):
    # 메모리 미리 할당
    batch_size, seq_len, obs_dim = observations.shape
    result = torch.empty(batch_size, seq_len, hmm.num_states, 
                        device=observations.device)
    
    # 청크 단위 처리
    chunk_size = 100
    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        chunk = observations[:, i:end_idx]
        
        with torch.no_grad():
            result[:, i:end_idx] = hmm.forward(chunk)
    
    return result

# 2. 그래디언트 체크포인팅
import torch.utils.checkpoint as checkpoint

def checkpointed_forward(hmm, observations):
    return checkpoint.checkpoint(hmm.forward, observations)

# 3. 메모리 모니터링
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"메모리 사용량: {allocated:.2f}GB / {reserved:.2f}GB")
```

## 🎵 음성 처리

### Q9. 음성 데이터 전처리는 어떻게 하나요?

**A:** 음성 데이터 전처리 파이프라인:

```python
import torchaudio
import torchaudio.transforms as transforms

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mels=80):
        self.sample_rate = sample_rate
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        self.log_transform = transforms.AmplitudeToDB()
    
    def preprocess(self, audio_path):
        # 오디오 로드
        waveform, sr = torchaudio.load(audio_path)
        
        # 리샘플링
        if sr != self.sample_rate:
            resampler = transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 멜 스펙트로그램 변환
        mel_spec = self.mel_transform(waveform)
        log_mel = self.log_transform(mel_spec)
        
        # 정규화
        log_mel = (log_mel - log_mel.mean()) / log_mel.std()
        
        return log_mel.transpose(-1, -2)  # [time, mel_dim]

# 사용 예제
preprocessor = AudioPreprocessor()
features = preprocessor.preprocess("audio.wav")
```

### Q10. TTS 시스템에서 정렬 품질을 높이려면?

**A:** 정렬 품질 향상 방법들:

```python
# 1. 전이 행렬 제약 추가
def create_constrained_transitions(num_states, self_loop_prob=0.8):
    """Left-to-right 제약이 있는 전이 행렬"""
    P = torch.zeros(num_states, num_states)
    
    for i in range(num_states):
        if i == num_states - 1:  # 마지막 상태
            P[i, i] = 1.0
        else:
            P[i, i] = self_loop_prob        # 자기 자신
            P[i, i + 1] = 1 - self_loop_prob  # 다음 상태
    
    return P

# 2. 지속시간 모델링 추가
from pytorch_hmm import SemiMarkovHMM

duration_hmm = SemiMarkovHMM(
    num_states=num_phonemes,
    obs_dim=80,
    max_duration=10  # 최대 지속시간
)

# 3. 정렬 정확도 평가
def evaluate_alignment(predicted_alignment, ground_truth):
    """정렬 정확도 계산"""
    correct = (predicted_alignment == ground_truth).float()
    accuracy = correct.mean()
    
    # 경계 정확도 (음소 경계에서의 정확도)
    boundaries = (ground_truth[1:] != ground_truth[:-1]).nonzero()
    boundary_accuracy = correct[boundaries].mean() if len(boundaries) > 0 else 0
    
    return {
        'frame_accuracy': accuracy.item(),
        'boundary_accuracy': boundary_accuracy.item() if isinstance(boundary_accuracy, torch.Tensor) else boundary_accuracy
    }
```

## 🔬 고급 사용법

### Q11. 커스텀 관측 모델을 만들려면?

**A:** 사용자 정의 관측 모델 구현:

```python
import torch.nn as nn

class CustomEmissionModel(nn.Module):
    """커스텀 관측 모델 예제"""
    
    def __init__(self, num_states, obs_dim, hidden_dim=256):
        super().__init__()
        self.num_states = num_states
        self.obs_dim = obs_dim
        
        # 각 상태별 신경망
        self.state_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_states)
        ])
    
    def forward(self, observations):
        """관측 확률 계산"""
        batch_size, seq_len, _ = observations.shape
        log_probs = torch.zeros(batch_size, seq_len, self.num_states)
        
        for state in range(self.num_states):
            state_log_probs = self.state_networks[state](observations)
            log_probs[:, :, state] = state_log_probs.squeeze(-1)
        
        return log_probs

# HMM과 통합
class CustomHMM(nn.Module):
    def __init__(self, num_states, obs_dim):
        super().__init__()
        self.emission_model = CustomEmissionModel(num_states, obs_dim)
        self.transition_matrix = nn.Parameter(
            torch.randn(num_states, num_states)
        )
    
    def forward(self, observations):
        emission_probs = self.emission_model(observations)
        # Forward-backward 알고리즘 적용
        # ... (구현 생략)
        return emission_probs
```

### Q12. 다중 GPU에서 학습하려면?

**A:** 분산 학습 설정:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. 분산 환경 초기화
def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

# 2. 모델을 DDP로 래핑
def create_distributed_model(hmm):
    device = torch.device(f'cuda:{dist.get_rank()}')
    hmm = hmm.to(device)
    ddp_hmm = DDP(hmm, device_ids=[dist.get_rank()])
    return ddp_hmm

# 3. 분산 데이터 로더
from torch.utils.data.distributed import DistributedSampler

def create_distributed_dataloader(dataset, batch_size):
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=4
    )
    return dataloader

# 사용 예제
if __name__ == "__main__":
    setup_distributed()
    
    hmm = MixtureGaussianHMM(num_states=50, obs_dim=80)
    ddp_hmm = create_distributed_model(hmm)
    
    # 학습 루프
    for batch in distributed_dataloader:
        loss = ddp_hmm(batch)
        loss.backward()
        optimizer.step()
```

## 🚨 문제 해결

### Q13. "RuntimeError: CUDA out of memory" 오류 해결법?

**A:** 메모리 부족 오류 해결 단계:

```python
# 1. 메모리 사용량 확인
def check_memory_usage():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        print(f"총 GPU 메모리: {total_memory:.2f}GB")
        print(f"할당된 메모리: {allocated:.2f}GB")
        print(f"예약된 메모리: {reserved:.2f}GB")
        print(f"사용 가능: {total_memory - reserved:.2f}GB")

# 2. 메모리 최적화 함수
def optimize_memory_usage():
    # 캐시 정리
    torch.cuda.empty_cache()
    
    # 가비지 컬렉션
    import gc
    gc.collect()
    
    # 메모리 단편화 최소화
    torch.cuda.memory._record_memory_history(enabled=None)

# 3. 배치 크기 자동 조정
def auto_adjust_batch_size(model, data_shape, max_batch_size=64):
    for batch_size in [max_batch_size // (2**i) for i in range(6)]:
        try:
            dummy_data = torch.randn(batch_size, *data_shape[1:], device='cuda')
            with torch.no_grad():
                _ = model(dummy_data)
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    return 1
```

### Q14. 모델 저장/로드 시 주의사항은?

**A:** 모델 저장/로드 베스트 프랙티스:

```python
# 1. 안전한 모델 저장
def save_model_safely(model, path, metadata=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_states': model.num_states,
            'obs_dim': getattr(model, 'obs_dim', None),
            'num_mixtures': getattr(model, 'num_mixtures', None)
        },
        'pytorch_version': torch.__version__,
        'metadata': metadata or {}
    }
    
    # 원자적 저장 (임시 파일 사용)
    temp_path = path + '.tmp'
    torch.save(checkpoint, temp_path)
    
    # 저장 성공 시 원본 파일로 이동
    import os
    os.rename(temp_path, path)
    print(f"모델이 {path}에 저장되었습니다.")

# 2. 안전한 모델 로드
def load_model_safely(model_class, path, device='cpu'):
    try:
        checkpoint = torch.load(path, map_location=device)
        
        # 설정 정보로 모델 생성
        config = checkpoint['model_config']
        model = model_class(**config)
        
        # 상태 딕셔너리 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"모델이 {path}에서 로드되었습니다.")
        print(f"PyTorch 버전: {checkpoint.get('pytorch_version', 'Unknown')}")
        
        return model
    
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None

# 3. 버전 호환성 확인
def check_compatibility(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_version = checkpoint.get('pytorch_version', 'Unknown')
    current_version = torch.__version__
    
    if saved_version != current_version:
        print(f"⚠️ PyTorch 버전 불일치: 저장됨({saved_version}) vs 현재({current_version})")
        print("모델 동작에 문제가 있을 수 있습니다.")
    else:
        print("✅ PyTorch 버전 호환")
```

## 📞 추가 도움말

### 더 많은 정보가 필요하다면:

1. **[GitHub Issues](https://github.com/crlotwhite/pytorch_hmm/issues)** - 버그 신고 및 기능 요청
2. **[GitHub Discussions](https://github.com/crlotwhite/pytorch_hmm/discussions)** - 사용법 질문
3. **[문서](../README.md)** - 상세 문서 및 튜토리얼
4. **[예제 코드](../examples/)** - 실제 사용 예제들

### 질문하기 전 체크리스트:

- [ ] 최신 버전을 사용하고 있나요?
- [ ] 에러 메시지 전문을 확인했나요?
- [ ] 비슷한 이슈가 이미 있는지 검색했나요?
- [ ] 최소한의 재현 가능한 예제를 준비했나요?

---

**이 FAQ에서 답을 찾지 못했다면 언제든 GitHub Issues에 질문해주세요! 🤝** 