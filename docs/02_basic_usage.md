# 2. 기본 사용법

이 문서에서는 PyTorch HMM 라이브러리의 기본 사용법을 단계별로 학습합니다. 설치부터 첫 번째 모델 생성, 추론, 신경망 통합까지 실제 코드와 함께 설명합니다.

## 📚 목차

1. [설치 및 설정](#1-설치-및-설정)
2. [첫 번째 HMM 모델](#2-첫-번째-hmm-모델)
3. [Forward-Backward vs Viterbi](#3-forward-backward-vs-viterbi)
4. [신경망과의 통합](#4-신경망과의-통합)
5. [배치 처리](#5-배치-처리)
6. [GPU 사용법](#6-gpu-사용법)
7. [실제 음성 데이터 예제](#7-실제-음성-데이터-예제)

## 1. 설치 및 설정

### 1.1 시스템 요구사항

```bash
# Python 버전 확인
python --version  # Python 3.8+ 필요

# PyTorch 설치 (CUDA 지원)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 기본 의존성
pip install numpy matplotlib scipy
```

### 1.2 PyTorch HMM 설치

```bash
# 개발 버전 설치 (권장)
git clone https://github.com/your-repo/pytorch_hmm.git
cd pytorch_hmm
pip install -e .

# 또는 PyPI에서 설치 (향후)
# pip install pytorch-hmm
```

### 1.3 설치 확인

```python
import torch
from pytorch_hmm import HMMPyTorch, HMMLayer, create_left_to_right_matrix

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print("PyTorch HMM 라이브러리 정상 설치됨!")
```

## 2. 첫 번째 HMM 모델

### 2.1 기본 HMM 생성

```python
import torch
from pytorch_hmm import HMMPyTorch, create_left_to_right_matrix

# 1. 전이 행렬 생성
num_states = 5
transition_matrix = create_left_to_right_matrix(
    num_states, 
    self_loop_prob=0.7  # 자기 자신으로의 전이 확률
)

print("전이 행렬:")
print(transition_matrix)
```

**출력:**
```
전이 행렬:
tensor([[0.7000, 0.3000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.7000, 0.3000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.7000, 0.3000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.7000, 0.3000],
        [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]])
```

### 2.2 HMM 모델 초기화

```python
# 2. HMM 모델 생성
hmm = HMMPyTorch(transition_matrix)

print(f"상태 수: {hmm.K}")
print(f"디바이스: {hmm.device}")
print(f"전이 행렬 형태: {hmm.A.shape}")
```

### 2.3 관측 데이터 준비

```python
# 3. 관측 데이터 생성 (예: 음성 특징)
batch_size = 2
seq_len = 100
feature_dim = num_states  # 단순화를 위해 상태 수와 동일

# 확률적 관측 데이터 (각 시점에서 각 상태에 대한 확률)
observations = torch.softmax(
    torch.randn(batch_size, seq_len, num_states), 
    dim=-1
)

print(f"관측 데이터 형태: {observations.shape}")
print(f"확률 합 확인: {observations[0, 0].sum()}")  # 1.0이어야 함
```

### 2.4 기본 추론

```python
# 4. Forward-backward 알고리즘
posteriors, forward, backward = hmm.forward_backward(observations)

print(f"후방확률 형태: {posteriors.shape}")
print(f"첫 번째 시점 후방확률 합: {posteriors[0, 0].sum()}")

# 5. Viterbi 디코딩
optimal_states, scores = hmm.viterbi_decode(observations)

print(f"최적 상태 시퀀스 형태: {optimal_states.shape}")
print(f"첫 10개 상태: {optimal_states[0, :10].tolist()}")
```

## 3. Forward-Backward vs Viterbi

두 알고리즘의 차이점과 사용 시기를 실제 예제로 비교해보겠습니다.

### 3.1 비교 실험 설정

```python
import time
import matplotlib.pyplot as plt

def compare_algorithms():
    # 더 긴 시퀀스로 테스트
    num_states = 6
    seq_len = 200
    
    P = create_left_to_right_matrix(num_states, self_loop_prob=0.8)
    hmm = HMMPyTorch(P)
    
    # 관측 데이터
    observations = torch.softmax(
        torch.randn(1, seq_len, num_states), dim=-1
    )
    
    print(f"테스트 설정: {num_states}개 상태, {seq_len} 길이 시퀀스")
    return hmm, observations

hmm, observations = compare_algorithms()
```

### 3.2 Forward-Backward 분석

```python
# Forward-Backward 실행
print("\n=== Forward-Backward 알고리즘 ===")
start_time = time.time()
posteriors, forward, backward = hmm.forward_backward(observations)
fb_time = time.time() - start_time

# 소프트 정렬 (가장 확률 높은 상태)
soft_alignment = torch.argmax(posteriors, dim=-1)[0]

print(f"실행 시간: {fb_time:.4f}초")
print(f"출력: 확률적 후방확률 (soft alignment)")
print(f"소프트 정렬 (처음 15개): {soft_alignment[:15].tolist()}")

# 불확실성 분석
uncertainty = -torch.sum(posteriors * torch.log(posteriors + 1e-8), dim=-1)
print(f"평균 불확실성: {uncertainty.mean():.3f}")
```

### 3.3 Viterbi 분석

```python
# Viterbi 실행
print("\n=== Viterbi 알고리즘 ===")
start_time = time.time()
hard_alignment, scores = hmm.viterbi_decode(observations)
viterbi_time = time.time() - start_time

print(f"실행 시간: {viterbi_time:.4f}초")
print(f"출력: 결정적 정렬 (hard alignment)")
print(f"하드 정렬 (처음 15개): {hard_alignment[0, :15].tolist()}")
print(f"최적 경로 점수: {scores[0]:.3f}")
```

### 3.4 결과 비교 및 분석

```python
# 정렬 일치도 분석
agreement = (soft_alignment == hard_alignment[0]).float().mean()
print(f"\n=== 비교 결과 ===")
print(f"속도 비율 (Viterbi/FB): {viterbi_time/fb_time:.2f}x")
print(f"정렬 일치도: {agreement:.3f} ({agreement*100:.1f}%)")

# 상태 지속시간 비교
from pytorch_hmm.utils import compute_state_durations

soft_durations = compute_state_durations(soft_alignment)
hard_durations = compute_state_durations(hard_alignment[0])

print(f"평균 지속시간 (soft): {soft_durations.float().mean():.2f}")
print(f"평균 지속시간 (hard): {hard_durations.float().mean():.2f}")
```

### 3.5 시각화

```python
def visualize_alignments(soft_align, hard_align, seq_len=50):
    """정렬 결과 시각화"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # 소프트 정렬
    ax1.plot(soft_align[:seq_len].numpy(), 'b-', linewidth=2, label='Soft Alignment')
    ax1.set_title('Forward-Backward (Soft Alignment)')
    ax1.set_ylabel('State')
    ax1.grid(True, alpha=0.3)
    
    # 하드 정렬
    ax2.plot(hard_align[:seq_len].numpy(), 'r-', linewidth=2, label='Hard Alignment')
    ax2.set_title('Viterbi (Hard Alignment)')
    ax2.set_ylabel('State')
    ax2.grid(True, alpha=0.3)
    
    # 차이점 표시
    diff = (soft_align != hard_align).float()[:seq_len]
    ax3.fill_between(range(seq_len), diff.numpy(), alpha=0.5, color='orange')
    ax3.set_title('Alignment Differences')
    ax3.set_xlabel('Time Frame')
    ax3.set_ylabel('Different')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 시각화 실행
visualize_alignments(soft_alignment, hard_alignment[0])
```

### 3.6 언제 어떤 알고리즘을 사용할까?

```python
print("\n=== 사용 가이드 ===")
print("📊 Forward-Backward를 사용하는 경우:")
print("   • 모델 학습 (gradient 계산 필요)")
print("   • 불확실성 정량화가 중요한 경우")
print("   • 여러 가능성을 고려한 소프트 결정")
print("   • 앙상블이나 융합에서 확률 정보 활용")

print("\n🎯 Viterbi를 사용하는 경우:")
print("   • 최종 추론 (명확한 결정 필요)")
print("   • 실시간 처리 (속도 우선)")
print("   • 메모리 효율성이 중요한 경우")
print("   • 명확한 경계 검출이 필요한 경우")
```

## 4. 신경망과의 통합

PyTorch HMM의 가장 강력한 기능 중 하나는 신경망과의 자연스러운 통합입니다.

### 4.1 HMMLayer 기본 사용법

```python
import torch.nn as nn
from pytorch_hmm import HMMLayer

class SimpleAlignmentModel(nn.Module):
    def __init__(self, input_dim, num_states, hidden_dim=128):
        super().__init__()
        
        # 특징 인코더
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # HMM 정렬 레이어
        self.hmm_layer = HMMLayer(
            num_states=num_states,
            learnable_transitions=True,  # 전이 확률 학습 가능
            transition_type="left_to_right",
            viterbi_inference=False  # 학습 시 soft alignment
        )
        
        # 출력 디코더
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x, return_alignment=False):
        # 1. 특징 인코딩
        encoded = self.encoder(x)
        
        # 2. HMM 정렬
        aligned_features, posteriors = self.hmm_layer(encoded)
        
        # 3. 디코딩
        output = self.decoder(aligned_features)
        
        if return_alignment:
            return output, posteriors
        return output

# 모델 생성 및 테스트
input_dim = 80  # 예: 멜 스펙트로그램 차원
num_states = 10  # 음소별 상태 수
model = SimpleAlignmentModel(input_dim, num_states)

# 테스트 데이터
batch_size, seq_len = 4, 150
test_input = torch.randn(batch_size, seq_len, input_dim)

# Forward pass
output, alignment = model(test_input, return_alignment=True)

print(f"입력 형태: {test_input.shape}")
print(f"출력 형태: {output.shape}")
print(f"정렬 형태: {alignment.shape}")
```

### 4.2 음성 합성 모델 예제

```python
class TTSAlignmentModel(nn.Module):
    """Text-to-Speech 정렬 모델"""
    
    def __init__(self, text_vocab_size, num_phonemes, mel_dim=80):
        super().__init__()
        
        # 텍스트 임베딩
        self.text_embedding = nn.Embedding(text_vocab_size, 256)
        
        # 텍스트 인코더
        self.text_encoder = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # HMM 정렬 (음소별)
        self.hmm_layers = nn.ModuleList([
            HMMLayer(
                num_states=3,  # 각 음소당 3개 상태
                learnable_transitions=True,
                transition_type="left_to_right"
            ) for _ in range(num_phonemes)
        ])
        
        # 멜 스펙트로그램 디코더
        self.mel_decoder = nn.Sequential(
            nn.Linear(512, 512),  # bidirectional LSTM output
            nn.ReLU(),
            nn.Linear(512, mel_dim)
        )
        
        self.num_phonemes = num_phonemes
    
    def forward(self, text_tokens, phoneme_ids, target_length=None):
        batch_size, text_len = text_tokens.shape
        
        # 1. 텍스트 인코딩
        embedded = self.text_embedding(text_tokens)
        encoded, _ = self.text_encoder(embedded)
        
        # 2. 음소별 HMM 정렬
        aligned_features = []
        all_alignments = []
        
        for i, phoneme_id in enumerate(phoneme_ids.unique()):
            # 해당 음소의 특징 추출
            phoneme_mask = (phoneme_ids == phoneme_id)
            if phoneme_mask.sum() == 0:
                continue
                
            phoneme_features = encoded[phoneme_mask]
            
            # HMM 정렬
            aligned, alignment = self.hmm_layers[phoneme_id](
                phoneme_features.unsqueeze(0)
            )
            
            aligned_features.append(aligned.squeeze(0))
            all_alignments.append(alignment.squeeze(0))
        
        # 3. 특징 연결
        if aligned_features:
            combined_features = torch.cat(aligned_features, dim=0)
        else:
            combined_features = encoded.mean(dim=1, keepdim=True)
        
        # 4. 멜 스펙트로그램 생성
        mel_output = self.mel_decoder(combined_features)
        
        return mel_output, all_alignments

# 사용 예제
vocab_size = 1000
num_phonemes = 40
model = TTSAlignmentModel(vocab_size, num_phonemes)

# 샘플 데이터
text_tokens = torch.randint(0, vocab_size, (2, 20))  # 배치 크기 2, 길이 20
phoneme_ids = torch.randint(0, num_phonemes, (20,))  # 음소 ID

mel_output, alignments = model(text_tokens, phoneme_ids)
print(f"멜 스펙트로그램 출력: {mel_output.shape}")
```

### 4.3 학습 가능한 전이 행렬

```python
class LearnableTransitionHMM(nn.Module):
    def __init__(self, num_states, feature_dim):
        super().__init__()
        
        # 학습 가능한 전이 행렬 파라미터
        self.transition_logits = nn.Parameter(
            torch.randn(num_states, num_states)
        )
        
        # 관측 모델
        self.observation_model = nn.Linear(feature_dim, num_states)
        
        self.num_states = num_states
    
    def get_transition_matrix(self):
        """소프트맥스로 정규화된 전이 행렬 반환"""
        return torch.softmax(self.transition_logits, dim=-1)
    
    def forward(self, features):
        # 1. 관측 확률 계산
        obs_logits = self.observation_model(features)
        obs_probs = torch.softmax(obs_logits, dim=-1)
        
        # 2. 전이 행렬 얻기
        transition_matrix = self.get_transition_matrix()
        
        # 3. HMM 생성 및 추론
        hmm = HMMPyTorch(transition_matrix)
        posteriors, _, _ = hmm.forward_backward(obs_probs)
        
        return posteriors, transition_matrix

# 학습 예제
model = LearnableTransitionHMM(num_states=5, feature_dim=80)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 샘플 데이터
features = torch.randn(2, 100, 80)
target_alignment = torch.randint(0, 5, (2, 100))

# 학습 루프
for epoch in range(10):
    optimizer.zero_grad()
    
    posteriors, trans_matrix = model(features)
    
    # 손실 계산 (예: 크로스 엔트로피)
    loss = nn.CrossEntropyLoss()(
        posteriors.reshape(-1, 5),
        target_alignment.reshape(-1)
    )
    
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("학습된 전이 행렬:")
print(model.get_transition_matrix())
```

## 5. 배치 처리

효율적인 배치 처리는 실제 응용에서 매우 중요합니다.

### 5.1 기본 배치 처리

```python
def batch_processing_example():
    # 다양한 길이의 시퀀스들
    sequences = [
        torch.randn(50, 5),   # 50 프레임
        torch.randn(75, 5),   # 75 프레임
        torch.randn(100, 5),  # 100 프레임
        torch.randn(60, 5),   # 60 프레임
    ]
    
    print("원본 시퀀스 길이:", [len(seq) for seq in sequences])
    
    # 패딩을 통한 배치 생성
    from torch.nn.utils.rnn import pad_sequence
    
    # 패딩 (배치 우선)
    padded_batch = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # 길이 정보 저장
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    print(f"패딩된 배치 형태: {padded_batch.shape}")
    print(f"실제 길이: {lengths.tolist()}")
    
    return padded_batch, lengths

padded_batch, lengths = batch_processing_example()
```

### 5.2 마스킹을 통한 효율적 처리

```python
def masked_hmm_processing(observations, lengths):
    """마스킹을 사용한 HMM 배치 처리"""
    
    batch_size, max_len, num_states = observations.shape
    
    # 1. 마스크 생성
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    print(f"마스크 형태: {mask.shape}")
    
    # 2. HMM 모델
    P = create_left_to_right_matrix(num_states, self_loop_prob=0.8)
    hmm = HMMPyTorch(P)
    
    # 3. 마스킹된 관측 확률
    masked_obs = observations * mask.unsqueeze(-1).float()
    
    # 4. Forward-backward (마스킹 고려)
    posteriors, forward, backward = hmm.forward_backward(masked_obs)
    
    # 5. 마스킹된 결과만 사용
    masked_posteriors = posteriors * mask.unsqueeze(-1).float()
    
    return masked_posteriors, mask

# 실행
masked_posteriors, mask = masked_hmm_processing(
    torch.softmax(padded_batch, dim=-1), lengths
)

print(f"마스킹된 후방확률 형태: {masked_posteriors.shape}")

# 실제 시퀀스별 결과 추출
for i, length in enumerate(lengths):
    seq_posteriors = masked_posteriors[i, :length]
    print(f"시퀀스 {i} 후방확률 형태: {seq_posteriors.shape}")
```

### 5.3 동적 배치 크기 처리

```python
class DynamicBatchHMM(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.num_states = num_states
        
        # 학습 가능한 HMM 레이어
        self.hmm_layer = HMMLayer(
            num_states=num_states,
            learnable_transitions=True
        )
    
    def forward(self, observations, lengths=None):
        batch_size = observations.size(0)
        
        if lengths is None:
            # 모든 시퀀스가 같은 길이
            return self.hmm_layer(observations)
        
        # 길이가 다른 경우 개별 처리 후 결합
        results = []
        alignments = []
        
        for i in range(batch_size):
            seq_len = lengths[i]
            seq_obs = observations[i, :seq_len].unsqueeze(0)
            
            aligned, alignment = self.hmm_layer(seq_obs)
            
            results.append(aligned.squeeze(0))
            alignments.append(alignment.squeeze(0))
        
        # 다시 패딩하여 배치로 만들기
        from torch.nn.utils.rnn import pad_sequence
        
        padded_results = pad_sequence(results, batch_first=True)
        padded_alignments = pad_sequence(alignments, batch_first=True)
        
        return padded_results, padded_alignments

# 사용 예제
dynamic_hmm = DynamicBatchHMM(num_states=5)
aligned_features, alignments = dynamic_hmm(padded_batch, lengths)

print(f"정렬된 특징 형태: {aligned_features.shape}")
print(f"정렬 정보 형태: {alignments.shape}")
```

## 6. GPU 사용법

GPU 가속을 통해 대용량 데이터 처리 성능을 크게 향상시킬 수 있습니다.

### 6.1 기본 GPU 설정

```python
def setup_gpu():
    """GPU 설정 및 확인"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU 이름: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"현재 메모리 사용량: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    return device

device = setup_gpu()
```

### 6.2 GPU HMM 처리

```python
def gpu_hmm_example():
    """GPU에서 HMM 처리 예제"""
    
    # 큰 배치 크기와 긴 시퀀스로 테스트
    batch_size = 32
    seq_len = 500
    num_states = 10
    
    print(f"테스트 설정: 배치 크기 {batch_size}, 시퀀스 길이 {seq_len}")
    
    # 데이터를 GPU로 이동
    observations = torch.softmax(
        torch.randn(batch_size, seq_len, num_states), dim=-1
    ).to(device)
    
    # HMM 모델도 GPU로 이동
    P = create_left_to_right_matrix(num_states, self_loop_prob=0.8)
    hmm = HMMPyTorch(P.to(device))
    
    print(f"관측 데이터 디바이스: {observations.device}")
    print(f"HMM 모델 디바이스: {hmm.device}")
    
    # GPU에서 처리 시간 측정
    if device.type == 'cuda':
        torch.cuda.synchronize()  # GPU 동기화
    
    start_time = time.time()
    
    # Forward-backward
    posteriors, forward, backward = hmm.forward_backward(observations)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    gpu_time = time.time() - start_time
    
    print(f"GPU 처리 시간: {gpu_time:.4f}초")
    print(f"메모리 사용량: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    return posteriors

gpu_posteriors = gpu_hmm_example()
```

### 6.3 메모리 효율적 처리

```python
def memory_efficient_processing():
    """메모리 효율적 HMM 처리"""
    
    # 매우 큰 데이터셋 시뮬레이션
    total_sequences = 1000
    chunk_size = 50  # 한 번에 처리할 시퀀스 수
    
    num_states = 8
    seq_len = 200
    
    P = create_left_to_right_matrix(num_states).to(device)
    hmm = HMMPyTorch(P)
    
    all_results = []
    
    print(f"총 {total_sequences}개 시퀀스를 {chunk_size}개씩 청크로 처리")
    
    for chunk_start in range(0, total_sequences, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_sequences)
        current_chunk_size = chunk_end - chunk_start
        
        # 청크 데이터 생성
        chunk_obs = torch.softmax(
            torch.randn(current_chunk_size, seq_len, num_states), dim=-1
        ).to(device)
        
        # 처리
        with torch.no_grad():  # 메모리 절약을 위해
            posteriors, _, _ = hmm.forward_backward(chunk_obs)
            
            # CPU로 이동하여 저장 (GPU 메모리 절약)
            all_results.append(posteriors.cpu())
        
        # GPU 메모리 정리
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if chunk_start % (chunk_size * 5) == 0:
            print(f"처리 완료: {chunk_end}/{total_sequences}")
    
    print("모든 청크 처리 완료")
    return all_results

# 실행 (GPU가 있는 경우만)
if device.type == 'cuda':
    results = memory_efficient_processing()
    print(f"총 결과 청크 수: {len(results)}")
```

## 7. 실제 음성 데이터 예제

실제 음성 처리 시나리오에 가까운 예제를 살펴보겠습니다.

### 7.1 음소 정렬 시뮬레이션

```python
def phoneme_alignment_example():
    """음소-음향 정렬 예제"""
    
    # 음소 시퀀스 (예: "HELLO")
    phonemes = ['H', 'E', 'L', 'L', 'O']
    phoneme_to_id = {p: i for i, p in enumerate(set(phonemes))}
    phoneme_ids = [phoneme_to_id[p] for p in phonemes]
    
    print(f"음소 시퀀스: {phonemes}")
    print(f"음소 ID: {phoneme_ids}")
    
    # 각 음소당 3개 상태 (시작-중간-끝)
    states_per_phoneme = 3
    total_states = len(set(phonemes)) * states_per_phoneme
    
    # 음성 특징 시뮬레이션 (예: MFCC 13차원)
    audio_frames = 150  # 1.5초 (10ms 프레임)
    feature_dim = 13
    
    # 실제로는 음성 파일에서 추출
    audio_features = torch.randn(1, audio_frames, feature_dim)
    
    print(f"음성 특징 형태: {audio_features.shape}")
    
    # 특징을 상태 확률로 변환 (실제로는 가우시안 모델 등 사용)
    feature_to_state = nn.Linear(feature_dim, total_states)
    state_probs = torch.softmax(feature_to_state(audio_features), dim=-1)
    
    # 음소별 HMM 정렬
    aligned_phonemes = []
    frame_idx = 0
    
    for phoneme, phoneme_id in zip(phonemes, phoneme_ids):
        # 각 음소의 예상 지속시간 (실제로는 언어 모델에서 예측)
        expected_duration = audio_frames // len(phonemes)
        end_frame = min(frame_idx + expected_duration + 20, audio_frames)
        
        # 해당 구간의 상태 확률
        phoneme_states = slice(phoneme_id * states_per_phoneme, 
                              (phoneme_id + 1) * states_per_phoneme)
        segment_probs = state_probs[:, frame_idx:end_frame, phoneme_states]
        
        # 음소별 HMM
        P_phoneme = create_left_to_right_matrix(states_per_phoneme, 0.8)
        hmm_phoneme = HMMPyTorch(P_phoneme)
        
        # 정렬
        posteriors, _, _ = hmm_phoneme.forward_backward(segment_probs)
        optimal_states, _ = hmm_phoneme.viterbi_decode(segment_probs)
        
        # 실제 프레임으로 변환
        phoneme_alignment = optimal_states[0] + phoneme_id * states_per_phoneme
        
        aligned_phonemes.append({
            'phoneme': phoneme,
            'start_frame': frame_idx,
            'end_frame': end_frame,
            'alignment': phoneme_alignment,
            'duration': end_frame - frame_idx
        })
        
        frame_idx = end_frame
    
    # 결과 출력
    print("\n=== 음소 정렬 결과 ===")
    for info in aligned_phonemes:
        duration_ms = info['duration'] * 10  # 10ms per frame
        print(f"음소 '{info['phoneme']}': "
              f"프레임 {info['start_frame']:3d}-{info['end_frame']:3d} "
              f"({duration_ms:3d}ms)")
    
    return aligned_phonemes

# 실행
alignment_result = phoneme_alignment_example()
```

### 7.2 음성 품질 평가

```python
def speech_quality_evaluation():
    """정렬 품질 평가 예제"""
    
    # 합성된 음성과 참조 음성 시뮬레이션
    reference_alignment = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    predicted_alignment = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    
    print("참조 정렬:", reference_alignment.tolist())
    print("예측 정렬:", predicted_alignment.tolist())
    
    # 프레임 단위 정확도
    frame_accuracy = (reference_alignment == predicted_alignment).float().mean()
    print(f"프레임 정확도: {frame_accuracy:.3f}")
    
    # 경계 검출 정확도
    def find_boundaries(alignment):
        boundaries = []
        for i in range(1, len(alignment)):
            if alignment[i] != alignment[i-1]:
                boundaries.append(i)
        return boundaries
    
    ref_boundaries = find_boundaries(reference_alignment)
    pred_boundaries = find_boundaries(predicted_alignment)
    
    print(f"참조 경계: {ref_boundaries}")
    print(f"예측 경계: {pred_boundaries}")
    
    # 경계 허용 오차 내 정확도 (±1 프레임)
    tolerance = 1
    correct_boundaries = 0
    
    for ref_b in ref_boundaries:
        for pred_b in pred_boundaries:
            if abs(ref_b - pred_b) <= tolerance:
                correct_boundaries += 1
                break
    
    boundary_accuracy = correct_boundaries / len(ref_boundaries) if ref_boundaries else 0
    print(f"경계 정확도 (±{tolerance} 프레임): {boundary_accuracy:.3f}")
    
    return frame_accuracy, boundary_accuracy

# 실행
frame_acc, boundary_acc = speech_quality_evaluation()
```

### 7.3 실시간 처리 시뮬레이션

```python
def streaming_hmm_simulation():
    """스트리밍 HMM 처리 시뮬레이션"""
    
    num_states = 5
    chunk_size = 20  # 20 프레임씩 처리
    total_frames = 200
    
    # HMM 모델
    P = create_left_to_right_matrix(num_states, 0.8)
    hmm = HMMPyTorch(P)
    
    # 스트리밍 상태 유지
    streaming_state = {
        'previous_forward': None,
        'accumulated_posteriors': [],
        'current_frame': 0
    }
    
    print(f"스트리밍 처리: {chunk_size} 프레임씩, 총 {total_frames} 프레임")
    
    for chunk_start in range(0, total_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_frames)
        current_chunk_size = chunk_end - chunk_start
        
        # 현재 청크 데이터
        chunk_obs = torch.softmax(
            torch.randn(1, current_chunk_size, num_states), dim=-1
        )
        
        # HMM 처리
        posteriors, forward, backward = hmm.forward_backward(chunk_obs)
        
        # 결과 누적
        streaming_state['accumulated_posteriors'].append(posteriors)
        streaming_state['current_frame'] += current_chunk_size
        
        # 실시간 정렬 (최근 청크만)
        current_alignment = torch.argmax(posteriors, dim=-1)[0]
        
        print(f"청크 {chunk_start:3d}-{chunk_end:3d}: "
              f"정렬 {current_alignment.tolist()}")
        
        # 지연 시간 시뮬레이션
        time.sleep(0.01)  # 10ms 지연
    
    # 전체 결과 결합
    all_posteriors = torch.cat(streaming_state['accumulated_posteriors'], dim=1)
    final_alignment = torch.argmax(all_posteriors, dim=-1)[0]
    
    print(f"\n최종 정렬 (총 {len(final_alignment)} 프레임):")
    print(f"처음 20개: {final_alignment[:20].tolist()}")
    print(f"마지막 20개: {final_alignment[-20:].tolist()}")
    
    return final_alignment

# 실행
streaming_result = streaming_hmm_simulation()
```

## 🎯 다음 단계

기본 사용법을 익혔다면, 이제 더 고급 기능들을 탐험해볼 시간입니다:

**다음 문서**: [고급 기능](03_advanced_features.md) - Neural HMM, Semi-Markov, 정렬 알고리즘

## 📝 요약

이 문서에서 다룬 주요 내용:

1. **설치 및 설정**: PyTorch HMM 라이브러리 환경 구성
2. **기본 HMM**: 전이 행렬, 관측 데이터, Forward-Backward, Viterbi
3. **알고리즘 비교**: Forward-Backward vs Viterbi의 차이점과 사용 시기
4. **신경망 통합**: HMMLayer를 사용한 end-to-end 학습
5. **배치 처리**: 효율적인 대용량 데이터 처리
6. **GPU 가속**: CUDA를 활용한 성능 최적화
7. **실제 응용**: 음성 처리에서의 실제 사용 예제

## 💡 핵심 포인트

- **Forward-Backward**: 학습과 불확실성 정량화에 사용
- **Viterbi**: 최종 추론과 실시간 처리에 사용
- **HMMLayer**: 신경망과의 자연스러운 통합 제공
- **배치 처리**: 패딩과 마스킹으로 효율적 처리
- **GPU 가속**: 대용량 데이터 처리 성능 향상

이제 이 기초를 바탕으로 더 복잡한 모델과 응용을 탐험해보세요!

---

**다음**: [고급 기능](03_advanced_features.md)에서 Neural HMM, Semi-Markov HMM, 정렬 알고리즘 등을 학습해보세요.

## 🎯 다음 단계

기본 사용법을 익혔다면, 이제 더 고급 기능들을 탐험해볼 시간입니다:

**다음 문서**: [고급 기능](03_advanced_features.md) - Neural HMM, Semi-Markov HMM, 정렬 알고리즘

## 📝 요약

이 문서에서 다룬 주요 내용:

1. **설치 및 설정**: PyTorch HMM 라이브러리 환경 구성
2. **기본 HMM**: 전이 행렬, 관측 데이터, Forward-Backward, Viterbi
3. **알고리즘 비교**: Forward-Backward vs Viterbi의 차이점과 사용 시기
4. **신경망 통합**: HMMLayer를 사용한 end-to-end 학습
5. **배치 처리**: 효율적인 대용량 데이터 처리
6. **GPU 가속**: CUDA를 활용한 성능 최적화
7. **실제 응용**: 음성 처리에서의 실제 사용 예제

## 💡 핵심 포인트

- **Forward-Backward**: 학습과 불확실성 정량화에 사용
- **Viterbi**: 최종 추론과 실시간 처리에 사용
- **HMMLayer**: 신경망과의 자연스러운 통합 제공
- **배치 처리**: 패딩과 마스킹으로 효율적 처리
- **GPU 가속**: 대용량 데이터 처리 성능 향상

이제 이 기초를 바탕으로 더 복잡한 모델과 응용을 탐험해보세요!

---

**다음**: [고급 기능](03_advanced_features.md)에서 Neural HMM, Semi-Markov HMM, 정렬 알고리즘 등을 학습해보세요. 