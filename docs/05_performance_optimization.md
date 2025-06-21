# PyTorch HMM 라이브러리 성능 최적화 가이드

이 문서에서는 PyTorch HMM 라이브러리를 사용한 애플리케이션의 성능을 최적화하는 방법을 다룹니다.

## 목차

1. [메모리 최적화](#메모리-최적화)
2. [연산 최적화](#연산-최적화)
3. [GPU 가속화](#gpu-가속화)
4. [배치 처리 최적화](#배치-처리-최적화)
5. [모델 압축](#모델-압축)
6. [실시간 처리 최적화](#실시간-처리-최적화)
7. [프로파일링과 벤치마킹](#프로파일링과-벤치마킹)

## 메모리 최적화

### Gradient Checkpointing

메모리 사용량을 줄이기 위해 gradient checkpointing을 사용할 수 있습니다.

```python
import torch
import torch.utils.checkpoint as checkpoint
from pytorch_hmm import NeuralHMM

class MemoryEfficientHMM(torch.nn.Module):
    def __init__(self, num_states, observation_dim, context_dim):
        super().__init__()
        self.hmm = NeuralHMM(
            num_states=num_states,
            observation_dim=observation_dim,
            context_dim=context_dim,
            hidden_dim=256
        )
        
    def forward(self, observations, context=None):
        # Gradient checkpointing으로 메모리 절약
        return checkpoint.checkpoint(
            self.hmm, 
            observations, 
            context,
            preserve_rng_state=True
        )

# 사용 예제
model = MemoryEfficientHMM(num_states=50, observation_dim=80, context_dim=64)
observations = torch.randn(2, 1000, 80)  # 긴 시퀀스
context = torch.randn(2, 1000, 64)

# 메모리 효율적인 forward pass
posteriors, forward, backward = model(observations, context)
print(f"Memory efficient processing completed")
```

### 청크 기반 처리

긴 시퀀스를 작은 청크로 나누어 처리합니다.

```python
def process_long_sequence(model, observations, chunk_size=500, overlap=50):
    """긴 시퀀스를 청크로 나누어 처리"""
    batch_size, seq_len, obs_dim = observations.shape
    
    if seq_len <= chunk_size:
        return model(observations)
    
    results = []
    
    for start in range(0, seq_len, chunk_size - overlap):
        end = min(start + chunk_size, seq_len)
        chunk = observations[:, start:end]
        
        with torch.no_grad():
            chunk_result = model(chunk)
            
            # 오버랩 부분 제거 (첫 번째와 마지막 청크 제외)
            if start > 0 and end < seq_len:
                chunk_result = chunk_result[:, overlap//2:-overlap//2]
            elif start > 0:
                chunk_result = chunk_result[:, overlap//2:]
            elif end < seq_len:
                chunk_result = chunk_result[:, :-overlap//2]
            
            results.append(chunk_result)
    
    return torch.cat(results, dim=1)

# 사용 예제
model = NeuralHMM(num_states=50, observation_dim=80, context_dim=0, hidden_dim=256)
long_observations = torch.randn(1, 5000, 80)  # 매우 긴 시퀀스

# 청크 기반 처리
result = process_long_sequence(model, long_observations, chunk_size=1000, overlap=100)
print(f"Processed sequence shape: {result[0].shape}")
```

## 연산 최적화

### 벡터화된 연산

루프 대신 벡터화된 연산을 사용합니다.

```python
import torch
import torch.nn.functional as F

class OptimizedHMM(torch.nn.Module):
    def __init__(self, num_states, observation_dim):
        super().__init__()
        self.num_states = num_states
        self.observation_dim = observation_dim
        
        # 파라미터 초기화
        self.log_transition = torch.nn.Parameter(torch.randn(num_states, num_states))
        self.log_initial = torch.nn.Parameter(torch.randn(num_states))
        self.observation_net = torch.nn.Linear(observation_dim, num_states)
        
    def forward(self, observations):
        """벡터화된 forward algorithm"""
        batch_size, seq_len, _ = observations.shape
        
        # 관측 확률 계산 (전체 시퀀스를 한 번에 처리)
        log_obs_probs = self.observation_net(observations.view(-1, self.observation_dim))
        log_obs_probs = log_obs_probs.view(batch_size, seq_len, self.num_states)
        
        # Forward pass (벡터화된 구현)
        log_alpha = self.log_initial.unsqueeze(0) + log_obs_probs[:, 0]
        
        # 모든 시간 단계를 벡터화하여 처리
        for t in range(1, seq_len):
            # 브로드캐스팅을 활용한 전이 확률 계산
            transition_scores = (
                log_alpha.unsqueeze(2) + 
                self.log_transition.unsqueeze(0)
            )
            
            # LogSumExp 연산
            log_alpha = torch.logsumexp(transition_scores, dim=1) + log_obs_probs[:, t]
        
        # 최종 로그 우도
        log_likelihood = torch.logsumexp(log_alpha, dim=1)
        
        return log_likelihood, log_alpha
```

### JIT 컴파일

TorchScript를 사용하여 모델을 컴파일합니다.

```python
@torch.jit.script
def jit_logsumexp_transition(log_alpha, log_transition, log_obs):
    """JIT 컴파일된 전이 연산"""
    transition_scores = log_alpha.unsqueeze(2) + log_transition.unsqueeze(0)
    new_alpha = torch.logsumexp(transition_scores, dim=1) + log_obs
    return new_alpha

class JITOptimizedHMM(torch.nn.Module):
    def __init__(self, num_states, observation_dim):
        super().__init__()
        self.num_states = num_states
        self.log_transition = torch.nn.Parameter(torch.randn(num_states, num_states))
        self.log_initial = torch.nn.Parameter(torch.randn(num_states))
        self.observation_net = torch.nn.Linear(observation_dim, num_states)
        
    def forward(self, observations):
        batch_size, seq_len, _ = observations.shape
        
        # 관측 확률 계산
        log_obs_probs = self.observation_net(observations.view(-1, self.observation_dim))
        log_obs_probs = log_obs_probs.view(batch_size, seq_len, self.num_states)
        
        # 초기 상태
        log_alpha = self.log_initial.unsqueeze(0) + log_obs_probs[:, 0]
        
        # JIT 컴파일된 함수 사용
        for t in range(1, seq_len):
            log_alpha = jit_logsumexp_transition(
                log_alpha, 
                self.log_transition, 
                log_obs_probs[:, t]
            )
        
        return torch.logsumexp(log_alpha, dim=1)

# JIT 모델 생성 및 사용
model = JITOptimizedHMM(num_states=50, observation_dim=80)
model = torch.jit.script(model)  # JIT 컴파일

# 사용 예제
observations = torch.randn(16, 100, 80)
result = model(observations)
print(f"JIT compiled model result: {result.shape}")
```

## GPU 가속화

### 효율적인 GPU 사용

```python
import torch
import torch.nn as nn

class GPUOptimizedHMM(nn.Module):
    def __init__(self, num_states, observation_dim, device='cuda'):
        super().__init__()
        self.device = device
        self.num_states = num_states
        
        # 모든 파라미터를 GPU에 초기화
        self.log_transition = nn.Parameter(
            torch.randn(num_states, num_states, device=device)
        )
        self.log_initial = nn.Parameter(
            torch.randn(num_states, device=device)
        )
        self.observation_net = nn.Linear(observation_dim, num_states).to(device)
        
    def forward(self, observations):
        # 입력이 GPU에 있는지 확인
        if observations.device != self.device:
            observations = observations.to(self.device)
        
        batch_size, seq_len, _ = observations.shape
        
        # GPU 메모리에서 직접 연산
        log_obs_probs = self.observation_net(
            observations.view(-1, observations.shape[-1])
        ).view(batch_size, seq_len, self.num_states)
        
        # Forward algorithm
        log_alpha = self.log_initial.unsqueeze(0) + log_obs_probs[:, 0]
        
        for t in range(1, seq_len):
            # GPU에서 효율적인 브로드캐스팅
            transition_scores = (
                log_alpha.unsqueeze(2) + 
                self.log_transition.unsqueeze(0)
            )
            log_alpha = torch.logsumexp(transition_scores, dim=1) + log_obs_probs[:, t]
        
        return torch.logsumexp(log_alpha, dim=1)

# GPU 사용 예제
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = 'cpu'
    print("Using CPU")

model = GPUOptimizedHMM(num_states=100, observation_dim=80, device=device)
observations = torch.randn(64, 500, 80, device=device)

# GPU에서 추론
with torch.no_grad():
    result = model(observations)
    print(f"GPU processing result: {result.shape}")
```

### 혼합 정밀도 학습

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def train_step(self, observations, targets):
        """혼합 정밀도 학습 단계"""
        self.optimizer.zero_grad()
        
        # autocast 컨텍스트에서 forward pass
        with autocast():
            predictions = self.model(observations)
            loss = torch.nn.functional.mse_loss(predictions, targets)
        
        # 스케일된 backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

# 혼합 정밀도 학습 사용
model = GPUOptimizedHMM(num_states=50, observation_dim=80, device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = MixedPrecisionTrainer(model, optimizer)

# 학습 데이터
observations = torch.randn(32, 200, 80, device='cuda')
targets = torch.randn(32, device='cuda')

# 학습 수행
loss = trainer.train_step(observations, targets)
print(f"Training loss: {loss:.4f}")
```

## 실시간 처리 최적화

### 스트리밍 처리

```python
class StreamingHMMProcessor:
    def __init__(self, model, buffer_size=1000, overlap=100):
        self.model = model
        self.buffer_size = buffer_size
        self.overlap = overlap
        
        # 스트리밍 상태
        self.feature_buffer = torch.zeros(0, model.observation_dim)
        self.state_buffer = None
        
        # 성능 최적화를 위한 사전 할당
        self.temp_buffer = torch.zeros(buffer_size, model.observation_dim)
        
    def process_chunk(self, new_features):
        """새로운 특징 청크 처리"""
        # 버퍼에 새 특징 추가
        self.feature_buffer = torch.cat([self.feature_buffer, new_features])
        
        results = []
        
        # 충분한 데이터가 있으면 처리
        while len(self.feature_buffer) >= self.buffer_size:
            # 현재 윈도우 추출
            current_window = self.feature_buffer[:self.buffer_size]
            
            # 모델 추론
            with torch.no_grad():
                result = self.model(current_window.unsqueeze(0))
                results.append(result.squeeze(0))
            
            # 버퍼 업데이트 (오버랩 유지)
            self.feature_buffer = self.feature_buffer[self.buffer_size - self.overlap:]
        
        return results
    
    def reset(self):
        """스트리밍 상태 초기화"""
        self.feature_buffer = torch.zeros(0, self.model.observation_dim)
        self.state_buffer = None

# 실시간 처리 시뮬레이션
model = OptimizedHMM(num_states=50, observation_dim=80)
processor = StreamingHMMProcessor(model, buffer_size=500, overlap=50)

# 스트리밍 데이터 시뮬레이션
for i in range(20):  # 20개 청크
    # 새로운 특징 청크 (실제로는 마이크로폰 입력)
    chunk = torch.randn(100, 80)  # 100 프레임
    
    # 청크 처리
    results = processor.process_chunk(chunk)
    
    if results:
        print(f"Chunk {i}: processed {len(results)} windows")
```

## 프로파일링과 벤치마킹

### 성능 프로파일링

```python
import time

def benchmark_models(models, input_data, num_iterations=100):
    """여러 모델 성능 비교"""
    results = {}
    
    for name, model in models.items():
        model.eval()
        
        # 워밍업
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        # 실제 측정
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                result = model(input_data)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        results[name] = avg_time
        
        print(f"{name}: {avg_time * 1000:.2f}ms per iteration")
    
    return results

# 사용 예제
input_data = torch.randn(16, 200, 80)

# 다양한 모델 비교
models = {
    'Basic HMM': OptimizedHMM(num_states=50, observation_dim=80),
    'JIT HMM': torch.jit.script(JITOptimizedHMM(num_states=50, observation_dim=80))
}

# 벤치마킹 수행
benchmark_results = benchmark_models(models, input_data)
```

### 메모리 사용량 모니터링

```python
import psutil
import torch

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = self.get_memory_usage()
        
    def get_memory_usage(self):
        """현재 메모리 사용량 반환 (MB)"""
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024
    
    def get_gpu_memory_usage(self):
        """GPU 메모리 사용량 반환 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    def print_memory_stats(self, label=""):
        """메모리 통계 출력"""
        cpu_memory = self.get_memory_usage()
        gpu_memory = self.get_gpu_memory_usage()
        
        print(f"Memory Stats {label}:")
        print(f"  CPU Memory: {cpu_memory:.1f} MB")
        print(f"  GPU Memory: {gpu_memory:.1f} MB")
        print(f"  Memory increase: {cpu_memory - self.start_memory:.1f} MB")

# 메모리 모니터링 사용 예제
monitor = MemoryMonitor()
monitor.print_memory_stats("(Initial)")

# 모델 생성
model = OptimizedHMM(num_states=100, observation_dim=80)
monitor.print_memory_stats("(After model creation)")

# 대용량 데이터 처리
large_data = torch.randn(64, 1000, 80)
result = model(large_data)
monitor.print_memory_stats("(After inference)")

# 메모리 정리
del large_data, result
torch.cuda.empty_cache() if torch.cuda.is_available() else None
monitor.print_memory_stats("(After cleanup)")
```

## 마무리

이 문서에서는 PyTorch HMM 라이브러리의 성능을 최적화하는 다양한 방법을 살펴보았습니다:

- **메모리 최적화**: Gradient checkpointing, 청크 처리
- **연산 최적화**: 벡터화, JIT 컴파일
- **GPU 가속화**: 효율적인 GPU 사용, 혼합 정밀도
- **실시간 처리**: 스트리밍 최적화
- **프로파일링**: 성능 측정 및 모니터링

이러한 최적화 기법들을 적절히 조합하여 사용하면 실제 제품 환경에서도 효율적으로 HMM 모델을 활용할 수 있습니다.

마지막으로 [예제 모음](06_examples.md)을 참고하여 다양한 사용 사례를 확인해보시기 바랍니다. 