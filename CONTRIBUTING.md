# 🤝 Contributing to PyTorch HMM

PyTorch HMM 프로젝트에 기여해주셔서 감사합니다! 이 가이드는 효과적이고 일관된 기여를 위한 지침을 제공합니다.

## 📋 목차
- [시작하기](#시작하기)
- [개발 환경 설정](#개발-환경-설정)
- [코딩 표준](#코딩-표준)
- [테스트 가이드라인](#테스트-가이드라인)
- [Pull Request 프로세스](#pull-request-프로세스)
- [이슈 리포팅](#이슈-리포팅)
- [코드 리뷰 프로세스](#코드-리뷰-프로세스)

## 🚀 시작하기

### 기여 유형
다음과 같은 기여를 환영합니다:
- 🐛 **버그 수정**: 기존 문제 해결
- ✨ **새로운 기능**: HMM 모델, 알고리즘, 유틸리티 추가
- 📚 **문서화**: 문서 개선, 예제 추가, 튜토리얼 작성
- 🔧 **성능 최적화**: 속도, 메모리 사용량 개선
- 🧪 **테스트**: 테스트 커버리지 확대, 테스트 품질 개선
- 📝 **예제**: 실용적인 사용 사례 추가

### 기여 전 확인사항
1. [이슈](https://github.com/your-username/pytorch_hmm/issues)를 확인하여 중복 작업 방지
2. 큰 변경사항은 사전에 이슈로 논의
3. [로드맵](ROADMAP.md)을 확인하여 프로젝트 방향성 이해

## 🛠️ 개발 환경 설정

### 1. 저장소 포크 및 클론
```bash
# 1. GitHub에서 저장소 포크
# 2. 로컬에 클론
git clone https://github.com/YOUR-USERNAME/pytorch_hmm.git
cd pytorch_hmm

# 3. 원본 저장소를 upstream으로 추가
git remote add upstream https://github.com/original-owner/pytorch_hmm.git
```

### 2. 개발 환경 구성
```bash
# uv를 사용한 환경 설정 (권장)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
uv pip install -e ".[dev,test]"

# 또는 pip 사용
pip install -e ".[dev,test]"
```

### 3. 개발 도구 설정
```bash
# pre-commit 훅 설치
pre-commit install

# 코드 포맷팅 확인
black --check pytorch_hmm/
isort --check-only pytorch_hmm/

# 린터 실행
flake8 pytorch_hmm/
mypy pytorch_hmm/
```

### 4. 테스트 실행
```bash
# 전체 테스트 실행
pytest

# 커버리지와 함께 실행
pytest --cov=pytorch_hmm --cov-report=html

# 특정 테스트만 실행
pytest tests/test_hmm.py::TestHMM::test_forward_backward
```

## 📏 코딩 표준

### 코드 스타일
- **포맷터**: [Black](https://black.readthedocs.io/) (line length: 88)
- **임포트 정렬**: [isort](https://pycqa.github.io/isort/)
- **린터**: [flake8](https://flake8.pycqa.org/)
- **타입 힌트**: [mypy](https://mypy.readthedocs.io/) 사용 권장

### 네이밍 컨벤션
```python
# 클래스: PascalCase
class MixtureGaussianHMM:
    pass

# 함수/변수: snake_case
def forward_backward(observations, transition_matrix):
    hidden_states = []
    return hidden_states

# 상수: UPPER_SNAKE_CASE
DEFAULT_NUM_STATES = 5
MAX_ITERATIONS = 100

# 비공개 메서드: _leading_underscore
def _compute_emission_probs(self):
    pass
```

### 문서화 스타일
```python
def forward_backward(
    observations: torch.Tensor,
    transition_matrix: torch.Tensor,
    emission_matrix: torch.Tensor,
    initial_probs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-Backward 알고리즘을 사용하여 상태 확률을 계산합니다.
    
    Args:
        observations: 관측 시퀀스 [seq_len, obs_dim]
        transition_matrix: 전이 확률 행렬 [num_states, num_states]
        emission_matrix: 방출 확률 행렬 [num_states, obs_dim]
        initial_probs: 초기 상태 확률 [num_states]
    
    Returns:
        forward_probs: Forward 확률 [seq_len, num_states]
        backward_probs: Backward 확률 [seq_len, num_states]
    
    Example:
        >>> hmm = HMMPyTorch(num_states=3, obs_dim=2)
        >>> obs = torch.randn(10, 2)
        >>> forward, backward = hmm.forward_backward(obs)
        >>> print(forward.shape)  # torch.Size([10, 3])
    """
    # 구현...
```

### 에러 처리
```python
# 명확한 에러 메시지와 함께 적절한 예외 사용
def validate_input(observations: torch.Tensor) -> None:
    if observations.dim() != 2:
        raise ValueError(
            f"observations must be 2D tensor, got {observations.dim()}D"
        )
    
    if observations.size(0) == 0:
        raise ValueError("observations cannot be empty")
```

## 🧪 테스트 가이드라인

### 테스트 구조
```
tests/
├── test_hmm.py              # 기본 HMM 테스트
├── test_mixture_gaussian.py # GMM-HMM 테스트
├── test_streaming.py        # 스트리밍 테스트
├── test_integration.py      # 통합 테스트
└── conftest.py             # pytest 설정
```

### 테스트 작성 원칙
1. **단위 테스트**: 각 함수/메서드별 개별 테스트
2. **통합 테스트**: 전체 워크플로우 테스트
3. **성능 테스트**: 속도, 메모리 사용량 검증
4. **엣지 케이스**: 경계값, 예외 상황 테스트

### 테스트 예제
```python
import pytest
import torch
from pytorch_hmm import HMMPyTorch

class TestHMM:
    def test_forward_backward_shapes(self):
        """Forward-backward 출력 형태 검증"""
        hmm = HMMPyTorch(num_states=3, obs_dim=2)
        observations = torch.randn(10, 2)
        
        forward, backward = hmm.forward_backward(observations)
        
        assert forward.shape == (10, 3)
        assert backward.shape == (10, 3)
    
    def test_forward_backward_probabilities(self):
        """Forward-backward 확률 유효성 검증"""
        hmm = HMMPyTorch(num_states=3, obs_dim=2)
        observations = torch.randn(5, 2)
        
        forward, backward = hmm.forward_backward(observations)
        
        # 확률 값이 0과 1 사이인지 확인
        assert torch.all(forward >= 0)
        assert torch.all(forward <= 1)
        assert torch.all(backward >= 0)
        assert torch.all(backward <= 1)
    
    @pytest.mark.parametrize("num_states,obs_dim", [
        (2, 1), (3, 2), (5, 3), (10, 5)
    ])
    def test_different_dimensions(self, num_states, obs_dim):
        """다양한 차원에서의 동작 검증"""
        hmm = HMMPyTorch(num_states=num_states, obs_dim=obs_dim)
        observations = torch.randn(8, obs_dim)
        
        forward, backward = hmm.forward_backward(observations)
        
        assert forward.shape == (8, num_states)
        assert backward.shape == (8, num_states)
```

### 성능 테스트
```python
def test_performance_benchmark():
    """성능 벤치마크 테스트"""
    import time
    
    hmm = HMMPyTorch(num_states=10, obs_dim=5)
    observations = torch.randn(1000, 5)
    
    start_time = time.time()
    forward, backward = hmm.forward_backward(observations)
    elapsed_time = time.time() - start_time
    
    # 1초 이내에 완료되어야 함
    assert elapsed_time < 1.0
```

## 🔄 Pull Request 프로세스

### 1. 브랜치 생성
```bash
# 최신 main 브랜치로 업데이트
git checkout main
git pull upstream main

# 새 기능 브랜치 생성
git checkout -b feature/new-neural-hmm
# 또는 버그 수정: git checkout -b fix/tensor-dimension-error
```

### 2. 커밋 가이드라인
```bash
# 커밋 메시지 형식 (영어)
feat: add ContextualNeuralHMM with attention mechanism
fix: resolve tensor dimension mismatch in Semi-Markov HMM
docs: update README with performance benchmarks
test: add integration tests for streaming processor
refactor: optimize memory usage in forward-backward algorithm
```

**커밋 메시지 타입**:
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `test`: 테스트 추가/수정
- `refactor`: 코드 리팩토링
- `perf`: 성능 개선
- `style`: 코드 스타일 변경 (기능 변경 없음)

### 3. PR 템플릿
```markdown
## 📝 변경사항 요약
간단한 변경사항 설명

## 🎯 변경 유형
- [ ] 버그 수정
- [ ] 새로운 기능
- [ ] 문서 업데이트
- [ ] 성능 개선
- [ ] 테스트 추가
- [ ] 기타: ___________

## 🧪 테스트
- [ ] 기존 테스트 모두 통과
- [ ] 새로운 테스트 추가됨
- [ ] 수동 테스트 완료

## 📋 체크리스트
- [ ] 코드가 프로젝트 스타일 가이드를 따름
- [ ] 변경사항에 대한 문서 업데이트
- [ ] 새로운 의존성이 있다면 문서화됨
- [ ] Breaking changes가 있다면 CHANGELOG.md 업데이트

## 🔗 관련 이슈
Closes #123
```

### 4. PR 제출 전 체크리스트
```bash
# 코드 포맷팅
black pytorch_hmm/
isort pytorch_hmm/

# 린터 검사
flake8 pytorch_hmm/
mypy pytorch_hmm/

# 테스트 실행
pytest --cov=pytorch_hmm

# 문서 생성 테스트
cd docs && make html
```

## 🐛 이슈 리포팅

### 버그 리포트 템플릿
```markdown
## 🐛 버그 설명
버그에 대한 명확하고 간결한 설명

## 🔄 재현 단계
1. '...' 로 이동
2. '...' 클릭
3. '...' 스크롤
4. 오류 확인

## 🎯 예상 동작
예상했던 동작에 대한 설명

## 📱 환경
- OS: [예: Ubuntu 20.04]
- Python 버전: [예: 3.9.7]
- PyTorch 버전: [예: 2.0.1]
- CUDA 버전: [예: 11.8]
- pytorch_hmm 버전: [예: 0.2.1]

## 📎 추가 정보
스크린샷, 로그, 기타 관련 정보
```

### 기능 요청 템플릿
```markdown
## 🚀 기능 요청
원하는 기능에 대한 설명

## 💡 동기
이 기능이 필요한 이유와 해결하고자 하는 문제

## 📋 상세 설명
기능의 구체적인 동작 방식

## 🎯 대안
고려해본 다른 해결 방법들

## 📎 추가 정보
관련 자료, 참고 문헌 등
```

## 👀 코드 리뷰 프로세스

### 리뷰어 가이드라인
1. **건설적인 피드백**: 개선 방향 제시
2. **명확한 설명**: 변경이 필요한 이유 설명
3. **코드 품질**: 가독성, 성능, 보안 검토
4. **테스트 커버리지**: 적절한 테스트 확인

### 리뷰 체크리스트
- [ ] 코드가 요구사항을 만족하는가?
- [ ] 테스트가 충분한가?
- [ ] 성능에 부정적 영향은 없는가?
- [ ] 보안 이슈는 없는가?
- [ ] 문서화가 적절한가?
- [ ] Breaking change가 있다면 명시되었는가?

## 🏆 인정과 감사

모든 기여자는 다음과 같이 인정받습니다:
- README.md의 Contributors 섹션에 추가
- 릴리즈 노트에서 기여 내용 언급
- 중요한 기여에 대해서는 별도 감사 표시

## 📞 연락처

질문이나 도움이 필요하면:
- [GitHub Issues](https://github.com/your-username/pytorch_hmm/issues)
- [GitHub Discussions](https://github.com/your-username/pytorch_hmm/discussions)
- 이메일: your-email@example.com

---

**함께 더 나은 PyTorch HMM을 만들어 갑시다! 🚀** 