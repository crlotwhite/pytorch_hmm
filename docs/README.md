# 📚 PyTorch HMM 문서

PyTorch HMM 프로젝트의 종합 문서 모음입니다.

## 📋 문서 구조

### 🚀 **시작하기**
- [프로젝트 README](../README.md) - 전체 프로젝트 개요
- [빠른 시작 가이드](tutorials/quickstart.md) - 5분 만에 시작하기
- [설치 가이드](tutorials/installation.md) - 상세 설치 방법

### 📖 **튜토리얼**
- [기본 HMM 사용법](tutorials/basic_hmm.md) - HMM 기초 사용법
- [고급 모델 활용](tutorials/advanced_models.md) - MixtureGaussian, HSMM, Neural HMM
- [실시간 스트리밍](tutorials/streaming.md) - 실시간 음성 처리
- [정렬 알고리즘](tutorials/alignment.md) - DTW, CTC 정렬 기법
- [성능 최적화](tutorials/optimization.md) - GPU 가속 및 메모리 최적화

### 🔧 **API 레퍼런스**
- [HMM 모듈](api/hmm.md) - 기본 HMM 클래스
- [HSMM 모듈](api/hsmm.md) - Hidden Semi-Markov Model
- [MixtureGaussian 모듈](api/mixture_gaussian.md) - 가우시안 혼합 HMM
- [Neural HMM 모듈](api/neural.md) - 신경망 기반 HMM
- [스트리밍 모듈](api/streaming.md) - 실시간 처리
- [정렬 모듈](api/alignment.md) - DTW/CTC 정렬
- [유틸리티 모듈](api/utils.md) - 보조 함수들

### 💡 **예제 코드**
- [기본 예제](examples/basic_examples.md) - 간단한 사용 예제
- [고급 예제](examples/advanced_examples.md) - 복잡한 시나리오
- [통합 예제](examples/integration_examples.md) - 다른 시스템과의 통합
- [벤치마크 예제](examples/benchmark_examples.md) - 성능 측정

### 🛠️ **개발 가이드**
- [기여 가이드](../CONTRIBUTING.md) - 프로젝트 기여 방법
- [코딩 스타일](tutorials/coding_style.md) - 코드 작성 규칙
- [테스트 가이드](tutorials/testing.md) - 테스트 작성 및 실행
- [디버깅 가이드](troubleshooting/debugging.md) - 문제 해결 방법

### 📊 **성능 및 벤치마크**
- [성능 가이드](../PERFORMANCE.md) - 성능 최적화 방법
- [벤치마크 결과](troubleshooting/benchmark_results.md) - 상세 성능 데이터
- [하드웨어 요구사항](troubleshooting/hardware_requirements.md) - 시스템 요구사항

### 🚨 **문제 해결**
- [FAQ](troubleshooting/faq.md) - 자주 묻는 질문
- [알려진 문제](troubleshooting/known_issues.md) - 알려진 문제와 해결책
- [오류 해결](troubleshooting/error_solutions.md) - 일반적인 오류 해결
- [성능 문제](troubleshooting/performance_issues.md) - 성능 관련 문제

### 📈 **프로젝트 정보**
- [로드맵](../ROADMAP.md) - 개발 계획
- [변경사항](../CHANGELOG.md) - 버전별 변경사항
- [라이선스](../LICENSE) - 라이선스 정보

## 🔍 빠른 검색

### 사용 목적별 가이드

#### 🎯 **음성 합성 시스템 개발자**
1. [빠른 시작 가이드](tutorials/quickstart.md)
2. [기본 HMM 사용법](tutorials/basic_hmm.md)
3. [성능 최적화](tutorials/optimization.md)
4. [실시간 스트리밍](tutorials/streaming.md)

#### 🔬 **연구자**
1. [고급 모델 활용](tutorials/advanced_models.md)
2. [API 레퍼런스](api/)
3. [고급 예제](examples/advanced_examples.md)
4. [벤치마크 결과](troubleshooting/benchmark_results.md)

#### 🛠️ **시스템 통합 개발자**
1. [통합 예제](examples/integration_examples.md)
2. [하드웨어 요구사항](troubleshooting/hardware_requirements.md)
3. [성능 가이드](../PERFORMANCE.md)
4. [문제 해결](troubleshooting/)

#### 🤝 **기여자**
1. [기여 가이드](../CONTRIBUTING.md)
2. [코딩 스타일](tutorials/coding_style.md)
3. [테스트 가이드](tutorials/testing.md)
4. [디버깅 가이드](troubleshooting/debugging.md)

## 📞 도움말

### 문서 관련 문의
- **GitHub Issues**: [문서 관련 이슈](https://github.com/crlotwhite/pytorch_hmm/issues?q=label%3Adocumentation)
- **개선 제안**: 문서 개선 사항이 있으시면 이슈나 PR로 제안해주세요

### 기술 지원
- **사용법 문의**: [GitHub Discussions](https://github.com/crlotwhite/pytorch_hmm/discussions)
- **버그 신고**: [GitHub Issues](https://github.com/crlotwhite/pytorch_hmm/issues)
- **기능 요청**: [GitHub Issues](https://github.com/crlotwhite/pytorch_hmm/issues)

## 📝 문서 작성 가이드

문서 기여를 원하시는 분들을 위한 가이드라인:

### 문서 작성 원칙
- **명확성**: 기술적 내용을 이해하기 쉽게 설명
- **완전성**: 필요한 모든 정보 포함
- **정확성**: 최신 코드와 일치하는 내용
- **실용성**: 실제 사용 가능한 예제 제공

### 문서 구조
- **제목**: 명확하고 구체적인 제목
- **개요**: 문서의 목적과 범위 설명
- **단계별 설명**: 논리적 순서로 내용 구성
- **예제 코드**: 동작하는 실제 코드 제공
- **참고 자료**: 관련 문서 링크

### 코드 예제 작성
- **완전한 예제**: 바로 실행 가능한 코드
- **주석**: 중요한 부분에 한국어 주석
- **오류 처리**: 예상 가능한 오류 상황 고려
- **성능 고려**: 효율적인 코드 작성

---

**PyTorch HMM으로 고품질 음성 합성 시스템을 구축하세요! 🎵** 