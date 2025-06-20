"""
PyTorch HMM Performance Benchmark

이 스크립트는 다양한 조건에서 PyTorch HMM 라이브러리의 성능을 측정합니다.
음성 합성 및 시퀀스 모델링 워크로드에서의 실시간 처리 가능성을 평가합니다.

측정 항목:
- Forward-backward 알고리즘 성능
- Viterbi 디코딩 성능  
- Neural HMM 성능
- DTW/CTC 정렬 성능
- GPU vs CPU 성능 비교
- 메모리 사용량
- 실시간 처리 가능성 (>80fps)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional
import argparse
import json
from dataclasses import dataclass

# PyTorch HMM imports
from pytorch_hmm import (
    HMMPyTorch, HMMLayer, NeuralHMM, SemiMarkovHMM,
    DTWAligner, CTCAligner, GaussianHMMLayer,
    create_left_to_right_matrix
)


@dataclass
class BenchmarkConfig:
    """벤치마크 설정"""
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_states_list: List[int] = None
    num_iterations: int = 10
    warmup_iterations: int = 3
    use_gpu: bool = True
    measure_memory: bool = True
    test_basic_hmm: bool = True
    test_neural_hmm: bool = True
    test_alignment: bool = True
    test_semi_markov: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.sequence_lengths is None:
            self.sequence_lengths = [50, 100, 200, 500]
        if self.num_states_list is None:
            self.num_states_list = [5, 10, 20, 50]


class PerformanceBenchmark:
    """성능 벤치마크 실행기"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = self._setup_device()
        self.results = {}
        
        print(f"🚀 PyTorch HMM Performance Benchmark")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        print()
    
    def _setup_device(self) -> str:
        """디바이스 설정"""
        if self.config.use_gpu and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def _measure_time_and_memory(self, func, *args, **kwargs) -> Tuple[float, float, any]:
        """실행 시간과 메모리 사용량 측정"""
        # GPU 동기화 및 메모리 정리
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # CPU 메모리 측정 시작
        process = psutil.Process()
        cpu_mem_start = process.memory_info().rss / 1024**2  # MB
        
        # GPU 메모리 측정 시작
        gpu_mem_start = 0
        if self.device == 'cuda':
            gpu_mem_start = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # 함수 실행
        start_time = time.time()
        result = func(*args, **kwargs)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        execution_time = time.time() - start_time
        
        # 메모리 사용량 측정
        cpu_mem_end = process.memory_info().rss / 1024**2
        cpu_mem_used = cpu_mem_end - cpu_mem_start
        
        gpu_mem_used = 0
        if self.device == 'cuda':
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2
            gpu_mem_used = gpu_mem_peak - gpu_mem_start
        
        total_mem_used = cpu_mem_used + gpu_mem_used
        
        return execution_time, total_mem_used, result
    
    def _run_benchmark(self, func, *args, **kwargs) -> Dict:
        """반복 실행으로 안정적인 벤치마크 측정"""
        times = []
        memory_usages = []
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            func(*args, **kwargs)
        
        # 실제 측정
        for _ in range(self.config.num_iterations):
            exec_time, mem_usage, _ = self._measure_time_and_memory(func, *args, **kwargs)
            times.append(exec_time)
            memory_usages.append(mem_usage)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_memory': np.mean(memory_usages),
            'max_memory': np.max(memory_usages)
        }
    
    def benchmark_basic_hmm(self) -> Dict:
        """기본 HMM 성능 벤치마크"""
        print("📊 Benchmarking Basic HMM...")
        results = {}
        
        for num_states in self.config.num_states_list:
            results[f'states_{num_states}'] = {}
            
            # HMM 모델 생성
            P = create_left_to_right_matrix(num_states)
            hmm = HMMPyTorch(P).to(self.device)
            
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    
                    # 테스트 데이터 생성
                    observations = torch.softmax(
                        torch.randn(batch_size, seq_len, num_states), dim=-1
                    ).to(self.device)
                    
                    config_key = f'b{batch_size}_s{seq_len}'
                    results[f'states_{num_states}'][config_key] = {}
                    
                    # Forward-backward 벤치마크
                    def forward_backward():
                        return hmm.forward_backward(observations)
                    
                    fb_result = self._run_benchmark(forward_backward)
                    results[f'states_{num_states}'][config_key]['forward_backward'] = fb_result
                    
                    # Viterbi 벤치마크
                    def viterbi():
                        return hmm.viterbi_decode(observations)
                    
                    viterbi_result = self._run_benchmark(viterbi)
                    results[f'states_{num_states}'][config_key]['viterbi'] = viterbi_result
                    
                    # 성능 계산
                    total_frames = batch_size * seq_len
                    fb_fps = total_frames / fb_result['mean_time']
                    viterbi_fps = total_frames / viterbi_result['mean_time']
                    
                    results[f'states_{num_states}'][config_key]['fb_fps'] = fb_fps
                    results[f'states_{num_states}'][config_key]['viterbi_fps'] = viterbi_fps
                    results[f'states_{num_states}'][config_key]['realtime_capable'] = {
                        'forward_backward': fb_fps > 80,
                        'viterbi': viterbi_fps > 80
                    }
                    
                    print(f"   States: {num_states:2d}, Batch: {batch_size:2d}, Seq: {seq_len:3d} "
                          f"| FB: {fb_fps:6.0f} fps, Viterbi: {viterbi_fps:6.0f} fps")
        
        return results
    
    def benchmark_neural_hmm(self) -> Dict:
        """Neural HMM 성능 벤치마크"""
        print("\n🧠 Benchmarking Neural HMM...")
        results = {}
        
        # Neural HMM 설정
        observation_dim = 80
        context_dim = 64
        hidden_dim = 128
        
        for num_states in [5, 10, 20]:  # Neural HMM은 계산 비용이 높아 적은 수로 테스트
            results[f'states_{num_states}'] = {}
            
            # Neural HMM 모델 생성
            neural_hmm = NeuralHMM(
                num_states=num_states,
                observation_dim=observation_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                transition_type='mlp',
                observation_type='gaussian'
            ).to(self.device)
            
            for batch_size in [1, 4, 8]:  # 작은 배치로 테스트
                for seq_len in [50, 100, 200]:
                    
                    # 테스트 데이터 생성
                    observations = torch.randn(batch_size, seq_len, observation_dim).to(self.device)
                    context = torch.randn(batch_size, seq_len, context_dim).to(self.device)
                    
                    config_key = f'b{batch_size}_s{seq_len}'
                    results[f'states_{num_states}'][config_key] = {}
                    
                    # Forward pass 벤치마크
                    def neural_forward():
                        with torch.no_grad():
                            return neural_hmm(observations, context)
                    
                    forward_result = self._run_benchmark(neural_forward)
                    results[f'states_{num_states}'][config_key]['forward'] = forward_result
                    
                    # Viterbi 벤치마크
                    def neural_viterbi():
                        with torch.no_grad():
                            return neural_hmm.viterbi_decode(observations, context)
                    
                    viterbi_result = self._run_benchmark(neural_viterbi)
                    results[f'states_{num_states}'][config_key]['viterbi'] = viterbi_result
                    
                    # 성능 계산
                    total_frames = batch_size * seq_len
                    forward_fps = total_frames / forward_result['mean_time']
                    viterbi_fps = total_frames / viterbi_result['mean_time']
                    
                    results[f'states_{num_states}'][config_key]['forward_fps'] = forward_fps
                    results[f'states_{num_states}'][config_key]['viterbi_fps'] = viterbi_fps
                    results[f'states_{num_states}'][config_key]['realtime_capable'] = {
                        'forward': forward_fps > 80,
                        'viterbi': viterbi_fps > 80
                    }
                    
                    print(f"   States: {num_states:2d}, Batch: {batch_size:2d}, Seq: {seq_len:3d} "
                          f"| Forward: {forward_fps:6.0f} fps, Viterbi: {viterbi_fps:6.0f} fps")
        
        return results
    
    def benchmark_alignment_algorithms(self) -> Dict:
        """정렬 알고리즘 성능 벤치마크"""
        print("\n🎯 Benchmarking Alignment Algorithms...")
        results = {}
        
        # DTW 벤치마크
        print("   DTW Alignment:")
        dtw_results = {}
        aligner = DTWAligner(distance_fn='euclidean')
        
        feature_dim = 40
        
        for seq_len1 in [20, 50, 100]:
            for seq_len2 in [25, 60, 120]:
                
                # 테스트 시퀀스 생성
                x = torch.randn(seq_len1, feature_dim).to(self.device)
                y = torch.randn(seq_len2, feature_dim).to(self.device)
                
                def dtw_align():
                    return aligner(x, y)
                
                dtw_result = self._run_benchmark(dtw_align)
                
                config_key = f's1_{seq_len1}_s2_{seq_len2}'
                dtw_results[config_key] = dtw_result
                
                # DTW 특수 메트릭
                dtw_complexity = seq_len1 * seq_len2
                dtw_efficiency = dtw_complexity / dtw_result['mean_time']
                dtw_results[config_key]['complexity'] = dtw_complexity
                dtw_results[config_key]['efficiency'] = dtw_efficiency
                
                print(f"      {seq_len1:3d}x{seq_len2:3d}: {dtw_result['mean_time']:.4f}s "
                      f"({dtw_efficiency:.0f} ops/s)")
        
        results['dtw'] = dtw_results
        
        # CTC 벤치마크
        print("   CTC Alignment:")
        ctc_results = {}
        
        vocab_size = 30
        ctc_aligner = CTCAligner(num_classes=vocab_size, blank_id=0)
        
        for batch_size in [1, 4, 8]:
            for seq_len in [50, 100, 200]:
                for target_len in [5, 10, 15]:
                    
                    # CTC 테스트 데이터
                    log_probs = torch.log_softmax(
                        torch.randn(seq_len, batch_size, vocab_size), dim=-1
                    ).to(self.device)
                    
                    targets = torch.randint(1, vocab_size, (batch_size, target_len)).to(self.device)
                    input_lengths = torch.full((batch_size,), seq_len).to(self.device)
                    target_lengths = torch.full((batch_size,), target_len).to(self.device)
                    
                    def ctc_loss():
                        return ctc_aligner(log_probs, targets, input_lengths, target_lengths)
                    
                    def ctc_decode():
                        return ctc_aligner.decode(log_probs, input_lengths)
                    
                    loss_result = self._run_benchmark(ctc_loss)
                    decode_result = self._run_benchmark(ctc_decode)
                    
                    config_key = f'b{batch_size}_s{seq_len}_t{target_len}'
                    ctc_results[config_key] = {
                        'loss': loss_result,
                        'decode': decode_result
                    }
                    
                    # CTC 특수 메트릭
                    total_frames = batch_size * seq_len
                    decode_fps = total_frames / decode_result['mean_time']
                    ctc_results[config_key]['decode_fps'] = decode_fps
                    ctc_results[config_key]['realtime_capable'] = decode_fps > 80
                    
                    print(f"      B:{batch_size}, S:{seq_len:3d}, T:{target_len:2d}: "
                          f"Decode {decode_fps:6.0f} fps")
        
        results['ctc'] = ctc_results
        
        return results
    
    def benchmark_semi_markov_hmm(self) -> Dict:
        """Semi-Markov HMM 성능 벤치마크"""
        print("\n⏱️ Benchmarking Semi-Markov HMM...")
        results = {}
        
        observation_dim = 20
        max_duration = 15
        
        for num_states in [3, 5, 8]:
            results[f'states_{num_states}'] = {}
            
            # HSMM 모델 생성
            hsmm = SemiMarkovHMM(
                num_states=num_states,
                observation_dim=observation_dim,
                max_duration=max_duration,
                duration_distribution='gamma',
                observation_model='gaussian'
            ).to(self.device)
            
            for seq_len in [30, 50, 80]:
                
                # 테스트 데이터
                observations = torch.randn(1, seq_len, observation_dim).to(self.device)
                
                config_key = f's{seq_len}'
                results[f'states_{num_states}'][config_key] = {}
                
                # Sampling 벤치마크
                def hsmm_sample():
                    return hsmm.sample(num_states=5, max_length=seq_len)
                
                sample_result = self._run_benchmark(hsmm_sample)
                results[f'states_{num_states}'][config_key]['sampling'] = sample_result
                
                # Duration model 벤치마크
                def duration_inference():
                    state_indices = torch.randint(0, num_states, (10,)).to(self.device)
                    return hsmm.duration_model(state_indices)
                
                duration_result = self._run_benchmark(duration_inference)
                results[f'states_{num_states}'][config_key]['duration'] = duration_result
                
                print(f"   States: {num_states}, Seq: {seq_len:2d} "
                      f"| Sample: {sample_result['mean_time']:.4f}s, "
                      f"Duration: {duration_result['mean_time']:.4f}s")
        
        return results
    
    def benchmark_memory_usage(self) -> Dict:
        """메모리 사용량 상세 분석"""
        print("\n💾 Benchmarking Memory Usage...")
        
        # 큰 모델로 메모리 테스트
        num_states = 20
        batch_size = 16
        seq_len = 300
        
        memory_results = {}
        
        # 기본 HMM 메모리 사용량
        P = create_left_to_right_matrix(num_states)
        hmm = HMMPyTorch(P).to(self.device)
        observations = torch.softmax(
            torch.randn(batch_size, seq_len, num_states), dim=-1
        ).to(self.device)
        
        # 메모리 측정
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            base_memory = torch.cuda.memory_allocated()
            posteriors, _, _ = hmm.forward_backward(observations)
            peak_memory = torch.cuda.max_memory_allocated()
            
            memory_results['basic_hmm'] = {
                'base_memory_mb': base_memory / 1024**2,
                'peak_memory_mb': peak_memory / 1024**2,
                'forward_backward_memory_mb': (peak_memory - base_memory) / 1024**2
            }
            
            print(f"   Basic HMM: {(peak_memory - base_memory) / 1024**2:.1f} MB")
        
        # Neural HMM 메모리 사용량
        neural_hmm = NeuralHMM(
            num_states=10,  # 작게 설정
            observation_dim=40,
            context_dim=32,
            hidden_dim=64
        ).to(self.device)
        
        neural_obs = torch.randn(8, 100, 40).to(self.device)  # 작은 배치
        neural_context = torch.randn(8, 100, 32).to(self.device)
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            base_memory = torch.cuda.memory_allocated()
            neural_posteriors, _, _ = neural_hmm(neural_obs, neural_context)
            peak_memory = torch.cuda.max_memory_allocated()
            
            memory_results['neural_hmm'] = {
                'base_memory_mb': base_memory / 1024**2,
                'peak_memory_mb': peak_memory / 1024**2,
                'forward_memory_mb': (peak_memory - base_memory) / 1024**2
            }
            
            print(f"   Neural HMM: {(peak_memory - base_memory) / 1024**2:.1f} MB")
        
        return memory_results
    
    def run_full_benchmark(self) -> Dict:
        """전체 벤치마크 실행"""
        print("Starting comprehensive performance benchmark...\n")
        
        all_results = {
            'config': self.config.__dict__,
            'device': self.device,
            'timestamp': time.time()
        }
        
        if self.config.test_basic_hmm:
            all_results['basic_hmm'] = self.benchmark_basic_hmm()
        
        if self.config.test_neural_hmm:
            all_results['neural_hmm'] = self.benchmark_neural_hmm()
        
        if self.config.test_alignment:
            all_results['alignment'] = self.benchmark_alignment_algorithms()
        
        if self.config.test_semi_markov:
            all_results['semi_markov'] = self.benchmark_semi_markov_hmm()
        
        if self.config.measure_memory:
            all_results['memory'] = self.benchmark_memory_usage()
        
        return all_results
    
    def generate_report(self, results: Dict) -> str:
        """벤치마크 결과 리포트 생성"""
        report = []
        report.append("=" * 80)
        report.append("🎯 PyTorch HMM Performance Benchmark Report")
        report.append("=" * 80)
        
        # 시스템 정보
        report.append(f"\n📋 System Information:")
        report.append(f"   Device: {results['device']}")
        if torch.cuda.is_available():
            report.append(f"   GPU: {torch.cuda.get_device_name()}")
            report.append(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        report.append(f"   PyTorch Version: {torch.__version__}")
        
        # 기본 HMM 성능 요약
        if 'basic_hmm' in results:
            report.append(f"\n🚀 Basic HMM Performance Summary:")
            
            # 실시간 처리 가능한 설정 찾기
            realtime_configs = []
            for state_key, state_data in results['basic_hmm'].items():
                for config_key, config_data in state_data.items():
                    if isinstance(config_data, dict) and 'realtime_capable' in config_data:
                        if config_data['realtime_capable']['viterbi']:
                            realtime_configs.append({
                                'states': state_key,
                                'config': config_key,
                                'viterbi_fps': config_data['viterbi_fps']
                            })
            
            if realtime_configs:
                best_realtime = max(realtime_configs, key=lambda x: x['viterbi_fps'])
                report.append(f"   ✅ Real-time capable (>80fps): YES")
                report.append(f"   📊 Best real-time config: {best_realtime['states']}, {best_realtime['config']}")
                report.append(f"   🚄 Peak Viterbi performance: {best_realtime['viterbi_fps']:.0f} fps")
            else:
                report.append(f"   ❌ Real-time capable (>80fps): NO")
        
        # Neural HMM 성능 요약
        if 'neural_hmm' in results:
            report.append(f"\n🧠 Neural HMM Performance Summary:")
            
            # 최고 성능 찾기
            best_performance = 0
            best_config = ""
            for state_key, state_data in results['neural_hmm'].items():
                for config_key, config_data in state_data.items():
                    if isinstance(config_data, dict) and 'viterbi_fps' in config_data:
                        if config_data['viterbi_fps'] > best_performance:
                            best_performance = config_data['viterbi_fps']
                            best_config = f"{state_key}, {config_key}"
            
            report.append(f"   🚄 Peak performance: {best_performance:.0f} fps ({best_config})")
            if best_performance > 80:
                report.append(f"   ✅ Real-time capable: YES")
            else:
                report.append(f"   ⚠️ Real-time capable: Limited configurations")
        
        # 정렬 알고리즘 성능 요약
        if 'alignment' in results:
            report.append(f"\n🎯 Alignment Algorithm Performance:")
            
            if 'dtw' in results['alignment']:
                # DTW 최고 효율성
                best_dtw_efficiency = 0
                for config_key, config_data in results['alignment']['dtw'].items():
                    if 'efficiency' in config_data:
                        if config_data['efficiency'] > best_dtw_efficiency:
                            best_dtw_efficiency = config_data['efficiency']
                
                report.append(f"   📐 DTW peak efficiency: {best_dtw_efficiency:.0f} ops/s")
            
            if 'ctc' in results['alignment']:
                # CTC 최고 성능
                best_ctc_fps = 0
                for config_key, config_data in results['alignment']['ctc'].items():
                    if 'decode_fps' in config_data:
                        if config_data['decode_fps'] > best_ctc_fps:
                            best_ctc_fps = config_data['decode_fps']
                
                report.append(f"   📝 CTC peak decode: {best_ctc_fps:.0f} fps")
                if best_ctc_fps > 80:
                    report.append(f"   ✅ CTC real-time capable: YES")
        
        # 메모리 사용량 요약
        if 'memory' in results:
            report.append(f"\n💾 Memory Usage Summary:")
            if 'basic_hmm' in results['memory']:
                basic_mem = results['memory']['basic_hmm']['forward_backward_memory_mb']
                report.append(f"   📊 Basic HMM: {basic_mem:.1f} MB")
            
            if 'neural_hmm' in results['memory']:
                neural_mem = results['memory']['neural_hmm']['forward_memory_mb']
                report.append(f"   🧠 Neural HMM: {neural_mem:.1f} MB")
        
        # 권장사항
        report.append(f"\n💡 Recommendations:")
        report.append(f"   • Use Viterbi decoding for real-time applications")
        report.append(f"   • Batch processing improves GPU utilization")
        report.append(f"   • Consider model complexity vs performance trade-offs")
        report.append(f"   • Neural HMMs require more computational resources")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def main():
    """메인 벤치마크 실행"""
    parser = argparse.ArgumentParser(description='PyTorch HMM Performance Benchmark')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--seq-lengths', nargs='+', type=int, default=[50, 100, 200],
                       help='Sequence lengths to test')
    parser.add_argument('--num-states', nargs='+', type=int, default=[5, 10, 20],
                       help='Number of states to test')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per test')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with limited configurations')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--skip-neural', action='store_true',
                       help='Skip neural HMM tests (faster)')
    parser.add_argument('--skip-alignment', action='store_true',
                       help='Skip alignment algorithm tests')
    parser.add_argument('--skip-semi-markov', action='store_true',
                       help='Skip semi-Markov HMM tests')
    
    args = parser.parse_args()
    
    # 빠른 벤치마크 설정
    if args.quick:
        config = BenchmarkConfig(
            batch_sizes=[1, 4],
            sequence_lengths=[50, 100],
            num_states_list=[5, 10],
            num_iterations=5,
            warmup_iterations=2
        )
    else:
        config = BenchmarkConfig(
            batch_sizes=args.batch_sizes,
            sequence_lengths=args.seq_lengths,
            num_states_list=args.num_states,
            num_iterations=args.iterations
        )
    
    # 설정 적용
    config.use_gpu = not args.no_gpu
    config.test_neural_hmm = not args.skip_neural
    config.test_alignment = not args.skip_alignment
    config.test_semi_markov = not args.skip_semi_markov
    
    # 벤치마크 실행
    benchmark = PerformanceBenchmark(config)
    results = benchmark.run_full_benchmark()
    
    # 리포트 생성 및 출력
    report = benchmark.generate_report(results)
    print("\n" + report)
    
    # 결과 저장
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.save_results}")
    
    # 성능 요약
    print(f"\n🎯 Quick Performance Summary:")
    if 'basic_hmm' in results:
        # 가장 좋은 성능 찾기
        best_fps = 0
        for state_data in results['basic_hmm'].values():
            for config_data in state_data.values():
                if isinstance(config_data, dict) and 'viterbi_fps' in config_data:
                    best_fps = max(best_fps, config_data['viterbi_fps'])
        
        print(f"   🚄 Peak Viterbi performance: {best_fps:.0f} fps")
        print(f"   ⚡ Real-time capable: {'YES' if best_fps > 80 else 'NO'}")
        
        # 실시간 처리 권장 설정
        if best_fps > 80:
            print(f"   💡 Recommended for real-time: Small batches, moderate sequence lengths")
        else:
            print(f"   💡 Consider: Smaller models or GPU acceleration")


if __name__ == "__main__":
    main()
