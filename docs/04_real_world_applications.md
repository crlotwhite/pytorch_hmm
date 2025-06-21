# PyTorch HMM 라이브러리 실제 응용 사례

이 문서에서는 PyTorch HMM 라이브러리를 실제 음성 처리 프로젝트에 적용하는 방법을 다룹니다.

## 목차

1. [음성 합성 (TTS) 시스템](#음성-합성-tts-시스템)
2. [음성 인식 (ASR) 시스템](#음성-인식-asr-시스템)
3. [음성 변환 (Voice Conversion)](#음성-변환-voice-conversion)
4. [감정 인식 시스템](#감정-인식-시스템)
5. [실시간 음성 처리](#실시간-음성-처리)
6. [다중 화자 시스템](#다중-화자-시스템)

## 음성 합성 (TTS) 시스템

### 기본 TTS 파이프라인

```python
import torch
import torch.nn as nn
from pytorch_hmm import (
    ContextualNeuralHMM, SemiMarkovHMM, DurationModel,
    DTWAligner, comprehensive_speech_evaluation
)

class TextToSpeechSystem:
    def __init__(self, vocab_size=256, phoneme_size=50):
        self.vocab_size = vocab_size
        self.phoneme_size = phoneme_size
        
        # 구성 요소 초기화
        self.text_encoder = self._build_text_encoder()
        self.phoneme_aligner = self._build_phoneme_aligner()
        self.duration_model = self._build_duration_model()
        self.acoustic_model = self._build_acoustic_model()
        self.vocoder = self._build_vocoder()
    
    def _build_text_encoder(self):
        """텍스트를 음소 시퀀스로 변환"""
        return nn.Sequential(
            nn.Embedding(self.vocab_size, 128),
            nn.LSTM(128, 64, batch_first=True, bidirectional=True),
            nn.Linear(128, self.phoneme_size)
        )
    
    def _build_phoneme_aligner(self):
        """음소와 음성 특징 정렬"""
        return DTWAligner(
            distance_fn='cosine',
            step_pattern='symmetric',
            window_type='sakoe_chiba',
            window_size=20
        )
    
    def _build_duration_model(self):
        """음소별 지속 시간 예측"""
        duration_model = DurationModel(
            num_states=self.phoneme_size,
            distribution_type='gamma',
            max_duration=100
        )
        
        return SemiMarkovHMM(
            num_states=self.phoneme_size,
            observation_dim=64,  # 언어적 특징 차원
            duration_model=duration_model
        )
    
    def _build_acoustic_model(self):
        """음향 특징 생성"""
        return ContextualNeuralHMM(
            num_states=self.phoneme_size * 3,  # 각 음소당 3개 상태
            observation_dim=80,  # 멜 스펙트로그램 차원
            context_dim=128,     # 언어적 + 운율 특징
            hidden_dim=256,
            num_context_layers=3,
            use_attention=True
        )
    
    def _build_vocoder(self):
        """멜 스펙트로그램을 음성 파형으로 변환"""
        return nn.Sequential(
            nn.ConvTranspose1d(80, 512, 4, 2, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 4, 2, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, 4, 2, 1),
            nn.Tanh()
        )
    
    def synthesize(self, text, speaker_id=None, emotion=None):
        """텍스트로부터 음성 합성"""
        # 1. 텍스트 인코딩
        phoneme_logits, _ = self.text_encoder(text)
        phoneme_probs = torch.softmax(phoneme_logits, dim=-1)
        
        # 2. 지속 시간 예측
        durations = self.duration_model.predict_durations(phoneme_probs)
        
        # 3. 컨텍스트 특징 생성
        context_features = self._create_context_features(
            phoneme_probs, durations, speaker_id, emotion
        )
        
        # 4. 음향 특징 생성
        mel_spectrogram = self.acoustic_model.generate(
            context_features, durations
        )
        
        # 5. 음성 파형 생성
        waveform = self.vocoder(mel_spectrogram.transpose(1, 2))
        
        return waveform.squeeze(), mel_spectrogram, durations
    
    def _create_context_features(self, phoneme_probs, durations, speaker_id, emotion):
        """컨텍스트 특징 생성"""
        batch_size, seq_len, _ = phoneme_probs.shape
        
        # 언어적 특징
        linguistic_features = phoneme_probs
        
        # 운율 특징 (간단한 예제)
        prosodic_features = torch.zeros(batch_size, seq_len, 32)
        if emotion is not None:
            emotion_embedding = torch.randn(32)  # 실제로는 학습된 임베딩
            prosodic_features += emotion_embedding.unsqueeze(0).unsqueeze(0)
        
        # 화자 특징
        speaker_features = torch.zeros(batch_size, seq_len, 32)
        if speaker_id is not None:
            speaker_embedding = torch.randn(32)  # 실제로는 학습된 임베딩
            speaker_features += speaker_embedding.unsqueeze(0).unsqueeze(0)
        
        # 위치 특징
        position_features = torch.zeros(batch_size, seq_len, 32)
        for i in range(seq_len):
            position_features[:, i, :] = torch.sin(torch.arange(32) * i / seq_len)
        
        # 모든 특징 결합
        context = torch.cat([
            linguistic_features,
            prosodic_features,
            speaker_features,
            position_features
        ], dim=-1)
        
        return context

# TTS 시스템 사용 예제
tts_system = TextToSpeechSystem()

# 텍스트 입력 (문자 ID 시퀀스)
text_input = torch.randint(0, 256, (1, 20))  # "Hello, world!" 등

# 음성 합성
waveform, mel_spec, durations = tts_system.synthesize(
    text_input, 
    speaker_id=1, 
    emotion='happy'
)

print(f"Generated waveform shape: {waveform.shape}")
print(f"Mel spectrogram shape: {mel_spec.shape}")
print(f"Predicted durations: {durations}")
```

### 다중 화자 TTS

```python
class MultiSpeakerTTS(TextToSpeechSystem):
    def __init__(self, vocab_size=256, phoneme_size=50, num_speakers=100):
        super().__init__(vocab_size, phoneme_size)
        self.num_speakers = num_speakers
        
        # 화자 임베딩 레이어
        self.speaker_embedding = nn.Embedding(num_speakers, 64)
        
        # 화자 적응 레이어
        self.speaker_adaptation = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)  # 음향 모델의 hidden_dim과 일치
        )
    
    def synthesize(self, text, speaker_id, emotion=None):
        """화자별 음성 합성"""
        # 화자 임베딩 획득
        speaker_emb = self.speaker_embedding(torch.tensor([speaker_id]))
        speaker_adapt = self.speaker_adaptation(speaker_emb)
        
        # 기본 합성 과정
        waveform, mel_spec, durations = super().synthesize(text, speaker_id, emotion)
        
        # 화자 적응 적용
        adapted_mel = self._apply_speaker_adaptation(mel_spec, speaker_adapt)
        adapted_waveform = self.vocoder(adapted_mel.transpose(1, 2))
        
        return adapted_waveform.squeeze(), adapted_mel, durations
    
    def _apply_speaker_adaptation(self, mel_spec, speaker_adapt):
        """화자 적응 변환 적용"""
        # 간단한 선형 변환 (실제로는 더 복잡한 변환 사용)
        batch_size, seq_len, mel_dim = mel_spec.shape
        
        # 화자 적응 가중치 생성
        adaptation_weights = speaker_adapt.view(1, 1, -1).expand(batch_size, seq_len, -1)
        
        # 적응된 멜 스펙트로그램 생성
        adapted_mel = mel_spec + adaptation_weights[:, :, :mel_dim]
        
        return adapted_mel

# 다중 화자 TTS 사용
multi_speaker_tts = MultiSpeakerTTS(num_speakers=10)

# 다양한 화자로 같은 텍스트 합성
text = torch.randint(0, 256, (1, 15))
speakers = [0, 1, 2, 3, 4]

for speaker_id in speakers:
    waveform, _, _ = multi_speaker_tts.synthesize(text, speaker_id)
    print(f"Speaker {speaker_id} waveform shape: {waveform.shape}")
```

## 음성 인식 (ASR) 시스템

### End-to-End ASR 시스템

```python
from pytorch_hmm import CTCAligner, NeuralHMM

class SpeechRecognitionSystem:
    def __init__(self, vocab_size=1000, num_phonemes=50):
        self.vocab_size = vocab_size
        self.num_phonemes = num_phonemes
        
        # 구성 요소 초기화
        self.feature_extractor = self._build_feature_extractor()
        self.acoustic_model = self._build_acoustic_model()
        self.language_model = self._build_language_model()
        self.ctc_decoder = CTCAligner(vocab_size, blank_id=0)
        
    def _build_feature_extractor(self):
        """음성 신호에서 특징 추출"""
        return nn.Sequential(
            # 1D CNN for audio processing
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(80)  # 80차원 특징으로 압축
        )
    
    def _build_acoustic_model(self):
        """음향 모델 (음성 특징 -> 음소 확률)"""
        return NeuralHMM(
            num_states=self.num_phonemes * 3,  # 각 음소당 3개 상태
            observation_dim=80,
            context_dim=0,  # 컨텍스트 없음
            hidden_dim=512,
            transition_type='rnn'
        )
    
    def _build_language_model(self):
        """언어 모델"""
        return nn.Sequential(
            nn.LSTM(self.vocab_size, 256, batch_first=True, num_layers=2),
            nn.Linear(256, self.vocab_size)
        )
    
    def recognize(self, audio_waveform, beam_width=5):
        """음성 인식 수행"""
        # 1. 특징 추출
        features = self.feature_extractor(audio_waveform.unsqueeze(1))
        features = features.transpose(1, 2)  # (B, T, F)
        
        # 2. 음향 모델링
        phone_posteriors, _, _ = self.acoustic_model(features)
        
        # 3. CTC 디코딩
        input_lengths = torch.full((features.shape[0],), features.shape[1])
        
        # Greedy 디코딩
        greedy_result = self.ctc_decoder.decode(
            torch.log_softmax(phone_posteriors, dim=-1).transpose(0, 1),
            input_lengths
        )
        
        # Beam search 디코딩
        beam_results = self.ctc_decoder.beam_search_decode(
            torch.log_softmax(phone_posteriors, dim=-1).transpose(0, 1),
            input_lengths,
            beam_width=beam_width
        )
        
        return {
            'greedy': greedy_result,
            'beam_search': beam_results,
            'phone_posteriors': phone_posteriors
        }
    
    def train_step(self, audio_batch, text_targets, target_lengths):
        """학습 단계"""
        # 특징 추출
        features = self.feature_extractor(audio_batch)
        features = features.transpose(1, 2)
        
        # 음향 모델링
        phone_posteriors, _, _ = self.acoustic_model(features)
        
        # CTC Loss 계산
        input_lengths = torch.full((features.shape[0],), features.shape[1])
        ctc_loss = self.ctc_decoder(
            torch.log_softmax(phone_posteriors, dim=-1).transpose(0, 1),
            text_targets,
            input_lengths,
            target_lengths
        )
        
        return ctc_loss

# ASR 시스템 사용 예제
asr_system = SpeechRecognitionSystem()

# 음성 입력 (1초, 16kHz)
audio_input = torch.randn(2, 16000)  # 배치 크기 2

# 인식 수행
recognition_results = asr_system.recognize(audio_input, beam_width=10)

print("Recognition Results:")
print(f"Greedy: {recognition_results['greedy']}")
print(f"Beam search candidates: {len(recognition_results['beam_search'][0])}")
```

### 실시간 ASR 시스템

```python
class StreamingASR:
    def __init__(self, chunk_size=1600, overlap=400):  # 100ms 청크, 25ms 오버랩
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.asr_system = SpeechRecognitionSystem()
        
        # 스트리밍 상태
        self.audio_buffer = torch.zeros(0)
        self.feature_buffer = torch.zeros(0, 80)
        self.state_buffer = None
        
    def process_chunk(self, audio_chunk):
        """오디오 청크 처리"""
        # 버퍼에 새 오디오 추가
        self.audio_buffer = torch.cat([self.audio_buffer, audio_chunk])
        
        results = []
        
        # 충분한 데이터가 있으면 처리
        while len(self.audio_buffer) >= self.chunk_size:
            # 현재 청크 추출
            current_chunk = self.audio_buffer[:self.chunk_size]
            
            # 특징 추출
            features = self.asr_system.feature_extractor(current_chunk.unsqueeze(0).unsqueeze(0))
            features = features.squeeze(0).transpose(0, 1)  # (T, F)
            
            # 특징 버퍼에 추가
            self.feature_buffer = torch.cat([self.feature_buffer, features])
            
            # 인식 수행 (최근 N 프레임 사용)
            if self.feature_buffer.shape[0] >= 100:  # 최소 길이
                recent_features = self.feature_buffer[-100:].unsqueeze(0)  # (1, T, F)
                
                # 상태 연속성을 위한 처리
                with torch.no_grad():
                    phone_posteriors, forward, backward = self.asr_system.acoustic_model(recent_features)
                    
                    # 간단한 greedy 디코딩
                    predicted_phones = torch.argmax(phone_posteriors, dim=-1)
                    results.append(predicted_phones.squeeze().tolist())
            
            # 버퍼 업데이트 (오버랩 유지)
            self.audio_buffer = self.audio_buffer[self.chunk_size - self.overlap:]
            
            # 특징 버퍼 크기 제한
            if self.feature_buffer.shape[0] > 500:
                self.feature_buffer = self.feature_buffer[-300:]
        
        return results
    
    def reset(self):
        """스트리밍 상태 초기화"""
        self.audio_buffer = torch.zeros(0)
        self.feature_buffer = torch.zeros(0, 80)
        self.state_buffer = None

# 스트리밍 ASR 사용 예제
streaming_asr = StreamingASR()

# 시뮬레이션된 실시간 오디오 스트림
for i in range(10):  # 10개 청크
    # 100ms 오디오 청크 (16kHz)
    audio_chunk = torch.randn(1600)
    
    # 청크 처리
    results = streaming_asr.process_chunk(audio_chunk)
    
    if results:
        print(f"Chunk {i}: {results[-1][:10]}...")  # 처음 10개 음소만 출력
```

## 음성 변환 (Voice Conversion)

```python
class VoiceConversionSystem:
    def __init__(self, num_speakers=100):
        self.num_speakers = num_speakers
        
        # 구성 요소
        self.content_encoder = self._build_content_encoder()
        self.speaker_encoder = self._build_speaker_encoder()
        self.decoder = self._build_decoder()
        self.aligner = DTWAligner(distance_fn='cosine')
        
    def _build_content_encoder(self):
        """화자 독립적인 내용 인코더"""
        return NeuralHMM(
            num_states=50,  # 음소 수
            observation_dim=80,
            context_dim=0,
            hidden_dim=256
        )
    
    def _build_speaker_encoder(self):
        """화자 임베딩 인코더"""
        return nn.Sequential(
            nn.Conv1d(80, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128)
        )
    
    def _build_decoder(self):
        """멜 스펙트로그램 디코더"""
        return nn.Sequential(
            nn.Linear(256 + 128, 512),  # 내용 + 화자 특징
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 80)
        )
    
    def convert_voice(self, source_mel, target_speaker_mel):
        """음성 변환 수행"""
        # 1. 소스 음성에서 내용 특징 추출
        content_features, _, _ = self.content_encoder(source_mel.unsqueeze(0))
        content_features = content_features.squeeze(0)  # (T, H)
        
        # 2. 타겟 화자에서 화자 특징 추출
        speaker_features = self.speaker_encoder(target_speaker_mel.transpose(0, 1).unsqueeze(0))
        speaker_features = speaker_features.squeeze(0)  # (H,)
        
        # 3. 화자 특징을 시간 축으로 확장
        seq_len = content_features.shape[0]
        speaker_features = speaker_features.unsqueeze(0).expand(seq_len, -1)
        
        # 4. 내용과 화자 특징 결합
        combined_features = torch.cat([content_features, speaker_features], dim=-1)
        
        # 5. 변환된 멜 스펙트로그램 생성
        converted_mel = self.decoder(combined_features)
        
        return converted_mel

# 음성 변환 시스템 사용
vc_system = VoiceConversionSystem()

# 소스 화자 음성과 타겟 화자 음성
source_mel = torch.randn(100, 80)  # 소스 화자 멜 스펙트로그램
target_speaker_mel = torch.randn(80, 80)  # 타겟 화자 참조 음성

# 음성 변환 수행
converted_mel = vc_system.convert_voice(source_mel, target_speaker_mel)
print(f"Converted mel shape: {converted_mel.shape}")
```

## 감정 인식 시스템

```python
class EmotionRecognitionSystem:
    def __init__(self, num_emotions=7):  # 7가지 기본 감정
        self.num_emotions = num_emotions
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
        
        # 구성 요소
        self.feature_extractor = self._build_feature_extractor()
        self.emotion_classifier = self._build_emotion_classifier()
        self.temporal_model = self._build_temporal_model()
        
    def _build_feature_extractor(self):
        """음성에서 감정 관련 특징 추출"""
        return nn.Sequential(
            nn.Conv1d(80, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, 2, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
    
    def _build_emotion_classifier(self):
        """감정 분류기"""
        return nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_emotions)
        )
    
    def _build_temporal_model(self):
        """시간적 감정 변화 모델링"""
        return NeuralHMM(
            num_states=self.num_emotions,
            observation_dim=256,  # 특징 차원
            context_dim=0,
            hidden_dim=128
        )
    
    def recognize_emotion(self, mel_spectrogram, use_temporal=True):
        """감정 인식 수행"""
        # 멜 스펙트로그램을 청크로 분할
        chunk_size = 50  # 약 0.5초
        chunks = []
        
        for i in range(0, mel_spectrogram.shape[0], chunk_size):
            chunk = mel_spectrogram[i:i+chunk_size]
            if chunk.shape[0] < chunk_size:
                # 패딩
                padding = torch.zeros(chunk_size - chunk.shape[0], chunk.shape[1])
                chunk = torch.cat([chunk, padding], dim=0)
            chunks.append(chunk)
        
        # 각 청크에서 특징 추출
        chunk_features = []
        chunk_emotions = []
        
        for chunk in chunks:
            # 특징 추출
            features = self.feature_extractor(chunk.transpose(0, 1).unsqueeze(0))
            chunk_features.append(features.squeeze(0))
            
            # 감정 분류
            emotion_logits = self.emotion_classifier(features)
            chunk_emotions.append(emotion_logits.squeeze(0))
        
        chunk_features = torch.stack(chunk_features)  # (num_chunks, feature_dim)
        chunk_emotions = torch.stack(chunk_emotions)  # (num_chunks, num_emotions)
        
        if use_temporal:
            # 시간적 모델링 사용
            emotion_posteriors, _, _ = self.temporal_model(chunk_features.unsqueeze(0))
            emotion_posteriors = emotion_posteriors.squeeze(0)  # (num_chunks, num_emotions)
            
            # 전체 발화에 대한 감정 예측
            final_emotion = torch.mean(emotion_posteriors, dim=0)
        else:
            # 단순 평균
            final_emotion = torch.mean(chunk_emotions, dim=0)
        
        # 감정 확률과 예측 결과
        emotion_probs = torch.softmax(final_emotion, dim=0)
        predicted_emotion = torch.argmax(emotion_probs)
        
        return {
            'emotion': self.emotions[predicted_emotion],
            'probabilities': {
                emotion: prob.item() 
                for emotion, prob in zip(self.emotions, emotion_probs)
            },
            'temporal_emotions': [
                self.emotions[torch.argmax(chunk_emotion)] 
                for chunk_emotion in chunk_emotions
            ]
        }

# 감정 인식 시스템 사용
emotion_system = EmotionRecognitionSystem()

# 음성 입력 (멜 스펙트로그램)
speech_mel = torch.randn(200, 80)  # 2초 음성

# 감정 인식 수행
emotion_result = emotion_system.recognize_emotion(speech_mel)

print(f"Predicted emotion: {emotion_result['emotion']}")
print("Emotion probabilities:")
for emotion, prob in emotion_result['probabilities'].items():
    print(f"  {emotion}: {prob:.3f}")
print(f"Temporal emotions: {emotion_result['temporal_emotions']}")
```

## 마무리

이 문서에서는 PyTorch HMM 라이브러리를 사용한 다양한 실제 응용 사례를 살펴보았습니다:

- **음성 합성**: 텍스트를 자연스러운 음성으로 변환
- **음성 인식**: 음성을 텍스트로 변환
- **음성 변환**: 화자의 음성 특성 변경
- **감정 인식**: 음성에서 감정 상태 인식
- **실시간 처리**: 스트리밍 음성 처리
- **다중 화자**: 여러 화자를 지원하는 시스템

이러한 응용 사례들은 실제 제품이나 연구 프로젝트에서 바로 활용할 수 있는 형태로 구성되어 있습니다.

다음으로는 [성능 최적화 가이드](05_performance_optimization.md)를 참고하여 시스템의 성능을 향상시키는 방법을 알아보시기 바랍니다. 