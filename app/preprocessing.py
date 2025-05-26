"""Audio preprocessing utilities"""
import numpy as np
import librosa
import soundfile as sf
import io
from typing import Tuple
from .config import config


class AudioPreprocessor:
    def __init__(self):
        self.target_sr = config.SAMPLE_RATE
        self.n_fft = config.N_FFT
        self.hop_length = config.HOP_LENGTH
        self.win_length = config.WIN_LENGTH
        self.n_mels = config.N_MELS

    def load_audio(self, audio_data: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from bytes"""
        audio, sr = sf.read(io.BytesIO(audio_data))

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return audio, sr

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
        return audio

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1]"""
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-spectrogram features"""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.target_sr // 2,
            power=2.0
        )

        # Convert to log scale
        log_mel_spec = np.log(mel_spec + 1e-9)

        # Normalize per feature
        mean = np.mean(log_mel_spec, axis=1, keepdims=True)
        std = np.std(log_mel_spec, axis=1, keepdims=True)
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-9)

        return log_mel_spec.astype(np.float32)

    def process(self, audio_data: bytes) -> Tuple[np.ndarray, float, int]:
        """Complete preprocessing pipeline"""
        # Load audio
        audio, sr = self.load_audio(audio_data)

        # Resample if needed
        audio = self.resample(audio, sr)

        # Calculate duration
        duration = len(audio) / self.target_sr

        # Normalize
        audio = self.normalize_audio(audio)

        return audio, duration, self.target_sr


# Global preprocessor instance
preprocessor = AudioPreprocessor()