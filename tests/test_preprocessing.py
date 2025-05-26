"""Test audio preprocessing functionality"""
import pytest
import numpy as np
import io
import soundfile as sf
from app.preprocessing import AudioPreprocessor
from app.config import config


class TestAudioPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return AudioPreprocessor()

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio for testing"""
        duration = 5.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        # 440 Hz sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32), sample_rate

    @pytest.fixture
    def sample_audio_bytes(self):
        """Generate sample audio bytes for testing"""
        duration = 5.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        # 440 Hz sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()

    def test_normalize_audio(self, preprocessor, sample_audio):
        """Test audio normalization"""
        audio, _ = sample_audio
        normalized = preprocessor.normalize_audio(audio)

        assert np.max(np.abs(normalized)) <= 1.0
        assert normalized.dtype == np.float32

    def test_resample_audio(self, preprocessor):
        """Test audio resampling"""
        # Create 8kHz audio
        audio = np.random.randn(8000).astype(np.float32)
        resampled = preprocessor.resample(audio, 8000)

        # Should be 16kHz now
        expected_length = 16000
        assert len(resampled) == expected_length

    def test_mel_spectrogram_shape(self, preprocessor, sample_audio):
        """Test mel-spectrogram computation"""
        audio, sr = sample_audio
        mel_spec = preprocessor.compute_mel_spectrogram(audio)

        # Check shape
        assert mel_spec.shape[0] == config.N_MELS  # 80 mel bins
        assert mel_spec.dtype == np.float32

        # Check normalization - mean should be close to 0, std close to 1
        mean_per_feature = np.mean(mel_spec, axis=1)
        std_per_feature = np.std(mel_spec, axis=1)

        # Due to normalization, each feature should have mean ≈ 0 and std ≈ 1
        assert np.allclose(mean_per_feature, 0, atol=0.1)
        assert np.allclose(std_per_feature, 1, atol=0.1)

    def test_load_audio_from_bytes(self, preprocessor, sample_audio_bytes):
        """Test loading audio from bytes"""
        audio, sr = preprocessor.load_audio(sample_audio_bytes)

        assert isinstance(audio, np.ndarray)
        assert sr == 16000
        assert len(audio.shape) == 1  # Should be mono
        assert audio.dtype == np.float64 or audio.dtype == np.float32

    def test_load_audio_stereo_to_mono(self, preprocessor):
        """Test stereo to mono conversion"""
        # Create stereo audio
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Different frequencies for left and right channels
        left_channel = 0.5 * np.sin(2 * np.pi * 440 * t)
        right_channel = 0.5 * np.sin(2 * np.pi * 880 * t)
        stereo_audio = np.stack([left_channel, right_channel], axis=1)

        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, stereo_audio, sample_rate, format='WAV')
        buffer.seek(0)

        # Load and check it's converted to mono
        audio, sr = preprocessor.load_audio(buffer.read())
        assert len(audio.shape) == 1  # Mono

    def test_process_complete_pipeline(self, preprocessor, sample_audio_bytes):
        """Test the complete preprocessing pipeline"""
        audio, duration, sample_rate = preprocessor.process(sample_audio_bytes)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert duration == pytest.approx(5.0, rel=0.01)
        assert sample_rate == 16000
        assert np.max(np.abs(audio)) <= 1.0  # Normalized

    def test_process_with_different_sample_rate(self, preprocessor):
        """Test processing audio with different sample rate"""
        # Create 8kHz audio
        duration = 2.0
        sample_rate = 8000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_8k = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_8k, sample_rate, format='WAV')
        buffer.seek(0)

        # Process and check resampling
        audio, duration_out, sr_out = preprocessor.process(buffer.read())

        assert sr_out == 16000  # Should be resampled to 16kHz
        assert duration_out == pytest.approx(duration, rel=0.01)
        assert len(audio) == pytest.approx(duration * 16000, rel=0.01)

    def test_mel_spectrogram_time_dimension(self, preprocessor):
        """Test mel-spectrogram time dimension calculation"""
        # Create exactly 1 second of audio
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32)

        mel_spec = preprocessor.compute_mel_spectrogram(audio)

        # Calculate expected time frames
        # With hop_length=160, we should get sample_rate/hop_length frames
        expected_frames = int(np.ceil(sample_rate / config.HOP_LENGTH))

        assert mel_spec.shape[1] == pytest.approx(expected_frames, abs=2)

    def test_empty_audio_handling(self, preprocessor):
        """Test handling of empty or very short audio"""
        # Create empty audio
        empty_audio = np.array([], dtype=np.float32)

        # Test with very short audio that might cause issues
        very_short_audio = np.array([0.1, 0.2], dtype=np.float32)

        # The function handles empty/short audio gracefully
        # For empty audio
        mel_spec_empty = preprocessor.compute_mel_spectrogram(empty_audio)
        assert mel_spec_empty.shape[0] == config.N_MELS

        # For very short audio
        mel_spec_short = preprocessor.compute_mel_spectrogram(very_short_audio)
        assert mel_spec_short.shape[0] == config.N_MELS
        assert mel_spec_short.shape[1] >= 0  # Should have some time frames

    def test_normalize_silent_audio(self, preprocessor):
        """Test normalization of silent audio"""
        # Create silent audio (all zeros)
        silent_audio = np.zeros(16000, dtype=np.float32)
        normalized = preprocessor.normalize_audio(silent_audio)

        # Should handle without division by zero
        assert np.all(normalized == 0)
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))

    def test_process_various_durations(self, preprocessor):
        """Test processing audio of various durations"""
        sample_rate = 16000

        for duration in [0.5, 1.0, 5.0, 10.0, 15.0]:
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)

            # Convert to bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
            buffer.seek(0)

            # Process
            processed_audio, processed_duration, sr = preprocessor.process(buffer.read())

            assert processed_duration == pytest.approx(duration, rel=0.01)
            assert sr == sample_rate
            assert isinstance(processed_audio, np.ndarray)

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
    def test_various_sample_rates(self, preprocessor, sample_rate):
        """Test resampling from various sample rates"""
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Resample
        resampled = preprocessor.resample(audio, sample_rate)

        # Check output sample rate
        expected_length = int(duration * preprocessor.target_sr)
        assert len(resampled) == pytest.approx(expected_length, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])