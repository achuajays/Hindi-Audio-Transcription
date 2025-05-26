"""Model inference logic"""
import numpy as np
import onnxruntime as ort
from typing import Dict, Optional
from .config import config
from .vocabulary import vocabulary
from .preprocessing import preprocessor


class ASRModel:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None

    def load(self, model_path: str):
        """Load ONNX model"""
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        print(f"Model loaded from {model_path}")

        # Print model info
        print("Model inputs:")
        for input in self.session.get_inputs():
            print(f"  - {input.name}: {input.shape}")

    def prepare_inputs(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare model inputs"""
        # Compute mel-spectrogram
        mel_features = preprocessor.compute_mel_spectrogram(audio)

        # Add batch dimension
        audio_signal = np.expand_dims(mel_features, axis=0)

        # Create length array
        length = np.array([mel_features.shape[1]], dtype=np.int64)

        return {
            "audio_signal": audio_signal,
            "length": length
        }

    def decode_predictions(self, logprobs: np.ndarray) -> str:
        """Decode CTC predictions to text"""
        # Get most likely tokens
        predictions = np.argmax(logprobs[0], axis=-1)

        # CTC decoding
        decoded_tokens = []
        prev_idx = -1

        for idx in predictions:
            # Skip blanks and repeated tokens
            if idx != config.BLANK_TOKEN_INDEX and idx != prev_idx:
                token = vocabulary.get_token(idx)
                if token:
                    decoded_tokens.append(token)
            prev_idx = idx

        # Join tokens
        text = ''.join(decoded_tokens)

        # Replace word boundaries with spaces
        text = text.replace('▁', ' ')

        # Clean up
        while '  ' in text:
            text = text.replace('  ', ' ')

        return text.strip()

    def transcribe(self, audio: np.ndarray) -> str:
        """Run inference on audio"""
        if self.session is None:
            raise RuntimeError("Model not loaded")

        # Prepare inputs
        inputs = self.prepare_inputs(audio)

        # Run inference
        outputs = self.session.run(None, inputs)

        # Decode
        text = self.decode_predictions(outputs[0])

        return text


# Global model instance
asr_model = ASRModel()