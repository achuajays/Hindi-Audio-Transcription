"""Configuration settings for the ASR API"""
import os
from typing import List


class Config:
    # Model settings
    MODEL_PATH = os.environ.get("MODEL_PATH", "./model/asr_model_hi.onnx")
    VOCAB_PATH = os.environ.get("VOCAB_PATH", "./model/vocab.txt")

    # Audio settings
    SAMPLE_RATE = 16000
    MIN_DURATION = 0.5
    MAX_DURATION = 10.0

    # Mel-spectrogram settings
    N_FFT = 512
    HOP_LENGTH = 160  # 10ms at 16kHz
    WIN_LENGTH = 320  # 20ms at 16kHz
    N_MELS = 80

    # API settings
    MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8000))

    # CTC settings
    BLANK_TOKEN_INDEX = 128


config = Config()