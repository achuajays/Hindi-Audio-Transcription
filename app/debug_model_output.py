# debug_model_output.py
import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf
import io


def detailed_debug(model_path, audio_path):
    """Detailed debugging of model outputs"""
    # Load model
    session = ort.InferenceSession(model_path)

    # Load and preprocess audio
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_fft=512, hop_length=160,
        win_length=320, n_mels=80
    )
    log_mel = np.log(mel_spec + 1e-9)

    # Normalize
    mean = np.mean(log_mel, axis=1, keepdims=True)
    std = np.std(log_mel, axis=1, keepdims=True)
    log_mel = (log_mel - mean) / (std + 1e-9)

    # Prepare inputs
    audio_signal = np.expand_dims(log_mel, axis=0).astype(np.float32)
    length = np.array([log_mel.shape[1]], dtype=np.int64)

    # Run inference
    outputs = session.run(None, {"audio_signal": audio_signal, "length": length})
    logprobs = outputs[0]

    print(f"Output shape: {logprobs.shape}")
    print(f"Output dtype: {logprobs.dtype}")
    print(f"Output range: [{np.min(logprobs):.2f}, {np.max(logprobs):.2f}]")

    # Check if outputs are log probabilities or probabilities
    if np.min(logprobs) < 0:
        print("Output appears to be log probabilities")
        # Convert to probabilities for analysis
        probs = np.exp(logprobs)
    else:
        print("Output appears to be probabilities")
        probs = logprobs

    # Get predictions
    predictions = np.argmax(logprobs[0], axis=-1)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Unique predictions: {sorted(np.unique(predictions))}")
    print(f"First 30 predictions: {predictions[:30].tolist()}")

    # Analyze the most confident predictions
    print("\nMost confident predictions at each timestep (first 10):")
    for t in range(min(10, logprobs.shape[1])):
        top_indices = np.argsort(logprobs[0, t])[-5:][::-1]
        top_probs = logprobs[0, t][top_indices]
        print(f"  Time {t}: {list(zip(top_indices.tolist(), top_probs.tolist()))}")

    # Try different decoding strategies
    print("\n--- Trying different decoding strategies ---")

    # Strategy 1: Direct mapping
    print("\nStrategy 1 - Direct mapping:")
    decoded = []
    prev = -1
    for idx in predictions:
        if idx != 0 and idx != prev:
            decoded.append(idx)
        prev = idx
    print(f"Decoded indices: {decoded[:20]}")

    # Strategy 2: Check if blank is at a different index
    print("\nStrategy 2 - Finding blank token:")
    # Count occurrences of each index
    unique, counts = np.unique(predictions, return_counts=True)
    sorted_by_count = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
    print(f"Most common indices: {sorted_by_count[:10]}")

    return logprobs, predictions


# Run the debug
if __name__ == "__main__":
    model_path = "./model/asr_model_hi.onnx"
    audio_path = "./audio/output_8s_16khz.wav"
    logprobs, predictions = detailed_debug(model_path, audio_path)