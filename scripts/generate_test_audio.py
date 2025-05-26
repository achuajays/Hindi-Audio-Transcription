"""
Generate test audio files by extracting random 8-second segments
and converting them to 16kHz WAV format
"""
import os
import numpy as np
import soundfile as sf
import librosa
import argparse
from pathlib import Path
import random


def extract_random_segment(audio, sr, duration=8.0):
    """
    Extract a random segment of specified duration from audio

    Args:
        audio: Audio signal array
        sr: Sample rate
        duration: Duration in seconds

    Returns:
        Extracted audio segment
    """
    target_samples = int(duration * sr)

    # If audio is shorter than target duration, pad with silence
    if len(audio) < target_samples:
        padding = target_samples - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        return audio

    # If audio is longer, extract random segment
    max_start = len(audio) - target_samples
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + target_samples

    return audio[start_idx:end_idx]


def process_audio_file(input_path, output_path, segment_duration=8.0, target_sr=16000):
    """
    Process a single audio file: extract random segment and convert to 16kHz

    Args:
        input_path: Path to input audio file
        output_path: Path to save output audio file
        segment_duration: Duration of segment to extract (seconds)
        target_sr: Target sample rate (Hz)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        print(f"Loaded: {input_path}")
        print(f"  Original duration: {len(audio) / sr:.2f} seconds")
        print(f"  Original sample rate: {sr} Hz")

        # Extract random segment
        segment = extract_random_segment(audio, sr, segment_duration)

        # Resample to target sample rate if needed
        if sr != target_sr:
            segment = librosa.resample(segment, orig_sr=sr, target_sr=target_sr)
            print(f"  Resampled to {target_sr} Hz")

        # Normalize audio to prevent clipping
        if np.max(np.abs(segment)) > 0:
            segment = segment / np.max(np.abs(segment)) * 0.95

        # Save as WAV file
        sf.write(output_path, segment, target_sr, subtype='PCM_16')
        print(f"  Saved: {output_path}")
        print(f"  Output duration: {len(segment) / target_sr:.2f} seconds")

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def generate_test_dataset(input_dir, output_dir, num_samples=10, segment_duration=8.0, target_sr=16000):
    """
    Generate multiple test audio files from a directory of audio files

    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save test audio files
        num_samples: Number of test samples to generate
        segment_duration: Duration of each segment (seconds)
        target_sr: Target sample rate (Hz)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(Path(input_dir).glob(f'**/*{ext}'))

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio files")

    # Generate test samples
    generated = 0
    attempts = 0
    max_attempts = num_samples * 3  # Allow some failures

    while generated < num_samples and attempts < max_attempts:
        # Select random audio file
        source_file = random.choice(audio_files)

        # Generate output filename
        output_filename = f"test_audio_{generated + 1:03d}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # Process the file
        if process_audio_file(source_file, output_path, segment_duration, target_sr):
            generated += 1
            print(f"Generated {generated}/{num_samples} test files\n")

        attempts += 1

    print(f"\nGenerated {generated} test audio files in {output_dir}")


def generate_single_test_audio(input_file, output_file="test_audio_8sec_16khz.wav"):
    """
    Generate a single test audio file from an input file

    Args:
        input_file: Path to input audio file
        output_file: Path to save output audio file
    """
    process_audio_file(input_file, output_file, segment_duration=8.0, target_sr=16000)


# Standalone script for quick generation
def create_synthetic_test_audio(output_file="synthetic_test_8sec_16khz.wav", duration=8.0, sr=16000):
    """
    Create a synthetic test audio file with multiple tones
    """
    t = np.linspace(0, duration, int(sr * duration))

    # Create a more interesting test signal
    # Mix of different frequencies
    signal = np.zeros_like(t)

    # Add some tones
    signal += 0.3 * np.sin(2 * np.pi * 440 * t)  # A4
    signal += 0.2 * np.sin(2 * np.pi * 554.37 * t)  # C#5
    signal += 0.2 * np.sin(2 * np.pi * 659.25 * t)  # E5

    # Add some amplitude modulation
    signal *= (1 + 0.3 * np.sin(2 * np.pi * 2 * t))

    # Add a frequency sweep
    sweep_freq = 200 + 800 * t / duration
    signal += 0.2 * np.sin(2 * np.pi * sweep_freq * t)

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    # Save
    sf.write(output_file, signal, sr, subtype='PCM_16')
    print(f"Created synthetic test audio: {output_file}")
    print(f"Duration: {duration} seconds, Sample rate: {sr} Hz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test audio files")
    parser.add_argument("--input", "-i", help="Input audio file or directory")
    parser.add_argument("--output", "-o", default="test_audio_8sec_16khz.wav",
                        help="Output file or directory")
    parser.add_argument("--num-samples", "-n", type=int, default=10,
                        help="Number of test samples to generate (for directory input)")
    parser.add_argument("--duration", "-d", type=float, default=8.0,
                        help="Duration of each segment in seconds")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000,
                        help="Target sample rate in Hz")
    parser.add_argument("--synthetic", "-s", action="store_true",
                        help="Generate synthetic test audio instead")

    args = parser.parse_args()

    if args.synthetic:
        # Generate synthetic test audio
        create_synthetic_test_audio(args.output, args.duration, args.sample_rate)
    elif args.input:
        if os.path.isdir(args.input):
            # Process directory
            output_dir = args.output if os.path.isdir(args.output) else "test_audio_dataset"
            generate_test_dataset(args.input, output_dir, args.num_samples,
                                  args.duration, args.sample_rate)
        else:
            # Process single file
            generate_single_test_audio(args.input, args.output)
    else:
        # No input specified, create synthetic audio
        print("No input specified, generating synthetic test audio...")
        create_synthetic_test_audio("test_audio_8sec_16khz.wav", 8.0, 16000)

