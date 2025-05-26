# Hindi ASR API - NVIDIA NeMo Conformer Model

A production-ready FastAPI-based REST API for Hindi Automatic Speech Recognition using NVIDIA NeMo's `stt_hi_conformer_ctc_medium` model optimized with ONNX Runtime.

## 📋 Table of Contents
- Overview
- Features
- Architecture
- Prerequisites
- Quick Start
- API Documentation
- Testing
- Design Considerations
- Performance
- Troubleshooting

## 🎯 Overview

This API provides Hindi speech-to-text transcription capabilities using a state-of-the-art Conformer-based ASR model from NVIDIA NeMo. The model has been optimized for inference using ONNX Runtime and deployed as a containerized microservice.

### Model Details
- **Model**: `stt_hi_conformer_ctc_medium`
- **Architecture**: Conformer with CTC decoder
- **Tokenization**: BPE (Byte Pair Encoding) with 128 tokens
- **Input**: 16kHz WAV audio files (5-10 seconds recommended)
- **Output**: Hindi text transcription

## ✨ Features

- 🚀 **High Performance**: ONNX Runtime optimized inference
- 🔄 **Async Processing**: Non-blocking API with concurrent request handling
- 🐳 **Containerized**: Docker support for easy deployment
- 📊 **Auto Documentation**: Swagger UI and ReDoc integration
- 🛡️ **Input Validation**: Comprehensive validation for audio files
- 📈 **Scalable**: Thread pool executor for parallel processing
- 🔍 **Health Monitoring**: Built-in health check endpoints

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client App    │────▶│   FastAPI Server │────▶│  ONNX Runtime   │
│  (Audio Upload) │     │   (Async Handler)│     │ (Model Inference)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────┐           ┌──────────────┐
                        │Preprocessing │           │ BPE Decoder  │
                        │(Mel Features)│           │ (Text Output)│
                        └──────────────┘           └──────────────┘
```

## 📦 Prerequisites

### Local Development
- Python 3.9+
- Docker and Docker Compose
- 4GB+ RAM recommended
- ONNX model file (`asr_model_hi.onnx`)

### Model Files
1. Download the model from [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium)
2. Export to ONNX format (if not already done)
3. Place in the `model/` directory

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd hindi-asr-api
```

### 2. Prepare Model Files
```bash
# Create model directory
mkdir -p model

# if you dont have the model file, Download the model from Nvidia NGC and  convert to onnx format
python setup_model.py

# Place your ONNX model file here
cp /path/to/asr_model_hi.onnx model/


```

### 3. Build and Run with Docker

#### Using Docker Compose (Recommended)
```bash
# Build and start the container
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

#### Using Docker Directly
```bash
# Build the image
docker build -t hindi-asr-api .

# Run the container
docker run -d \
  --name hindi-asr \
  -p 8000:8000 \
  -v $(pwd)/model:/app/model \
  hindi-asr-api
```

### 4. Verify Installation
```bash
# Check if the service is running
curl http://localhost:8000/

# Expected response:
# {"status":"healthy","service":"Hindi ASR API","version":"1.0.0"}
```

## 📡 API Documentation

### Endpoints

#### 1. Health Check
```bash
GET /
```

#### 2. Vocabulary Information
```bash
GET /vocab
```

#### 3. Transcribe Audio
```bash
POST /transcribe
Content-Type: multipart/form-data
Body: file (audio/wav)
```

### Sample Requests

#### Using cURL
```bash
# Basic transcription request
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample_audio.wav" \
  | jq .

# With custom headers
curl -X POST http://localhost:8000/transcribe \
  -H "Accept: application/json" \
  -F "file=@sample_audio.wav"

# Save response to file
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample_audio.wav" \
  -o response.json
```

#### Using Python
```python
import requests

# Transcribe audio file
with open("sample_audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": ("audio.wav", f, "audio/wav")}
    )
    
result = response.json()
print(f"Transcription: {result['text']}")
print(f"Duration: {result['duration']} seconds")
```

#### Using Postman
1. Create a new POST request to `http://localhost:8000/transcribe`
2. In the Body tab, select `form-data`
3. Add a key named `file` with type `File`
4. Select your WAV audio file
5. Send the request

### Sample Response
```json
{
  "text": "नमस्ते यह एक परीक्षण है",
  "duration": 5.2,
  "sample_rate": 16000
}
```

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing

### Run Unit Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

### Test with Sample Audio
```bash
# Generate test audio (optional)
python scripts/generate_test_audio.py

# Test transcription
curl -X POST http://localhost:8000/transcribe \
  -F "file=@tests/test_audio.wav"
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load test (create locustfile.py first)
locust -f locustfile.py --host=http://localhost:8000
```

## 🎨 Design Considerations

### 1. **Async-Compatible Inference Pipeline**
- **Challenge**: ONNX Runtime doesn't provide native async Python bindings
- **Solution**: Implemented thread pool executor with `asyncio.run_in_executor()`
- **Benefit**: Non-blocking API that can handle multiple concurrent requests

```python
async def run_inference_async(audio: np.ndarray) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, asr_model.transcribe, audio)
```

### 2. **BPE Tokenization Handling**
- **Challenge**: Model uses subword tokenization with special markers
- **Solution**: Proper vocabulary mapping with word boundary handling (`▁`)
- **Benefit**: Accurate text reconstruction from model outputs

### 3. **Memory-Efficient Audio Processing**
- **Challenge**: Large audio files can consume significant memory
- **Solution**: Stream processing with chunking for longer files
- **Benefit**: Consistent memory usage regardless of file size

### 4. **Containerization Strategy**
- **Base Image**: `python:3.9-slim` for minimal footprint
- **Multi-stage Build**: Separate build and runtime stages (optional)
- **Dependencies**: Only essential system libraries installed
- **Result**: Image size < 1GB with all dependencies

### 5. **Error Handling and Validation**
- **File Type**: Strict WAV format validation
- **Duration**: Configurable min/max duration limits
- **Sample Rate**: Automatic resampling to 16kHz
- **Empty Files**: Explicit validation and error messages

### 6. **Scalability Considerations**
- **Thread Pool**: Configurable worker count via `MAX_WORKERS`
- **Resource Limits**: Semaphore-based rate limiting
- **Monitoring**: Performance metrics endpoint (optional)

## 📊 Performance

### Benchmarks
- **Latency**: ~200-500ms for 5-second audio (CPU)
- **Throughput**: 10-20 requests/second (4 CPU cores)
- **Memory**: ~500MB base + 100MB per concurrent request
- **Model Load Time**: ~2-3 seconds on startup

### Optimization Tips
1. **CPU Inference**: Use more CPU cores for better throughput
   ```bash
   docker run -e MAX_WORKERS=8 --cpus="8" hindi-asr-api
   ```

2. **GPU Inference**: Modify Dockerfile for GPU support
   ```dockerfile
   # Use NVIDIA base image
   FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
   # Install ONNX Runtime GPU
   pip install onnxruntime-gpu
   ```

3. **Batch Processing**: Implement batch endpoint for multiple files
   ```python
   @app.post("/transcribe-batch")
   async def transcribe_batch(files: List[UploadFile]):
       # Process multiple files concurrently
   ```

## 🔧 Configuration

### Environment Variables
```bash
# Model configuration
MODEL_PATH=/app/model/asr_model_hi.onnx
VOCAB_PATH=/app/model/vocab.txt

# API configuration
HOST=0.0.0.0
PORT=8000
MAX_WORKERS=4

# Audio constraints
MIN_DURATION=0.5
MAX_DURATION=15.0
SAMPLE_RATE=16000
```

### Docker Compose Override
```yaml
# docker-compose.override.yml
version: '3.8'

services:
  asr-api:
    environment:
      - MAX_WORKERS=8
      - LOG_LEVEL=DEBUG
    cpus: '8'
    mem_limit: 4g
```

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**
   ```bash
   # Ensure model file exists
   ls -la model/asr_model_hi.onnx
   ```

2. **Out of memory errors**
   ```bash
   # Increase Docker memory limit
   docker run -m 4g hindi-asr-api
   ```

3. **Slow inference**
   ```bash
   # Check CPU usage
   docker stats hindi-asr
   
   # Increase workers
   docker run -e MAX_WORKERS=8 hindi-asr-api
   ```

4. **Audio format errors**
   ```bash
   # Convert audio to correct format
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

### Debug Mode
```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG hindi-asr-api

# Check container logs
docker logs hindi-asr -f
```

## 📄 License

This project is licensed under the MIT License. The NVIDIA NeMo model is subject to its own license terms.

## 🙏 Acknowledgments

- NVIDIA NeMo team for the pre-trained Hindi ASR model
- FastAPI for the excellent web framework
- ONNX Runtime team for optimization tools

