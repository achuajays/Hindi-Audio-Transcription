"""Main FastAPI application"""
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .config import config
from .models import TranscriptionResponse, HealthResponse, VocabularyInfo
from .vocabulary import vocabulary
from .preprocessing import preprocessor
from .inference import asr_model

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    try:
        # Load vocabulary from file if available
        if os.path.exists(config.VOCAB_PATH):
            vocabulary.load_from_file(config.VOCAB_PATH)

        # Load model
        asr_model.load(config.MODEL_PATH)

        print("ASR API initialized successfully")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        raise

    yield

    # Shutdown
    executor.shutdown(wait=True)

# Initialize FastAPI app
app = FastAPI(
    title="Hindi ASR API",
    description="Automatic Speech Recognition API for Hindi using NeMo Conformer model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_inference_async(audio: np.ndarray) -> str:
    """Run inference in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, asr_model.transcribe, audio)

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="Hindi ASR API",
        version="1.0.0"
    )

@app.get("/vocab", response_model=VocabularyInfo)
async def get_vocabulary_info():
    """Get vocabulary information"""
    return VocabularyInfo(
        vocab_size=len(vocabulary),
        blank_token_index=config.BLANK_TOKEN_INDEX,
        sample_tokens={
            "1": vocabulary.get_token(1),
            "4": vocabulary.get_token(4),
            "9": vocabulary.get_token(9),
            "15": vocabulary.get_token(15),
            "22": vocabulary.get_token(22),
            "128": "BLANK"
        }
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe Hindi audio to text

    Args:
        file: WAV audio file (16kHz recommended, 0.5-10 seconds)

    Returns:
        Transcribed text with metadata
    """
    # Validate file type
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported"
        )

    try:
        # Read file
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(
                status_code=400,
                detail="Empty audio file"
            )

        # Preprocess audio
        audio, duration, sample_rate = preprocessor.process(audio_data)

        # Validate duration
        if duration < config.MIN_DURATION or duration > config.MAX_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Audio duration must be between {config.MIN_DURATION} and {config.MAX_DURATION} seconds. Got {duration:.2f} seconds"
            )

        # Run inference
        text = await run_inference_async(audio)

        return TranscriptionResponse(
            text=text or "",
            duration=float(duration),
            sample_rate=int(sample_rate)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )