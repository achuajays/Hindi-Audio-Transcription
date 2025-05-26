"""Pydantic models for API"""
from pydantic import BaseModel

class TranscriptionResponse(BaseModel):
    text: str
    duration: float
    sample_rate: int

class ErrorResponse(BaseModel):
    error: str
    detail: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class VocabularyInfo(BaseModel):
    vocab_size: int
    blank_token_index: int
    sample_tokens: dict