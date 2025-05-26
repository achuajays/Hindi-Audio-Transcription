"""API tests"""
import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data

def test_vocabulary_info():
    """Test vocabulary endpoint"""
    response = client.get("/vocab")
    assert response.status_code == 200
    data = response.json()
    assert "vocab_size" in data
    assert data["vocab_size"] == 128
    assert data["blank_token_index"] == 128

def test_transcribe_invalid_file():
    """Test with invalid file type"""
    response = client.post(
        "/transcribe",
        files={"file": ("test.txt", b"test content", "text/plain")}
    )
    assert response.status_code == 400
    assert "WAV files" in response.json()["detail"]

def test_transcribe_empty_file():
    """Test with empty file"""
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    assert "Empty" in response.json()["detail"]

@pytest.mark.skipif(not os.path.exists("tests/test_audio.wav"),
                    reason="Test audio file not found")
def test_transcribe_valid_audio():
    """Test with valid audio file"""
    with open("tests/test_audio.wav", "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", f, "audio/wav")}
        )

    if response.status_code == 200:
        data = response.json()
        assert "text" in data
        assert "duration" in data
        assert "sample_rate" in data
        assert isinstance(data["text"], str)
        assert data["duration"] > 0
        assert data["sample_rate"] == 16000