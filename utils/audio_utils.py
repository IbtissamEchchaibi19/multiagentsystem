"""
Audio utility functions
"""

import io
import wave
import numpy as np


def numpy_to_wav_bytes(sample_rate: int, audio_data: np.ndarray) -> bytes:
    if audio_data.dtype != np.int16:
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()


def wav_bytes_to_numpy(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    """
    Convert WAV bytes to numpy array
    
    Args:
        wav_bytes: WAV file as bytes
        
    Returns:
        Tuple of (sample_rate, audio_data)
    """
    wav_buffer = io.BytesIO(wav_bytes)
    with wave.open(wav_buffer, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(n_frames)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    
    return sample_rate, audio_data