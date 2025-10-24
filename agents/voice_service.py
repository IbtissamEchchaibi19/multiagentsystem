# STT/TTS service (Groq Whisper + AWS Polly)

import io
import wave
import numpy as np
import boto3
from groq import Groq

from config import logger

class VoiceService:
    def __init__(self, groq_api_key, aws_access_key, aws_secret_key, region="us-east-1"):
        self.groq_client = Groq(api_key=groq_api_key)
        self.polly_client = boto3.client(
            'polly',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        logger.info("✅ Voice Service initialized (Groq STT + AWS Polly TTS)")
    
    def stt(self, audio_bytes):
        """Speech to Text using Groq Whisper"""
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            transcription = self.groq_client.audio.transcriptions.create(
                file=("audio.wav", audio_file.read()),
                model="whisper-large-v3",
                temperature=0,
            )
            logger.info(f"🎤 Transcribed: {transcription.text}")
            return transcription.text
        except Exception as e:
            logger.error(f"STT Error: {str(e)}")
            return ""
    
    def tts(self, text):
        """Text to Speech using AWS Polly"""
        try:
            response = self.polly_client.synthesize_speech(
                Text=text[:3000],  # Polly limit
                OutputFormat='pcm',
                VoiceId='Joanna',
                Engine='standard',
                SampleRate='16000'
            )
            return response["AudioStream"].read() if "AudioStream" in response else None
        except Exception as e:
            logger.error(f"TTS Error: {str(e)}")
            return None