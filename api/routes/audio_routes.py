"""
Audio processing routes (STT/TTS)
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import io
import base64
import numpy as np

from api.models import (
    TextMessageRequest, 
    AudioTranscriptionResponse,
    AudioProcessResponse
)
from core.session_manager import SessionManager
from core.agent_initializer import AgentInitializer
from utils.audio_utils import numpy_to_wav_bytes
from config import logger

router = APIRouter()


@router.post("/audio/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    session_id: str = "default"
):
    """
    Transcribe audio to text using Groq Whisper
    
    - **audio**: Audio file (WAV format recommended)
    - **session_id**: Optional session identifier
    """
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        # Get voice service
        voice_service = AgentInitializer.get_voice_service()
        
        # Transcribe
        transcription = voice_service.stt(audio_bytes)
        
        if not transcription:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        return {
            "transcription": transcription,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/synthesize")
async def synthesize_speech(request: TextMessageRequest):
    """
    Convert text to speech using AWS Polly
    
    - **message**: Text to convert to speech
    - **session_id**: Optional session identifier
    
    Returns audio stream in WAV format
    """
    try:
        # Get voice service
        voice_service = AgentInitializer.get_voice_service()
        
        # Generate speech
        audio_bytes = voice_service.tts(request.message)
        
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Could not generate speech")
        
        # Convert PCM to WAV
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        wav_bytes = numpy_to_wav_bytes(16000, audio_array)
        
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/process", response_model=AudioProcessResponse)
async def process_audio_message(
    audio: UploadFile = File(...),
    session_id: str = "default"
):
    """
    Complete audio processing: STT -> Agent -> TTS
    
    - **audio**: Audio file with user's voice message
    - **session_id**: Optional session identifier
    
    Returns JSON with text response and base64 audio
    """
    try:
        # Get services
        voice_service = AgentInitializer.get_voice_service()
        master_router = AgentInitializer.get_master_router()
        session = SessionManager.get_or_create(session_id)
        
        # Step 1: Transcribe audio
        audio_bytes = await audio.read()
        user_text = voice_service.stt(audio_bytes)
        
        if not user_text:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # Step 2: Process through agents
        agent_name, response_text, updated_state = master_router.route(
            user_text,
            session
        )
        
        # Update state
        if agent_name == 'grocery_agent' and 'grocery_context' in updated_state:
            stage = updated_state['grocery_context'].get('confirmation_stage', 'initial')
            if stage in ['completed', 'cancelled']:
                updated_state['current_agent'] = None
            else:
                updated_state['current_agent'] = 'grocery_agent'
        else:
            updated_state['current_agent'] = agent_name
        
        updated_state['history'].append(f"You ({agent_name}): {user_text}")
        updated_state['history'].append(f"Assistant: {response_text}")
        
        # Save session
        SessionManager.update(session_id, updated_state)
        
        # Step 3: Generate speech
        response_audio_bytes = voice_service.tts(response_text)
        
        audio_data = None
        if response_audio_bytes:
            audio_array = np.frombuffer(response_audio_bytes, dtype=np.int16)
            wav_bytes = numpy_to_wav_bytes(16000, audio_array)
            audio_data = base64.b64encode(wav_bytes).decode('utf-8')
        
        # Get stage
        stage = None
        if agent_name == 'grocery_agent' and 'grocery_context' in updated_state:
            stage = updated_state['grocery_context'].get('confirmation_stage', 'initial')
        
        return {
            "transcription": user_text,
            "agent_name": agent_name,
            "response": response_text,
            "audio_base64": audio_data,
            "session_id": session_id,
            "conversation_history": updated_state['history'][-10:],
            "current_agent": updated_state.get('current_agent'),
            "stage": stage
        }
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))