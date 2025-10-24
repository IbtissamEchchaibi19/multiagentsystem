"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class TextMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class TextMessageResponse(BaseModel):
    agent_name: str
    response: str
    session_id: str
    conversation_history: List[str]
    current_agent: Optional[str] = None
    stage: Optional[str] = None


class AudioTranscriptionResponse(BaseModel):
    transcription: str
    session_id: str


class AudioProcessResponse(BaseModel):
    transcription: str
    agent_name: str
    response: str
    audio_base64: Optional[str] = None
    session_id: str
    conversation_history: List[str]
    current_agent: Optional[str] = None
    stage: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    history: List[str]
    current_agent: Optional[str]
    news_context: Dict[str, Any]
    grocery_context: Dict[str, Any]


class SessionClearResponse(BaseModel):
    message: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]


class AgentInfo(BaseModel):
    name: str
    capabilities: List[str]
    sub_agents: Optional[int] = None
    tools: Optional[int] = None
    requires_auth: Optional[bool] = False
    workflow_stages: Optional[List[str]] = None


class AgentsListResponse(BaseModel):
    agents: Dict[str, AgentInfo]