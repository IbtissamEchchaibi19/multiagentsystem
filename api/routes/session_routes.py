"""
Session management routes
"""

from fastapi import APIRouter, HTTPException
from api.models import SessionResponse, SessionClearResponse
from core.session_manager import SessionManager

router = APIRouter()


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get conversation session details
    
    - **session_id**: Session identifier
    """
    session = SessionManager.get(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "history": session.get('history', []),
        "current_agent": session.get('current_agent'),
        "news_context": session.get('news_context', {}),
        "grocery_context": session.get('grocery_context', {})
    }


@router.delete("/session/{session_id}", response_model=SessionClearResponse)
async def clear_session(session_id: str):
    """
    Clear conversation session
    
    - **session_id**: Session identifier
    """
    success = SessionManager.clear(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "message": "Session cleared successfully",
        "session_id": session_id
    }


@router.post("/session/{session_id}/clear", response_model=SessionClearResponse)
async def clear_session_post(session_id: str):
    """
    Clear conversation session (POST alternative)
    
    - **session_id**: Session identifier
    """
    success = SessionManager.clear(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "message": "Session cleared successfully",
        "session_id": session_id
    }