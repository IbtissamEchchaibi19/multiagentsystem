"""
Session management for conversation state
In production, replace with Redis or database
"""

from typing import Dict, Any, Optional
from config import logger


class SessionManager:
    """Manages conversation sessions (in-memory, use Redis in production)"""
    
    _sessions: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def get_or_create(cls, session_id: str) -> Dict[str, Any]:
        """Get existing session or create new one"""
        if session_id not in cls._sessions:
            cls._sessions[session_id] = {
                'history': [],
                'news_context': {},
                'grocery_context': {},
                'current_agent': None
            }
            logger.info(f"Created new session: {session_id}")
        
        return cls._sessions[session_id]
    
    @classmethod
    def get(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session if exists"""
        return cls._sessions.get(session_id)
    
    @classmethod
    def update(cls, session_id: str, data: Dict[str, Any]) -> None:
        """Update session data"""
        cls._sessions[session_id] = data
        logger.debug(f"Updated session: {session_id}")
    
    @classmethod
    def clear(cls, session_id: str) -> bool:
        """Clear specific session"""
        if session_id in cls._sessions:
            del cls._sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    @classmethod
    def clear_all(cls) -> None:
        """Clear all sessions"""
        count = len(cls._sessions)
        cls._sessions.clear()
        logger.info(f"Cleared all {count} sessions")
    
    @classmethod
    def get_session_count(cls) -> int:
        """Get total number of active sessions"""
        return len(cls._sessions)