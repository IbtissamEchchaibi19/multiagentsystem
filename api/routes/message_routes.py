"""
Message processing routes
"""

from fastapi import APIRouter, HTTPException
from api.models import TextMessageRequest, TextMessageResponse
from core.session_manager import SessionManager
from core.agent_initializer import AgentInitializer
from config import logger

router = APIRouter()


@router.post("/message", response_model=TextMessageResponse)
async def process_text_message(request: TextMessageRequest):
    """
    Process text message through the multi-agent system
    
    - **message**: User's text message
    - **session_id**: Optional session identifier for conversation context
    """
    try:
        # Get session
        session = SessionManager.get_or_create(request.session_id)
        
        # Get master router
        master_router = AgentInitializer.get_master_router()
        
        # Route and process
        agent_name, response_text, updated_state = master_router.route(
            request.message, 
            session
        )
        
        # Handle grocery agent state persistence
        if agent_name == 'grocery_agent' and 'grocery_context' in updated_state:
            stage = updated_state['grocery_context'].get('confirmation_stage', 'initial')
            if stage in ['completed', 'cancelled']:
                updated_state['current_agent'] = None
            else:
                updated_state['current_agent'] = 'grocery_agent'
        else:
            updated_state['current_agent'] = agent_name
        
        # Update session history
        updated_state['history'].append(f"You ({agent_name}): {request.message}")
        updated_state['history'].append(f"Assistant: {response_text}")
        
        # Save session
        SessionManager.update(request.session_id, updated_state)
        
        # Determine stage for grocery agent
        stage = None
        if agent_name == 'grocery_agent' and 'grocery_context' in updated_state:
            stage = updated_state['grocery_context'].get('confirmation_stage', 'initial')
        
        return {
            "agent_name": agent_name,
            "response": response_text,
            "session_id": request.session_id,
            "conversation_history": updated_state['history'][-10:],
            "current_agent": updated_state.get('current_agent'),
            "stage": stage
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))