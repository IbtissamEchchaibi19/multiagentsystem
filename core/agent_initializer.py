"""
Agent initialization and singleton management
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agents.voice_service import VoiceService
from agents.orchestrator import MasterRouterAgent
from config import logger

load_dotenv()


class AgentInitializer:
    """Singleton manager for agent initialization"""
    
    _voice_service = None
    _master_router = None
    _llm = None
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Initialize all services and agents"""
        if cls._initialized:
            logger.info("Agents already initialized")
            return
        
        logger.info("Initializing agents...")
        
        # Get environment variables
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
        AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
        GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        # Initialize voice service
        cls._voice_service = VoiceService(
            GROQ_API_KEY, 
            AWS_ACCESS_KEY, 
            AWS_SECRET_KEY
        )
        
        # Initialize LLM
        cls._llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.7
        )
        
        # Initialize master router (which initializes all agents)
        cls._master_router = MasterRouterAgent(cls._voice_service, cls._llm)
        
        cls._initialized = True
        logger.info("All agents initialized successfully")
    
    @classmethod
    def get_voice_service(cls) -> VoiceService:
        """Get voice service instance"""
        if not cls._initialized:
            cls.initialize()
        return cls._voice_service
    
    @classmethod
    def get_master_router(cls) -> MasterRouterAgent:
        """Get master router instance"""
        if not cls._initialized:
            cls.initialize()
        return cls._master_router
    
    @classmethod
    def get_llm(cls) -> ChatGroq:
        """Get LLM instance"""
        if not cls._initialized:
            cls.initialize()
        return cls._llm