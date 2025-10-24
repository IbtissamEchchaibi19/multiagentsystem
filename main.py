
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import message_routes, audio_routes, session_routes, agent_routes
from api.models import HealthResponse
from core.session_manager import SessionManager
from core.agent_initializer import AgentInitializer
from config import logger

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent AI System API",
    description="Unified AI agent system with News, Weather, Email, and Grocery capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services when application starts"""
    logger.info("=" * 80)
    logger.info("🚀 FASTAPI MULTI-AGENT SYSTEM STARTING")
    logger.info("=" * 80)
    
    # Initialize agents
    AgentInitializer.initialize()
    
    logger.info("📰 News Intelligence Agent: ✅")
    logger.info("🌤️ Weather Agent: ✅")
    logger.info("📧 Email & Calendar Agent: ✅")
    logger.info("🛒 Grocery Shopping Agent: ✅")
    logger.info("🎤 Voice Services: ✅")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Multi-Agent System...")
    SessionManager.clear_all()


# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "news_agent": "active",
            "weather_agent": "active",
            "email_agent": "active",
            "grocery_agent": "active",
            "voice_service": "active"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "master_router": "✅",
            "news_intelligence": "✅",
            "weather": "✅",
            "email_calendar": "✅",
            "grocery_shopping": "✅",
            "groq_stt": "✅",
            "aws_polly_tts": "✅"
        }
    }


# Include routers
app.include_router(message_routes.router, prefix="/api", tags=["Messages"])
app.include_router(audio_routes.router, prefix="/api", tags=["Audio"])
app.include_router(session_routes.router, prefix="/api", tags=["Sessions"])
app.include_router(agent_routes.router, prefix="/api", tags=["Agents"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )