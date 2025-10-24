"""
Agent information routes
"""

from fastapi import APIRouter
from api.models import AgentsListResponse, AgentInfo

router = APIRouter()


@router.get("/agents", response_model=AgentsListResponse)
async def list_agents():
    """
    List all available agents and their capabilities
    """
    return {
        "agents": {
            "news_agent": AgentInfo(
                name="News Intelligence Agent",
                capabilities=[
                    "Breaking news and headlines",
                    "Academic research papers",
                    "Local places and businesses",
                    "Product shopping search",
                    "Images and videos",
                    "General web search"
                ],
                sub_agents=7
            ),
            "weather_agent": AgentInfo(
                name="Weather Expert",
                capabilities=[
                    "Current weather conditions",
                    "5-day forecasts",
                    "Weather comparisons",
                    "Multi-city analysis"
                ],
                tools=3
            ),
            "email_agent": AgentInfo(
                name="Email & Calendar Assistant",
                capabilities=[
                    "Gmail inbox management",
                    "Email drafting and sending",
                    "Meeting scheduling",
                    "Calendar integration"
                ],
                requires_auth=True
            ),
            "grocery_agent": AgentInfo(
                name="Grocery Shopping Agent",
                capabilities=[
                    "Product search via OpenFoodFacts",
                    "Price estimation",
                    "Multi-stage order confirmation",
                    "Shopping cart management"
                ],
                workflow_stages=["initial", "awaiting_yes", "awaiting_final", "completed"]
            )
        }
    }