# Master Router (orchestrates all agents)
import json
from langchain_core.messages import HumanMessage
from .news_agent import NewsIntelligenceAgent
from .weather_agent import WeatherAgent
from .email_agent import EmailCalendarAgent
from .grocery_agent import GroceryShoppingAgent, HuaweiLLM
from config import  logger
from dotenv import load_dotenv
import os
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
class MasterRouterAgent:
    """Master router that delegates to specialized agents"""
    
    def __init__(self, voice_service, llm):
        self.voice = voice_service
        self.llm = llm
        
        # Initialize all specialized agents
        self.news_agent = NewsIntelligenceAgent(llm)
        self.weather_agent = WeatherAgent(llm)
        self.email_agent = EmailCalendarAgent(llm)
        self.grocery_agent = GroceryShoppingAgent(HuaweiLLM(DEEPSEEK_API_KEY))
        
        logger.info("✅ Master Router initialized with all agents")
    
    def route(self, user_input: str, conversation_state: dict) -> tuple[str, str, dict]:
        """Route to appropriate agent"""
        
        agent_name = None
        
        if conversation_state.get('current_agent') == 'grocery_agent':
            grocery_context = conversation_state.get('grocery_context', {})
            lower_input = user_input.lower().strip()
            
            if grocery_context.get('awaiting_confirmation', False) or grocery_context.get('confirmation_stage', 'initial') != 'initial':
                agent_name = 'grocery_agent'
            elif lower_input in ['yes', 'no', 'confirm', 'cancel', 'yeah', 'sure', 'ok', 'okay', 'proceed', 'go ahead', 'not now', 'stop', 'nevermind', 'nope']:
                agent_name = 'grocery_agent'
        
        if agent_name is None:
            routing_prompt = f"""Analyze user request and determine which specialized agent should handle it.

User: "{user_input}"

Context: {json.dumps(conversation_state, default=str)[:200]}

Agents:
1. news_agent: News, research papers, places, shopping products, images, videos, web search
2. weather_agent: Weather information, forecasts, comparisons, outfit recommendations
3. email_agent: Email management, Gmail, meeting scheduling, calendar
4. grocery_agent: Grocery shopping, food items, meal ingredients

If there is an ongoing conversation with an agent in context, prefer to stay with that agent unless the query clearly indicates switching.

Return ONLY the agent name: news_agent, weather_agent, email_agent, or grocery_agent"""

            try:
                agent_name = self.llm.invoke([HumanMessage(content=routing_prompt)]).content.strip().lower()
                
                # Clean response
                for valid in ["news_agent", "weather_agent", "email_agent", "grocery_agent"]:
                    if valid in agent_name:
                        agent_name = valid
                        break
                
                logger.info(f"🎯 Routing to: {agent_name}")
                
            except Exception as e:
                logger.error(f"Routing error: {str(e)}")
                return "error", f"Error: {str(e)}", conversation_state
        
        # Route to agent
        if agent_name == "news_agent":
            response, context = self.news_agent.process(user_input, conversation_state.get('news_context', {}))
            conversation_state['news_context'] = context
            
        elif agent_name == "weather_agent":
            response = self.weather_agent.process(user_input)
            
        elif agent_name == "email_agent":
            response = self.email_agent.process(user_input)
            
        elif agent_name == "grocery_agent":
            response, context = self.grocery_agent.process(user_input, conversation_state.get('grocery_context', {}))
            conversation_state['grocery_context'] = context
            
        else:
            agent_name = "general"
            response = ("🤖 **I can help you with:**\n\n"
                      "📰 News, research, places, shopping\n"
                      "🌤️ Weather forecasts and information\n"
                      "📧 Email and calendar management\n"
                      "🛒 Grocery shopping\n\n"
                      "What would you like to do?")
        
        return agent_name, response, conversation_state