# Weather Agent (LangGraph + 3 tools + Conversation Memory)

import json
import requests
from typing import TypedDict, List, Annotated
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
import operator
from dotenv import load_dotenv
import os

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

@tool
def get_current_weather(city: str) -> str:
    """Get current weather for a city"""
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return json.dumps({
                "city": data["name"],
                "temperature": round(data["main"]["temp"], 1),
                "feels_like": round(data["main"]["feels_like"], 1),
                "humidity": data["main"]["humidity"],
                "weather": data["weather"][0]["main"],
                "description": data["weather"][0]["description"],
                "wind_speed": round(data["wind"]["speed"], 1),
            })
        return json.dumps({"error": f"City '{city}' not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def get_weather_forecast(city: str, days: int = 5) -> str:
    """Get weather forecast for next 5 days"""
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            forecasts = []
            
            for item in data["list"][::8][:days]:
                forecasts.append({
                    "date": datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d"),
                    "temperature": round(item["main"]["temp"], 1),
                    "weather": item["weather"][0]["description"],
                    "rain_probability": round(item.get("pop", 0) * 100, 0)
                })
            
            return json.dumps({"city": data["city"]["name"], "forecasts": forecasts})
        return json.dumps({"error": f"City '{city}' not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def compare_weather(city1: str, city2: str) -> str:
    """Compare weather between two cities"""
    try:
        w1 = json.loads(get_current_weather.invoke({"city": city1}))
        w2 = json.loads(get_current_weather.invoke({"city": city2}))
        
        if "error" in w1 or "error" in w2:
            return json.dumps({"error": "One or both cities not found"})
        
        return json.dumps({
            "city1": {"name": w1["city"], "temp": w1["temperature"], "weather": w1["weather"]},
            "city2": {"name": w2["city"], "temp": w2["temperature"], "weather": w2["weather"]},
            "temp_diff": abs(w1["temperature"] - w2["temperature"])
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

class WeatherAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    iteration: int

class WeatherAgent:
    """Weather Agent with LangGraph and Conversation Memory"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [get_current_weather, get_weather_forecast, compare_weather]
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.graph = self._build_graph()
        
        # Conversation memory - stores all messages across interactions
        self.conversation_history: List[AnyMessage] = [
            SystemMessage(content="You are a helpful weather assistant with access to weather tools. "
                                "You can check current weather, forecasts, and compare cities. "
                                "Remember previous questions in the conversation and provide contextual responses.")
        ]
    
    def _build_graph(self):
        workflow = StateGraph(WeatherAgentState)
        
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END}
        )
        
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _agent_node(self, state: WeatherAgentState) -> WeatherAgentState:
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response], "iteration": state.get("iteration", 0) + 1}
    
    def _should_continue(self, state: WeatherAgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        if state.get("iteration", 0) > 10:
            return "end"
        return "end"
    
    def process(self, user_input: str) -> str:
        """Process weather query with conversation memory"""
        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_input))
        
        # Run graph with full conversation history
        state = {
            "messages": self.conversation_history.copy(),
            "iteration": 0
        }
        
        result = self.graph.invoke(state)
        
        # Extract all new messages from this interaction (excluding the history we passed in)
        new_messages = result["messages"][len(self.conversation_history):]
        
        # Add new messages to conversation history
        self.conversation_history.extend(new_messages)
        
        # Get final response
        final_message = result["messages"][-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)
    
    def clear_history(self):
        """Clear conversation history (keep only system message)"""
        self.conversation_history = [self.conversation_history[0]]
    
    def get_history(self) -> List[AnyMessage]:
        """Get current conversation history"""
        return self.conversation_history.copy()

