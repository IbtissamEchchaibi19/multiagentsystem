# News Intelligence Agent (7 sub-agents)

import json
from dotenv import load_dotenv
import os
load_dotenv()
import requests
from typing import TypedDict, Optional, Dict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
class SerperAPI:
    """Serper API Integration"""
    ENDPOINTS = {
        "search": "https://google.serper.dev/search",
        "news": "https://google.serper.dev/news",
        "images": "https://google.serper.dev/images",
        "videos": "https://google.serper.dev/videos",
        "places": "https://google.serper.dev/places",
        "shopping": "https://google.serper.dev/shopping",
        "scholar": "https://google.serper.dev/scholar",
    }
    
    @staticmethod
    def search(query: str, search_type: str = "news", num_results: int = 10) -> Dict:
        try:
            url = SerperAPI.ENDPOINTS.get(search_type, SerperAPI.ENDPOINTS["news"])
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            payload = {"q": query, "num": num_results}
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "search_type": search_type, "query": query}
            return {"success": False, "error": f"API returned {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

class NewsAgentState(TypedDict):
    messages: List[dict]
    current_agent: str
    search_results: Optional[Dict]
    conversation_context: Dict
    user_intent: str
    follow_up_needed: bool

class NewsIntelligenceAgent:
    """Complete News Intelligence System with specialized sub-agents"""
    
    def __init__(self, llm):
        self.llm = llm
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(NewsAgentState)
        
        workflow.add_node("router", self._router_agent)
        workflow.add_node("news_agent", self._news_agent)
        workflow.add_node("research_agent", self._research_agent)
        workflow.add_node("local_agent", self._local_agent)
        workflow.add_node("shopping_agent", self._shopping_agent)
        workflow.add_node("media_agent", self._media_agent)
        workflow.add_node("web_agent", self._web_agent)
        workflow.add_node("deepdive_agent", self._deepdive_agent)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            lambda state: state["current_agent"],
            {
                "news_agent": "news_agent",
                "research_agent": "research_agent",
                "local_agent": "local_agent",
                "shopping_agent": "shopping_agent",
                "media_agent": "media_agent",
                "web_agent": "web_agent",
                "deepdive_agent": "deepdive_agent"
            }
        )
        
        for agent in ["news_agent", "research_agent", "local_agent", "shopping_agent", "media_agent", "web_agent", "deepdive_agent"]:
            workflow.add_edge(agent, END)
        
        return workflow.compile()
    
    def _router_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        
        routing_prompt = f"""Analyze user query and determine which agent handles this.

User Query: {last_message}

Agents:
- news_agent: Breaking news, headlines, current events
- research_agent: Academic papers, patents, scholarly research
- local_agent: Places, restaurants, hotels, businesses
- shopping_agent: Products, prices, shopping
- media_agent: Images, videos
- web_agent: General web searches
- deepdive_agent: Follow-up questions about previous results

Return JSON: {{"agent": "agent_name", "reasoning": "why"}}"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=routing_prompt)])
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            routing = json.loads(content)
            state["current_agent"] = routing["agent"]
        except:
            # Fallback keyword routing
            last_lower = last_message.lower()
            if any(w in last_lower for w in ["news", "today", "latest", "breaking"]):
                state["current_agent"] = "news_agent"
            elif any(w in last_lower for w in ["research", "paper", "study", "scholar"]):
                state["current_agent"] = "research_agent"
            elif any(w in last_lower for w in ["place", "restaurant", "hotel", "near"]):
                state["current_agent"] = "local_agent"
            elif any(w in last_lower for w in ["buy", "price", "shop", "product"]):
                state["current_agent"] = "shopping_agent"
            elif any(w in last_lower for w in ["image", "video", "photo"]):
                state["current_agent"] = "media_agent"
            else:
                state["current_agent"] = "web_agent"
        
        return state
    
    def _news_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        news_results = SerperAPI.search(last_message, "news", 10)
        
        if news_results["success"]:
            articles = news_results["data"].get("news", [])
            state["search_results"] = news_results
            state["conversation_context"]["current_articles"] = articles
            
            response_text = f"📰 **Found {len(articles)} news articles**\n\n"
            for i, article in enumerate(articles[:5], 1):
                response_text += f"**{i}. {article.get('title', 'No title')}**\n"
                response_text += f"   Source: {article.get('source', 'Unknown')}\n"
                response_text += f"   {article.get('snippet', '')}\n\n"
            
            response_text += "💡 *Ask me about any specific article or search for more!*"
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": "❌ Couldn't fetch news."})
        
        return state
    
    def _research_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        scholar_results = SerperAPI.search(last_message, "scholar", 8)
        
        if scholar_results["success"]:
            papers = scholar_results["data"].get("organic", [])
            response_text = f"🎓 **Found {len(papers)} academic papers**\n\n"
            
            for i, paper in enumerate(papers[:5], 1):
                response_text += f"**{i}. {paper.get('title', 'No title')}**\n"
                response_text += f"   {paper.get('snippet', '')}\n\n"
            
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": "❌ No papers found."})
        
        return state
    
    def _local_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        places_results = SerperAPI.search(last_message, "places", 10)
        
        if places_results["success"]:
            places = places_results["data"].get("places", [])
            response_text = f"📍 **Found {len(places)} places**\n\n"
            
            for i, place in enumerate(places[:5], 1):
                response_text += f"**{i}. {place.get('title', 'No name')}**\n"
                response_text += f"   ⭐ Rating: {place.get('rating', 'N/A')}\n"
                response_text += f"   📍 {place.get('address', 'No address')}\n\n"
            
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": "❌ No places found."})
        
        return state
    
    def _shopping_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        shopping_results = SerperAPI.search(last_message, "shopping", 10)
        
        if shopping_results["success"]:
            products = shopping_results["data"].get("shopping", [])
            response_text = f"🛍️ **Found {len(products)} products**\n\n"
            
            for i, product in enumerate(products[:5], 1):
                response_text += f"**{i}. {product.get('title', 'No title')}**\n"
                response_text += f"   💰 Price: {product.get('price', 'N/A')}\n"
                response_text += f"   🪐 Source: {product.get('source', 'Unknown')}\n\n"
            
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": "❌ No products found."})
        
        return state
    
    def _media_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        search_type = "videos" if "video" in last_message.lower() else "images"
        results = SerperAPI.search(last_message, search_type, 10)
        
        if results["success"]:
            items = results["data"].get(search_type, [])
            response_text = f"🎬 **Found {len(items)} {search_type}**\n\n"
            
            for i, item in enumerate(items[:5], 1):
                response_text += f"**{i}. {item.get('title', 'No title')}**\n\n"
            
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": f"❌ No {search_type} found."})
        
        return state
    
    def _web_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        web_results = SerperAPI.search(last_message, "search", 10)
        
        if web_results["success"]:
            results = web_results["data"].get("organic", [])
            response_text = f"🌐 **Found {len(results)} web results**\n\n"
            
            for i, result in enumerate(results[:5], 1):
                response_text += f"**{i}. {result.get('title', 'No title')}**\n"
                response_text += f"   {result.get('snippet', '')}\n\n"
            
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": "❌ No results found."})
        
        return state
    
    def _deepdive_agent(self, state: NewsAgentState) -> NewsAgentState:
        state["messages"].append({
            "role": "assistant",
            "content": "I can provide more details about previous search results. Please specify which item you'd like to know more about!"
        })
        return state
    
    def process(self, user_input: str, conversation_state: dict) -> tuple[str, dict]:
        """Process user input through news intelligence graph"""
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "current_agent": "router",
            "search_results": None,
            "conversation_context": conversation_state,
            "user_intent": "",
            "follow_up_needed": False
        }
        
        result = self.graph.invoke(initial_state)
        response = result["messages"][-1]["content"] if result["messages"] else "No response generated"
        
        return response, result.get("conversation_context", {})