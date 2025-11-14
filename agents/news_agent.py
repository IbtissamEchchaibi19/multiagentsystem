# News Intelligence Agent (7 sub-agents) - WITH CONTEXT MAINTENANCE

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
    is_follow_up: bool  # NEW: Track if this is a follow-up question


class NewsIntelligenceAgent:
    """Complete News Intelligence System with specialized sub-agents and CONTEXT MAINTENANCE"""
    
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
    
    def _check_for_follow_up(self, state: NewsAgentState) -> bool:
        """Check if user is asking about previously retrieved content"""
        last_message = state["messages"][-1]["content"].lower()
        context = state["conversation_context"]
        
        # Check for follow-up indicators
        follow_up_words = [
            "tell me more", "more about", "what about", "about the", 
            "first one", "second one", "third", "number", "article",
            "that place", "that product", "this video", "the paper",
            "link", "detail", "explain", "summarize"
        ]
        
        # Check if context has stored results
        has_context = any([
            context.get('current_articles'),
            context.get('current_papers'),
            context.get('current_places'),
            context.get('current_products'),
            context.get('current_media'),
            context.get('current_results')
        ])
        
        is_follow_up = has_context and any(word in last_message for word in follow_up_words)
        
        return is_follow_up
    
    def _router_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        
        # Check if this is a follow-up question about existing results
        if self._check_for_follow_up(state):
            state["current_agent"] = "deepdive_agent"
            state["is_follow_up"] = True
            return state
        
        state["is_follow_up"] = False
        
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
            
            # STORE IN CONTEXT for later retrieval
            state["conversation_context"]["current_articles"] = articles
            state["conversation_context"]["last_search_type"] = "news"
            state["conversation_context"]["last_query"] = last_message
            
            response_text = f"📰 **Found {len(articles)} news articles**\n\n"
            for i, article in enumerate(articles[:5], 1):
                response_text += f"**{i}. {article.get('title', 'No title')}**\n"
                response_text += f"   Source: {article.get('source', 'Unknown')}\n"
                response_text += f"   {article.get('snippet', '')}\n"
                if article.get('link'):
                    response_text += f"   🔗 {article.get('link')}\n"
                response_text += "\n"
            
            response_text += "💡 *Ask me about any specific article for more details!*"
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": "❌ Couldn't fetch news."})
        
        return state
    
    def _research_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        scholar_results = SerperAPI.search(last_message, "scholar", 8)
        
        if scholar_results["success"]:
            papers = scholar_results["data"].get("organic", [])
            
            # STORE IN CONTEXT
            state["conversation_context"]["current_papers"] = papers
            state["conversation_context"]["last_search_type"] = "research"
            state["conversation_context"]["last_query"] = last_message
            
            response_text = f"🎓 **Found {len(papers)} academic papers**\n\n"
            
            for i, paper in enumerate(papers[:5], 1):
                response_text += f"**{i}. {paper.get('title', 'No title')}**\n"
                response_text += f"   {paper.get('snippet', '')}\n"
                if paper.get('link'):
                    response_text += f"   🔗 {paper.get('link')}\n"
                response_text += "\n"
            
            response_text += "💡 *Ask me about any paper for more details!*"
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": "❌ No papers found."})
        
        return state
    
    def _local_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        places_results = SerperAPI.search(last_message, "places", 10)
        
        if places_results["success"]:
            places = places_results["data"].get("places", [])
            
            # STORE IN CONTEXT
            state["conversation_context"]["current_places"] = places
            state["conversation_context"]["last_search_type"] = "places"
            state["conversation_context"]["last_query"] = last_message
            
            response_text = f"📍 **Found {len(places)} places**\n\n"
            
            for i, place in enumerate(places[:5], 1):
                response_text += f"**{i}. {place.get('title', 'No name')}**\n"
                response_text += f"   ⭐ Rating: {place.get('rating', 'N/A')}\n"
                response_text += f"   📍 {place.get('address', 'No address')}\n"
                if place.get('phoneNumber'):
                    response_text += f"   📞 {place.get('phoneNumber')}\n"
                response_text += "\n"
            
            response_text += "💡 *Ask me about any place for more details!*"
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": "❌ No places found."})
        
        return state
    
    def _shopping_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        shopping_results = SerperAPI.search(last_message, "shopping", 10)
        
        if shopping_results["success"]:
            products = shopping_results["data"].get("shopping", [])
            
            # STORE IN CONTEXT
            state["conversation_context"]["current_products"] = products
            state["conversation_context"]["last_search_type"] = "shopping"
            state["conversation_context"]["last_query"] = last_message
            
            response_text = f"🛍️ **Found {len(products)} products**\n\n"
            
            for i, product in enumerate(products[:5], 1):
                response_text += f"**{i}. {product.get('title', 'No title')}**\n"
                response_text += f"   💰 Price: {product.get('price', 'N/A')}\n"
                response_text += f"   🪐 Source: {product.get('source', 'Unknown')}\n"
                if product.get('link'):
                    response_text += f"   🔗 {product.get('link')}\n"
                response_text += "\n"
            
            response_text += "💡 *Ask me about any product for more details!*"
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
            
            # STORE IN CONTEXT
            state["conversation_context"]["current_media"] = items
            state["conversation_context"]["last_search_type"] = search_type
            state["conversation_context"]["last_query"] = last_message
            
            response_text = f"🎬 **Found {len(items)} {search_type}**\n\n"
            
            for i, item in enumerate(items[:5], 1):
                response_text += f"**{i}. {item.get('title', 'No title')}**\n"
                if item.get('link'):
                    response_text += f"   🔗 {item.get('link')}\n"
                response_text += "\n"
            
            response_text += "💡 *Ask me about any item for more details!*"
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": f"❌ No {search_type} found."})
        
        return state
    
    def _web_agent(self, state: NewsAgentState) -> NewsAgentState:
        last_message = state["messages"][-1]["content"]
        web_results = SerperAPI.search(last_message, "search", 10)
        
        if web_results["success"]:
            results = web_results["data"].get("organic", [])
            
            # STORE IN CONTEXT
            state["conversation_context"]["current_results"] = results
            state["conversation_context"]["last_search_type"] = "web"
            state["conversation_context"]["last_query"] = last_message
            
            response_text = f"🌐 **Found {len(results)} web results**\n\n"
            
            for i, result in enumerate(results[:5], 1):
                response_text += f"**{i}. {result.get('title', 'No title')}**\n"
                response_text += f"   {result.get('snippet', '')}\n"
                if result.get('link'):
                    response_text += f"   🔗 {result.get('link')}\n"
                response_text += "\n"
            
            response_text += "💡 *Ask me about any result for more details!*"
            state["messages"].append({"role": "assistant", "content": response_text})
        else:
            state["messages"].append({"role": "assistant", "content": "❌ No results found."})
        
        return state
    
    def _deepdive_agent(self, state: NewsAgentState) -> NewsAgentState:
        """Handle follow-up questions about previously retrieved content"""
        last_message = state["messages"][-1]["content"].lower()
        context = state["conversation_context"]
        
        # Extract item number if mentioned (e.g., "first", "second", "1", "2")
        item_index = None
        number_words = {
            "first": 0, "1": 0, "one": 0,
            "second": 1, "2": 1, "two": 1,
            "third": 2, "3": 2, "three": 2,
            "fourth": 3, "4": 3, "four": 3,
            "fifth": 4, "5": 4, "five": 4
        }
        
        for word, idx in number_words.items():
            if word in last_message:
                item_index = idx
                break
        
        # Retrieve appropriate stored results
        search_type = context.get("last_search_type", "")
        stored_items = None
        
        if search_type == "news":
            stored_items = context.get("current_articles", [])
        elif search_type == "research":
            stored_items = context.get("current_papers", [])
        elif search_type == "places":
            stored_items = context.get("current_places", [])
        elif search_type == "shopping":
            stored_items = context.get("current_products", [])
        elif search_type in ["images", "videos"]:
            stored_items = context.get("current_media", [])
        elif search_type == "web":
            stored_items = context.get("current_results", [])
        
        if not stored_items:
            response = "I don't have any previous results to discuss. Please search for something first!"
            state["messages"].append({"role": "assistant", "content": response})
            return state
        
        # If specific item requested
        if item_index is not None and item_index < len(stored_items):
            item = stored_items[item_index]
            response_text = f"📋 **Details about item #{item_index + 1}:**\n\n"
            response_text += f"**Title:** {item.get('title', 'No title')}\n"
            
            if search_type == "news":
                response_text += f"**Source:** {item.get('source', 'Unknown')}\n"
                response_text += f"**Date:** {item.get('date', 'N/A')}\n"
                response_text += f"**Summary:** {item.get('snippet', 'No summary available')}\n"
            elif search_type == "places":
                response_text += f"**Rating:** {item.get('rating', 'N/A')} ⭐\n"
                response_text += f"**Address:** {item.get('address', 'N/A')}\n"
                response_text += f"**Phone:** {item.get('phoneNumber', 'N/A')}\n"
            elif search_type == "shopping":
                response_text += f"**Price:** {item.get('price', 'N/A')}\n"
                response_text += f"**Source:** {item.get('source', 'Unknown')}\n"
            else:
                response_text += f"**Description:** {item.get('snippet', 'No description')}\n"
            
            if item.get('link'):
                response_text += f"\n🔗 **Link:** {item.get('link')}\n"
            
            state["messages"].append({"role": "assistant", "content": response_text})
        
        # General overview of all items
        else:
            response_text = f"📚 **You have {len(stored_items)} {search_type} results from your search: '{context.get('last_query', '')}'**\n\n"
            response_text += "Here's what I found:\n\n"
            
            for i, item in enumerate(stored_items[:5], 1):
                response_text += f"{i}. {item.get('title', 'No title')}\n"
            
            response_text += f"\n💡 *Ask about a specific item (e.g., 'tell me about the first one') for more details!*"
            state["messages"].append({"role": "assistant", "content": response_text})
        
        return state
    
    def process(self, user_input: str, conversation_state: dict) -> tuple[str, dict]:
        """Process user input through news intelligence graph"""
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "current_agent": "router",
            "search_results": None,
            "conversation_context": conversation_state,
            "user_intent": "",
            "follow_up_needed": False,
            "is_follow_up": False
        }
        
        result = self.graph.invoke(initial_state)
        response = result["messages"][-1]["content"] if result["messages"] else "No response generated"
        
        return response, result.get("conversation_context", {})