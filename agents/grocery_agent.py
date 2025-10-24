# Grocery Shopping Agent (multi-stage)

import json
import requests
import random
import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from config import logger

class HuaweiLLM:
    def __init__(self, token, model="deepseek-r1-distil-qwen-32b_raziqt"):
        self.token = token
        self.model = model
        self.url = "https://pangu.ap-southeast-1.myhuaweicloud.com/api/v2/chat/completions"
    
    def invoke(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Exception: {str(e)}"

class GroceryAPIs:
    def __init__(self):
        self.openfoodfacts_base = "https://world.openfoodfacts.org/cgi/search.pl"
        self.openprices_base = "https://prices.openfoodfacts.org/api/v1"
        self.headers = {"User-Agent": "GroceryVoiceAgent/1.0"}
    
    def search_openfoodfacts(self, query):
        """Search OpenFoodFacts database for products"""
        logger.info(f"🔍 Searching OpenFoodFacts for: {query}")
        
        try:
            params = {
                "search_terms": query,
                "search_simple": 1,
                "json": 1,
                "page_size": 10,
                "fields": "product_name,brands,stores,code,countries_tags,image_url"
            }
            
            response = requests.get(
                self.openfoodfacts_base,
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"OpenFoodFacts API error: {response.status_code}")
                return []
            
            data = response.json()
            products = data.get("products", [])
            logger.info(f"✅ Found {len(products)} products from OpenFoodFacts")
            
            results = []
            for product in products[:5]:
                product_name = product.get("product_name", "Unknown Product")
                brands = product.get("brands", "Generic")
                stores = product.get("stores", "Various Stores")
                barcode = product.get("code", "")
                
                if not product_name or product_name == "Unknown Product":
                    continue
                
                price = self.get_price_estimate(barcode) if barcode else None
                if not price:
                    price = self._estimate_price(product_name)
                
                result = {
                    "name": f"{product_name} - {brands}",
                    "price": price,
                    "store": stores if stores else "Multiple Stores",
                    "barcode": barcode
                }
                results.append(result)
                logger.info(f"  📦 {result['name']} - ${result['price']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching OpenFoodFacts: {str(e)}")
            return []
    
    def get_price_estimate(self, barcode):
        """Try to get price from OpenPrices API"""
        try:
            url = f"{self.openprices_base}/product/{barcode}.json"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("items") and len(data["items"]) > 0:
                    latest_price = data["items"][0].get("price")
                    if latest_price:
                        return f"{latest_price:.2f}"
        except Exception as e:
            logger.debug(f"Could not get price from OpenPrices: {str(e)}")
        return None
    
    def _estimate_price(self, product_name):
        """Generate realistic price estimate based on product category"""
        product_lower = product_name.lower()
        
        if any(word in product_lower for word in ["organic", "premium", "artisan"]):
            base_price = 6.99
        elif any(word in product_lower for word in ["milk", "yogurt", "cheese", "dairy"]):
            base_price = 4.49
        elif any(word in product_lower for word in ["bread", "baguette", "roll"]):
            base_price = 3.29
        elif any(word in product_lower for word in ["pasta", "rice", "noodle"]):
            base_price = 2.99
        elif any(word in product_lower for word in ["meat", "chicken", "beef", "pork"]):
            base_price = 7.99
        elif any(word in product_lower for word in ["fish", "salmon", "tuna"]):
            base_price = 9.99
        elif any(word in product_lower for word in ["fruit", "apple", "banana", "orange"]):
            base_price = 3.49
        elif any(word in product_lower for word in ["vegetable", "tomato", "lettuce", "carrot"]):
            base_price = 2.99
        elif any(word in product_lower for word in ["snack", "chips", "cookie"]):
            base_price = 3.99
        elif any(word in product_lower for word in ["juice", "soda", "drink", "beverage"]):
            base_price = 4.99
        elif any(word in product_lower for word in ["cereal", "oatmeal", "granola"]):
            base_price = 4.49
        else:
            base_price = 3.99
        
        variation = random.uniform(-0.50, 0.50)
        final_price = max(0.99, base_price + variation)
        return f"{final_price:.2f}"
    
    def search_all(self, query):
        """Search using OpenFoodFacts API"""
        logger.info(f"🛒 SEARCHING FOR: '{query}'")
        results = self.search_openfoodfacts(query)
        
        if not results:
            logger.warning("⚠️ No results, using generic fallback")
            results = self._generic_fallback(query)
        
        return results
    
    def _generic_fallback(self, query):
        """Fallback search with generic results"""
        return [
            {"name": f"Fresh {query.title()}", "price": "3.99", "store": "Local Grocery", "barcode": ""},
            {"name": f"Organic {query.title()}", "price": "5.49", "store": "Organic Market", "barcode": ""},
            {"name": f"{query.title()} (Store Brand)", "price": "2.99", "store": "Supermarket", "barcode": ""}
        ]

class AgentState(TypedDict):
    user_text: str
    agent_response: str
    cart: List[dict]
    search_results: List[dict]
    awaiting_confirmation: bool
    conversation_history: List[str]
    items_to_search: List[str]
    user_action: str
    order_confirmed: bool
    confirmation_stage: str
    search_results_by_item: dict

class GroceryShoppingAgent:
    """Grocery Shopping with Multi-Stage Confirmation"""
    
    def __init__(self, llm):
        self.llm = llm
        self.apis = GroceryAPIs()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("understand", self._understand_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("respond", self._respond_node)
        
        workflow.set_entry_point("understand")
        workflow.add_edge("understand", "search")
        workflow.add_edge("search", "reason")
        workflow.add_edge("reason", "respond")
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    def _extract_items_simple(self, text):
        """Simple rule-based item extraction without LLM"""
        ignore_words = {
            'i', 'want', 'to', 'buy', 'need', 'get', 'some', 'a', 'an', 'the',
            'please', 'can', 'could', 'would', 'like', 'love', 'shopping', 'for',
            'me', 'my', 'and', 'or', 'also', 'plus'
        }
        
        text_lower = text.lower()
        text_lower = re.sub(r'\bi want to buy\b', '', text_lower)
        text_lower = re.sub(r'\bi need\b', '', text_lower)
        text_lower = re.sub(r'\bplease get me\b', '', text_lower)
        text_lower = re.sub(r'\bcan i have\b', '', text_lower)
        
        separators = [',', ' and ', '&', 'plus']
        items = [text_lower.strip()]
        
        for sep in separators:
            new_items = []
            for item in items:
                new_items.extend([x.strip() for x in item.split(sep)])
            items = new_items
        
        cleaned_items = []
        for item in items:
            words = item.split()
            words = [w for w in words if w not in ignore_words or len(words) == 1]
            
            if words:
                cleaned_item = ' '.join(words)
                if len(cleaned_item) > 2:
                    cleaned_items.append(cleaned_item)
        
        return cleaned_items if cleaned_items else [text.strip()]
    
    def _understand_node(self, state: AgentState) -> AgentState:
        """Extract intent and items"""
        logger.info("=== UNDERSTAND NODE ===")
        
        user_text_lower = state['user_text'].lower().strip()
        yes_words = ['yes', 'yeah', 'sure', 'ok', 'okay', 'proceed', 'go ahead']
        no_words = ['no', 'cancel', 'not now', 'stop', 'nevermind', 'nope']
        
        # Handle confirmation stages
        if state.get('confirmation_stage') == 'awaiting_yes':
            if any(word in user_text_lower for word in yes_words):
                state['user_action'] = 'confirm_yes'
                state['items_to_search'] = []
                return state
            elif any(word in user_text_lower for word in no_words):
                state['user_action'] = 'cancel'
                state['items_to_search'] = []
                return state
        
        if state.get('confirmation_stage') == 'awaiting_final':
            if any(word in user_text_lower for word in yes_words + ['confirm']):
                state['user_action'] = 'final_confirm'
                state['items_to_search'] = []
                return state
            elif any(word in user_text_lower for word in no_words):
                state['user_action'] = 'cancel'
                state['items_to_search'] = []
                return state
        
        # For initial confirmation: Now flexible to accept 'yes' as 'confirm'
        if state.get('awaiting_confirmation') and any(word in user_text_lower for word in ['confirm'] + yes_words):
            state['user_action'] = 'confirm'
            state['items_to_search'] = []
            return state
        
        if any(word in user_text_lower for word in no_words + ['clear', 'reset']):
            state['user_action'] = 'cancel'
            state['items_to_search'] = []
            return state
        
        # Try LLM first, fallback to simple extraction
        extracted_items = []
        
        try:
            prompt = f"""Extract only the grocery item names from this sentence: "{state['user_text']}"

Return ONLY a JSON object with this format:
{{"items": ["item1", "item2", "item3"]}}

Examples:
- "I want tomatoes and eggs" -> {{"items": ["tomatoes", "eggs"]}}
- "Get me pasta, rice, and milk" -> {{"items": ["pasta", "rice", "milk"]}}

Now extract from: "{state['user_text']}" """
            
            response = self.llm.invoke(prompt)
            
            if response and not response.startswith('Error'):
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = response[start:end]
                    parsed = json.loads(json_str)
                    extracted_items = parsed.get('items', [])
                    logger.info(f"✅ LLM extracted: {extracted_items}")
        
        except Exception as e:
            logger.warning(f"LLM extraction failed: {str(e)}")
        
        if not extracted_items:
            logger.warning("⚠️ Using simple extraction")
            extracted_items = self._extract_items_simple(state['user_text'])
        
        state['items_to_search'] = extracted_items
        state['user_action'] = 'search'
        
        return state
    
    def _search_node(self, state: AgentState) -> AgentState:
        """Search across all markets"""
        logger.info("=== SEARCH NODE ===")
        
        if state.get('user_action') in ['cancel', 'confirm', 'confirm_yes', 'final_confirm']:
            return state
        
        all_results = []
        results_by_item = {}
        
        for item in state.get('items_to_search', []):
            try:
                results = self.apis.search_all(item)
                results_by_item[item] = results
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Search error for {item}: {str(e)}")
        
        state['search_results'] = all_results
        state['search_results_by_item'] = results_by_item
        
        return state
    
    def _reason_node(self, state: AgentState) -> AgentState:
        """Select best items"""
        logger.info("=== REASON NODE ===")
        
        if state.get('user_action') == 'final_confirm':
            state['order_confirmed'] = True
            state['confirmation_stage'] = 'completed'
            state['awaiting_confirmation'] = False
            state['cart'] = []  # Clear cart after order placed
            return state
        
        if state.get('user_action') == 'cancel':
            state['cart'] = []
            state['confirmation_stage'] = 'cancelled'
            state['awaiting_confirmation'] = False
            return state
        
        if state.get('user_action') == 'confirm' and state.get('awaiting_confirmation'):
            state['confirmation_stage'] = 'awaiting_yes'
            return state
        
        if state.get('user_action') == 'confirm_yes':
            state['confirmation_stage'] = 'awaiting_final'
            return state
        
        if not state.get('search_results_by_item'):
            state['cart'] = []
            return state
        
        selected_items = []
        
        for item_name, results in state.get('search_results_by_item', {}).items():
            if not results:
                continue
            
            valid_results = [r for r in results if r.get('price') and r['price'] != 'N/A']
            
            if valid_results:
                best_item = min(valid_results, key=lambda x: float(x['price'].replace('$', '')))
                selected_items.append(best_item)
            else:
                selected_items.append(results[0])
        
        state['cart'] = selected_items
        state['awaiting_confirmation'] = True
        state['confirmation_stage'] = 'initial'
        
        return state
    
    def _respond_node(self, state: AgentState) -> AgentState:
        """Generate voice response"""
        logger.info("=== RESPOND NODE ===")
        
        if state.get('order_confirmed') and state.get('confirmation_stage') == 'completed':
            response_text = ("Perfect! Your order has been placed successfully! "
                           "Please pay attention to your phone - you will receive a delivery call soon. "
                           "Thank you for shopping with us!")
        
        elif state.get('confirmation_stage') == 'cancelled':
            response_text = "Okay, no problem! Your cart has been cleared. I'm here whenever you need me!"
        
        elif state.get('confirmation_stage') == 'awaiting_final':
            if not state.get('cart'):
                response_text = "Sorry, your cart is empty. Please start a new order."
            else:
                response_text = ("Are you sure you want to proceed? "
                               "I will use your saved card details to complete the payment. "
                               "Say yes to confirm or no to cancel.")
        
        elif state.get('confirmation_stage') == 'awaiting_yes':
            if not state.get('cart'):
                response_text = "Sorry, your cart is empty. Please start a new order."
            else:
                cart_items = state['cart']
                total_price = sum(float(item['price'].replace('$', '')) for item in cart_items)
                items_list = ", ".join([f"{item['name']} for ${item['price']}" for item in cart_items])
                response_text = (f"Let me confirm your order. You selected: {items_list}. "
                               f"The total is ${total_price:.2f}. "
                               f"Would you like to proceed? Say yes to continue or no to cancel.")
        
        elif not state.get('cart'):
            response_text = "Sorry, I couldn't find those items. Please try again with different items."
        
        else:
            cart_items = state['cart']
            total_price = sum(float(item['price'].replace('$', '')) for item in cart_items)
            
            if len(cart_items) == 1:
                items_desc = f"{cart_items[0]['name']} for ${cart_items[0]['price']}"
            else:
                items_desc = ", ".join([f"{item['name']} for ${item['price']}" for item in cart_items[:-1]])
                items_desc += f", and {cart_items[-1]['name']} for ${cart_items[-1]['price']}"
            
            response_text = (f"I found {len(cart_items)} items for you: {items_desc}. "
                           f"Total price: ${total_price:.2f}. "
                           f"Say confirm to review your order, or cancel to clear your cart.")
        
        state['agent_response'] = response_text
        state['conversation_history'].append(f"Agent: {response_text}")
        
        return state
    
    def process(self, user_input: str, conversation_state: dict) -> tuple[str, dict]:
        """Process grocery shopping request"""
        initial_state = {
            "user_text": user_input,
            "agent_response": "",
            "cart": conversation_state.get('cart', []),
            "search_results": [],
            "awaiting_confirmation": conversation_state.get('awaiting_confirmation', False),
            "conversation_history": conversation_state.get('history', []),
            "items_to_search": [],
            "user_action": "search",
            "order_confirmed": False,
            "confirmation_stage": conversation_state.get('confirmation_stage', 'initial'),
            "search_results_by_item": {}
        }
        
        result = self.graph.invoke(initial_state)
        
        return result['agent_response'], {
            'cart': result.get('cart', []),
            'confirmation_stage': result.get('confirmation_stage', 'initial'),
            'awaiting_confirmation': result.get('awaiting_confirmation', False),
            'history': result.get('conversation_history', [])
        }