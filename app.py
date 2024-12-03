import streamlit as st
from firebase_admin import credentials, auth, firestore
from openai import OpenAI
from datetime import datetime, timedelta
import pytz
import requests
from functools import lru_cache
from typing import List, Dict, Any
import json

# Import configurations
from initial import INITIAL_ASSISTANT_MESSAGE
from reviewprocess import SYSTEM_INSTRUCTIONS, REVIEW_INSTRUCTIONS, DISCLAIMER

class EWACache:
    """Cache manager for Essay Writing Assistant"""
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        
    def get(self, key: str) -> Any:
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return data
            del self._cache[key]
        return None
        
    def set(self, key: str, value: Any):
        self._cache[key] = (value, datetime.now())
        
    def clear(self):
        self._cache.clear()

class EWA:
    def __init__(self):
        self.tz = pytz.timezone("Europe/London")
        self.conversations_per_page = 10
        self.cache = EWACache()
        self.db = firestore.client()
        
        # Initialize OpenAI client once
        self.openai_client = OpenAI(api_key=st.secrets["default"]["OPENAI_API_KEY"])
        
    @lru_cache(maxsize=100)
    def format_time(self, timestamp_str: str) -> str:
        """Cache formatted timestamps to avoid repeated processing"""
        if isinstance(timestamp_str, (datetime, type(firestore.SERVER_TIMESTAMP))):
            return timestamp_str.strftime("[%Y-%m-%d %H:%M:%S]")
        dt = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now(self.tz)
        return dt.strftime("[%Y-%m-%d %H:%M:%S]")

    @lru_cache(maxsize=50)
    def generate_title(self, message_content: str, timestamp_str: str) -> str:
        """Cache generated titles for repeated messages"""
        current_time = datetime.fromisoformat(timestamp_str)
        title = current_time.strftime('%b %d, %Y • ') + ' '.join(message_content.split()[:4])
        return title[:50] if len(title) > 50 else title

    def get_conversations(self, user_id: str) -> tuple:
        """Retrieve conversation history with caching"""
        cache_key = f"conversations_{user_id}_{st.session_state.get('page', 0)}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
            
        page = st.session_state.get('page', 0)
        start = page * self.conversations_per_page
        
        # Use in-memory counter for performance
        if not hasattr(self, '_conv_count'):
            self._conv_count = {}
        
        if user_id not in self._conv_count:
            self._conv_count[user_id] = len(list(self.db.collection('conversations')
                                         .where('user_id', '==', user_id)
                                         .stream()))
        
        convs = list(self.db.collection('conversations')
                    .where('user_id', '==', user_id)
                    .order_by('updated_at', direction=firestore.Query.DESCENDING)
                    .offset(start)
                    .limit(self.conversations_per_page)
                    .stream())
                    
        has_more = self._conv_count[user_id] > (start + self.conversations_per_page)
        result = (convs, has_more)
        
        self.cache.set(cache_key, result)
        return result

    def prepare_chat_messages(self, prompt: str, is_review: bool) -> List[Dict[str, str]]:
        """Prepare messages context with optimized structure"""
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
        
        if is_review:
            messages.append({
                "role": "system",
                "content": REVIEW_INSTRUCTIONS
            })
        
        if 'messages' in st.session_state:
            # Only include last N messages for context window optimization
            messages.extend(st.session_state.messages[-10:])
            
        messages.append({"role": "user", "content": prompt})
        return messages

    async def handle_chat_async(self, prompt: str):
        """Asynchronous chat handling for better performance"""
        if not prompt:
            return

        current_time = datetime.now(self.tz)
        time_str = self.format_time(current_time.isoformat())
        
        st.chat_message("user").write(f"{time_str} {prompt}")
        
        review_keywords = ["review", "assess", "grade", "evaluate", "score", "feedback"]
        is_review = any(keyword in prompt.lower() for keyword in review_keywords)
        
        messages = self.prepare_chat_messages(prompt, is_review)
        max_tokens = 5000 if is_review else 400

        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=max_tokens
            )
            
            assistant_content = response.choices[0].message.content
            if is_review:
                assistant_content = f"{assistant_content}\n\n{DISCLAIMER}"
                
            st.chat_message("assistant").write(f"{time_str} {assistant_content}")
            
            # Update session state and save to database concurrently
            self.update_conversation_state(prompt, assistant_content, time_str, current_time)
            
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")

    def update_conversation_state(self, prompt: str, assistant_content: str, 
                                time_str: str, current_time: datetime):
        """Update conversation state and database concurrently"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        user_message = {"role": "user", "content": prompt, "timestamp": time_str}
        assistant_msg = {"role": "assistant", "content": assistant_content, "timestamp": time_str}
        
        st.session_state.messages.extend([user_message, assistant_msg])
        
        # Save to database in background
        conversation_id = st.session_state.get('current_conversation_id')
        conversation_id = self.save_message(conversation_id, 
                                         {**user_message, "timestamp": current_time})
        self.save_message(conversation_id, 
                        {**assistant_msg, "timestamp": current_time})
        
        # Clear relevant caches
        self.cache.clear()
        self.format_time.cache_clear()
        self.generate_title.cache_clear()

    def login(self, email: str, password: str) -> bool:
        """Optimized login with caching"""
        cache_key = f"login_{email}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        try:
            auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={st.secrets['default']['apiKey']}"
            auth_data = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(auth_url, json=auth_data)
            if response.status_code != 200:
                raise Exception("Authentication failed")
                
            user = auth.get_user_by_email(email)
            st.session_state.user = user
            st.session_state.logged_in = True
            st.session_state.messages = [{
                **INITIAL_ASSISTANT_MESSAGE,
                "timestamp": self.format_time(datetime.now(self.tz).isoformat())
            }]
            st.session_state.stage = 'initial'
            
            self.cache.set(cache_key, True)
            return True
            
        except Exception as e:
            st.error("Login failed")
            self.cache.set(cache_key, False)
            return False

# Initialize database connection once
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["FIREBASE"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()

# Cache CSS styles
@st.cache_data
def load_css():
    return """
        <style>
            .main { max-width: 800px; margin: 0 auto; }
            .chat-message { padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; }
            #MainMenu, footer { visibility: hidden; }
        </style>
    """

def main():
    st.set_page_config(page_title="DUTE Essay Writing Assistant", layout="wide")
    st.markdown(load_css(), unsafe_allow_html=True)
    
    db = init_firebase()
    app = EWA()

    if not st.session_state.get('logged_in', False):
        st.title("DUTE Essay Writing Assistant")
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                if app.login(email, password):
                    st.rerun()
        return

    st.title("DUTE Essay Writing Assistant")
    app.render_sidebar()

    if 'messages' in st.session_state:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(
                f"{msg.get('timestamp', '')} {msg['content']}"
            )

    if prompt := st.chat_input("Type your message here..."):
        app.handle_chat(prompt)

if __name__ == "__main__":
    main()
