import streamlit as st
# Page configuration
st.set_page_config(
    page_title="DUTE Essay Writing Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"  # Reduces initial load time
)
import firebase_admin
from firebase_admin import credentials, auth, firestore
from openai import OpenAI
from datetime import datetime
import pytz
import requests

# Import configurations
from initial import INITIAL_ASSISTANT_MESSAGE
from reviewprocess import SYSTEM_INSTRUCTIONS, REVIEW_INSTRUCTIONS, DISCLAIMER, SCORING_CRITERIA

# Cache the Firebase client initialization
@st.cache_resource
def get_firebase_client():
    """Initialize and cache Firebase client"""
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["FIREBASE"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()

# Initialize Firebase with caching
db = get_firebase_client()

# Cached styles
st.markdown(
    """
    <style>
        .main { max-width: 800px; margin: 0 auto; }
        .chat-message { padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; }
        #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

class EWA:
    def __init__(self):
        self.tz = pytz.timezone("Europe/London")
        self.conversations_per_page = 10
        self.db = get_firebase_client()

    @st.cache_data(ttl=60)  # Cache for 1 minute
    def format_time(self, dt=None):
        """Format datetime with consistent timezone"""
        if isinstance(dt, (datetime, type(firestore.SERVER_TIMESTAMP))):
            return dt.strftime("[%Y-%m-%d %H:%M:%S]")
        dt = dt or datetime.now(self.tz)
        return dt.strftime("[%Y-%m-%d %H:%M:%S]")

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def generate_title(self, message_content, current_time):
        """Generate title from date and first 4 words of message"""
        title = current_time.strftime('%b %d, %Y • ') + ' '.join(message_content.split()[:4])
        return title[:50] if len(title) > 50 else title

    @st.cache_data(ttl=300)
    def get_conversations_batch(self, user_id, page, limit):
        """Cache conversation retrieval"""
        conversations = list(self.db.collection('conversations')
            .where('user_id', '==', user_id)
            .order_by('updated_at', direction=firestore.Query.DESCENDING)
            .offset(page * limit)
            .limit(limit)
            .stream())
        return conversations

    def get_conversations(self, user_id):
        """Retrieve conversation history from Firestore"""
        page = st.session_state.get('page', 0)
        
        # Get total count efficiently
        count_query = self.db.collection('conversations')\
            .where('user_id', '==', user_id)\
            .count()
        total_count = count_query.get()[0][0].value
        
        conversations = self.get_conversations_batch(
            user_id, 
            page, 
            self.conversations_per_page
        )
        
        return conversations, total_count > ((page + 1) * self.conversations_per_page)

    def render_sidebar(self):
        """Render optimized sidebar"""
        with st.sidebar:
            st.title("Essay Writing Assistant")
            
            # Group buttons in columns
            col1, col2 = st.columns(2)
            with col1:
                if st.button("New Session", use_container_width=True):
                    self.clear_session_except_user()
                    st.rerun()
            
            with col2:
                if st.button("Latest Chat", use_container_width=True):
                    st.session_state.page = 0
                    st.rerun()
            
            st.divider()
            
            # Get cached conversations
            convs, has_more = self.get_conversations(st.session_state.user.uid)
            
            # Display conversations efficiently
            for conv in convs:
                conv_data = conv.to_dict()
                if st.button(
                    f"{conv_data.get('title', 'Untitled')}",
                    key=conv.id,
                    use_container_width=True
                ):
                    self.load_conversation(conv.id)
                    st.rerun()
            
            # Optimize pagination
            if st.session_state.page > 0 or has_more:
                cols = st.columns(2)
                with cols[0]:
                    if st.session_state.page > 0 and st.button("◀", use_container_width=True):
                        st.session_state.page -= 1
                        st.rerun()
                with cols[1]:
                    if has_more and st.button("▶", use_container_width=True):
                        st.session_state.page += 1
                        st.rerun()

    @st.cache_data(ttl=300)
    def get_conversation_messages(self, conv_id):
        """Cache message retrieval"""
        messages = list(self.db.collection('conversations')
            .document(conv_id)
            .collection('messages')
            .order_by('timestamp')
            .stream())
        return messages

    def load_conversation(self, conv_id):
        """Load conversation with cached messages"""
        messages = self.get_conversation_messages(conv_id)
        st.session_state.messages = []
        
        for msg in messages:
            msg_dict = msg.to_dict()
            if 'timestamp' in msg_dict:
                msg_dict['timestamp'] = self.format_time(msg_dict['timestamp'])
            st.session_state.messages.append(msg_dict)
        
        st.session_state.current_conversation_id = conv_id

    def save_message(self, conversation_id, message):
        """Save message with optimized database operations"""
        current_time = datetime.now(self.tz)
        firestore_time = firestore.SERVER_TIMESTAMP

        try:
            batch = self.db.batch()  # Use batch writes for better performance
            
            if not conversation_id:
                new_conv_ref = self.db.collection('conversations').document()
                conversation_id = new_conv_ref.id
                
                if message['role'] == 'user':
                    title = self.generate_title(message['content'], current_time)
                    batch.set(new_conv_ref, {
                        'user_id': st.session_state.user.uid,
                        'created_at': firestore_time,
                        'updated_at': firestore_time,
                        'title': title,
                        'status': 'active'
                    })
                    st.session_state.current_conversation_id = conversation_id

            if conversation_id:
                conv_ref = self.db.collection('conversations').document(conversation_id)
                msg_ref = conv_ref.collection('messages').document()
                
                batch.set(msg_ref, {
                    **message,
                    "timestamp": firestore_time
                })
                
                batch.update(conv_ref, {
                    'updated_at': firestore_time,
                    'last_message': message['content'][:100]
                })
                
                # Commit batch
                batch.commit()
            
            return conversation_id
            
        except Exception as e:
            st.error(f"Error saving message: {str(e)}")
            return conversation_id

    def handle_chat(self, prompt):
        """Process chat messages with optimized message handling"""
        if not prompt:
            return

        current_time = datetime.now(self.tz)
        time_str = self.format_time(current_time)

        # Display user message
        st.chat_message("user").write(f"{time_str} {prompt}")

        # Build messages context
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
        
        # Check for review/scoring keywords efficiently
        is_review = any(keyword in prompt.lower() for keyword in [
            "review", "assess", "grade", "evaluate", "score", "feedback"
        ])
        
        if is_review:
            messages.append({"role": "system", "content": REVIEW_INSTRUCTIONS})
            max_tokens = 5000
        else:
            max_tokens = 400

        # Add conversation history
        if 'messages' in st.session_state:
            messages.extend(st.session_state.messages)

        messages.append({"role": "user", "content": prompt})

        try:
            # Get AI response
            response = OpenAI(
                api_key=st.secrets["default"]["OPENAI_API_KEY"]
            ).chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=max_tokens
            )

            assistant_content = response.choices[0].message.content
            
            if is_review:
                assistant_content = f"{assistant_content}\n\n{DISCLAIMER}"
                
            st.chat_message("assistant").write(f"{time_str} {assistant_content}")

            # Update session state efficiently
            user_message = {"role": "user", "content": prompt, "timestamp": time_str}
            assistant_msg = {"role": "assistant", "content": assistant_content, "timestamp": time_str}
            
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            st.session_state.messages.extend([user_message, assistant_msg])

            # Save to database efficiently
            conversation_id = st.session_state.get('current_conversation_id')
            conversation_id = self.save_message(
                conversation_id, 
                {**user_message, "timestamp": current_time}
            )
            self.save_message(
                conversation_id,
                {**assistant_msg, "timestamp": current_time}
            )

        except Exception as e:
            st.error(f"Error processing message: {str(e)}")

    def clear_session_except_user(self):
        """Helper method for efficient session clearing"""
        user = st.session_state.user
        st.session_state.clear()
        st.session_state.user = user
        st.session_state.logged_in = True
        st.session_state.messages = [{
            **INITIAL_ASSISTANT_MESSAGE,
            "timestamp": self.format_time()
        }]

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def login(self, email, password):
        """Cache successful login attempts"""
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
            return user
            
        except Exception as e:
            st.error("Login failed")
            return None

def main():
    app = EWA()

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.page = 0

    # Login page
    if not st.session_state.get('logged_in', False):
        st.title("DUTE Essay Writing Assistant")
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                user = app.login(email, password)
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.session_state.messages = [{
                        **INITIAL_ASSISTANT_MESSAGE,
                        "timestamp": app.format_time()
                    }]
                    st.rerun()
        return

    # Main chat interface
    st.title("DUTE Essay Writing Assistant")
    app.render_sidebar()

    # Display message history efficiently
    if 'messages' in st.session_state:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(
                f"{msg.get('timestamp', '')} {msg['content']}"
            )

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        app.handle_chat(prompt)

if __name__ == "__main__":
    main()
