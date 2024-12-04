import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
from openai import OpenAI
from datetime import datetime
import pytz
import requests

# Import configurations
from initial import INITIAL_ASSISTANT_MESSAGE
from reviewprocess import SYSTEM_INSTRUCTIONS, REVIEW_INSTRUCTIONS, DISCLAIMER, SCORING_CRITERIA

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["FIREBASE"]))
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Page setup
st.set_page_config(page_title="DUTE Essay Writing Assistant", layout="wide")
st.markdown("""
    <style>
        .main { max-width: 800px; margin: 0 auto; }
        .chat-message { padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; }
        #MainMenu, footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

class EWA:
    REVIEW_KEYWORDS = {"grade", "score", "review", "assess", "evaluate", "feedback", "rubric"}
    MAX_TOKENS_REVIEW = 5000
    MAX_TOKENS_NORMAL = 600
    CONTEXT_WINDOW_REVIEW = 10
    CONTEXT_WINDOW_NORMAL = 6

    
    def __init__(self):        
        self.tz = pytz.timezone("Europe/London")
        self.conversations_per_page = 10  # Number of conversations per page
        # Add OpenAI client initialization
        self.openai_client = OpenAI(api_key=st.secrets["default"]["OPENAI_API_KEY"])


    def format_time(self, dt=None):
        """Format datetime with consistent timezone"""
        if isinstance(dt, (datetime, type(firestore.SERVER_TIMESTAMP))):
            return dt.strftime("[%Y-%m-%d %H:%M:%S]")
        dt = dt or datetime.now(self.tz)
        return dt.strftime("[%Y-%m-%d %H:%M:%S]")           

    def get_conversations(self, user_id):
        """Retrieve conversation history from Firestore"""
        # Get total conversation count
        count = len(list(db.collection('conversations')
                        .where('user_id', '==', user_id)
                        .stream()))
    
        # Calculate start position based on current page
        page = st.session_state.get('page', 0)
        start = page * 10
    
        return db.collection('conversations')\
                 .where('user_id', '==', user_id)\
                 .order_by('updated_at', direction=firestore.Query.DESCENDING)\
                 .offset(start)\
                 .limit(10)\
                 .stream(), count > (start + 10)

    def render_sidebar(self):
        """Render sidebar with conversation history"""
        with st.sidebar:
            st.title("Essay Writing Assistant")
        
            if st.button("New Session"):
                user = st.session_state.user
                st.session_state.clear()
                st.session_state.user = user
                st.session_state.logged_in = True
                st.session_state.messages = [
                    {**INITIAL_ASSISTANT_MESSAGE, "timestamp": self.format_time()}
                ]
                st.session_state.page = 0
                st.rerun()
            
            if st.button("Latest Chat History"):
                st.session_state.page = 0
                st.rerun()
            
            st.divider()
        
            # Initialize page if not exists
            if 'page' not in st.session_state:
                st.session_state.page = 0
            
            # Get conversations and has_more flag
            convs, has_more = self.get_conversations(st.session_state.user.uid)
        
            # Display conversations
            for conv in convs:
                conv_data = conv.to_dict()
                if st.button(f"{conv_data.get('title', 'Untitled')}", key=conv.id):
                    messages = db.collection('conversations').document(conv.id)\
                               .collection('messages').order_by('timestamp').stream()
                    st.session_state.messages = []
                    for msg in messages:
                        msg_dict = msg.to_dict()
                        if 'timestamp' in msg_dict:
                            msg_dict['timestamp'] = self.format_time(msg_dict['timestamp'])
                        st.session_state.messages.append(msg_dict)
                    st.session_state.current_conversation_id = conv.id
                    st.rerun()
            
            # Simple pagination controls
            cols = st.columns(2)
            with cols[0]:
                if st.session_state.page > 0:
                    if st.button("Previous"):
                        st.session_state.page -= 1
                        st.rerun()
            with cols[1]:
                if has_more:
                    if st.button("Next"):
                        st.session_state.page += 1
                        st.rerun()
    
    def get_recent_messages(self, conversation_id, limit):
        """Get recent messages from a conversation"""
        if not conversation_id:
            # If no conversation_id, return initial message
            return [INITIAL_ASSISTANT_MESSAGE] if 'messages' not in st.session_state else st.session_state.messages
        
        messages = (
            db.collection('conversations')
            .document(conversation_id)
            .collection('messages')
            .order_by('timestamp', direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
    
        msg_list = [
            {
                'role': msg.to_dict().get('role'),
                'content': msg.to_dict().get('content')
            }
            for msg in messages
        ]
    
        # Ensure initial message is included if this is start of conversation
        if not msg_list and 'messages' in st.session_state:
            return [msg for msg in st.session_state.messages if msg['role'] == 'assistant'][:1]
    
        return msg_list
    
    def handle_chat(self, prompt):
        """Process chat messages and manage conversation flow"""
        if not prompt:
            return

        current_time = datetime.now(self.tz)
        time_str = self.format_time(current_time)

        # Display user message
        st.chat_message("user").write(f"{time_str} {prompt}")

        try:
            # Build message context
            messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
            if 'messages' in st.session_state:
                # Always include the initial assistant message first
                initial_message = st.session_state.messages[0]
                messages.append({
                    "role": "assistant",
                    "content": initial_message['content']
                })
            
            # Then add recent context
            context_window = self.CONTEXT_WINDOW_REVIEW if any(keyword in prompt.lower() for keyword in self.REVIEW_KEYWORDS) else self.CONTEXT_WINDOW_NORMAL
            
            # Add recent messages but skip the initial one we already added
            recent_messages = st.session_state.messages[1:][-context_window:]
            for msg in recent_messages:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Get AI response with proper context
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=self.MAX_TOKENS_REVIEW if any(keyword in prompt.lower() for keyword in self.REVIEW_KEYWORDS) else self.MAX_TOKENS_NORMAL
        )

        assistant_content = response.choices[0].message.content
        
            # 6. Process Response
            assistant_content = response.choices[0].message.content
            if is_review:
                assistant_content = f"{assistant_content}\n\n{DISCLAIMER}"
            
            st.chat_message("assistant").write(f"{time_str} {assistant_content}")

            # 7. Update Session State
            if 'messages' not in st.session_state:
                st.session_state.messages = []

            new_messages = [
                {"role": "user", "content": prompt, "timestamp": time_str},
                {"role": "assistant", "content": assistant_content, "timestamp": time_str}
            ]
        
            st.session_state.messages.extend(new_messages)

            # 8. Save to Database
            for msg in new_messages:
                conversation_id = self.save_message(conv_id, 
                                                 {**msg, "timestamp": current_time})
        
            return True

        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
            return False

    def save_message(self, conversation_id, message):
        """Save message and update conversation metadata"""
        current_time = datetime.now(self.tz)
    
        try:
            # For new conversation
            if not conversation_id:
                conv_ref = db.collection('conversations').document()
                conversation_id = conv_ref.id
                conv_ref.set({
                    'user_id': st.session_state.user.uid,
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'updated_at': firestore.SERVER_TIMESTAMP,
                    'title': f"{current_time.strftime('%b %d, %Y')} • New Chat [1📝]",
                    'status': 'active'
                })
                st.session_state.current_conversation_id = conversation_id
                return conversation_id
        
            # For existing conversation
            conv_ref = db.collection('conversations').document(conversation_id)
            conv_ref.collection('messages').add({
                **message,
                "timestamp": firestore.SERVER_TIMESTAMP
            })

            # Get messages for count and context
            messages = list(conv_ref.collection('messages').get())
            count = len(messages)
        
            # Get last 5 messages for context
            recent_messages = [msg.to_dict()['content'] for msg in messages[-5:]]
            context = " ".join(recent_messages)
        
            # Get summary from GPT
            summary = OpenAI(api_key=st.secrets["default"]["OPENAI_API_KEY"]).chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Create a 2-3 word title for this conversation."},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=10
            ).choices[0].message.content.strip()
        
            # Update conversation with summary title and count
            conv_ref.set({
                'updated_at': firestore.SERVER_TIMESTAMP,
                'title': f"{current_time.strftime('%b %d, %Y')} • {summary} [{count}📝]"
            }, merge=True)
        
            return conversation_id
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return conversation_id
    
    def login(self, email, password):
        """Authenticate user with Firebase Auth REST API"""
        try:
            # Firebase Auth REST API endpoint
            auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={st.secrets['default']['apiKey']}"
        
            # Request body
            auth_data = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
        
            # Make authentication request
            response = requests.post(auth_url, json=auth_data)
            if response.status_code != 200:
                raise Exception("Authentication failed")
            
            # Get user details
            user = auth.get_user_by_email(email)
            st.session_state.user = user
            st.session_state.logged_in = True 
            st.session_state.messages = [{
                **INITIAL_ASSISTANT_MESSAGE,
                "timestamp": self.format_time()
                "role": "assistant"
            }]
            st.session_state.stage = 'initial'
            return True
        
        except Exception as e:
            st.error("Login failed")
            return False
        
def main():
    app = EWA()

    # Login page
    if not st.session_state.get('logged_in', False):
        st.title("DUTE Essay Writing Assistant")
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                if app.login(email, password):
                    st.rerun()
        return

    # Main chat interface
    st.title("DUTE Essay Writing Assistant")
    app.render_sidebar()

    # Display message history
    if 'messages' in st.session_state:
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")  # Default to assistant if role not set
            formatted_msg = f"{msg.get('timestamp', '')} {msg['content']}"
            st.chat_message(role).write(formatted_msg)
            
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        app.handle_chat(prompt)

if __name__ == "__main__":
    main()
