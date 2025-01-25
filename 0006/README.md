Here’s the complete `README.md` for your project, including the `docker-compose.yml` setup:

```markdown
# Multi-Model AI Chat Assistant 🤖

A versatile conversational AI assistant that supports multiple AI models (OpenAI, DeepSeek, and Gemini). Built with Streamlit, Pydantic, and Rich, this application provides a user-friendly chat interface with structured response validation, conversation history, and rich debugging logs.

![Chat Interface Demo](https://via.placeholder.com/800x400.png?text=Chat+Assistant+Demo)

## Features ✨
- **Multi-Model Support**: Switch between OpenAI, DeepSeek, and Gemini models.
- **Conversation History**: Track and display chat history with model-specific avatars.
- **Rich Debugging**: Detailed console logs for debugging and analysis.
- **Error Handling**: Graceful error handling and user feedback.
- **Custom UI**: Enhanced Streamlit UI with custom styles and buttons.
- **Docker Support**: Easy deployment using Docker and Nginx for web serving.

---

## Installation 🛠️

### Local Setup

1. **Clone repository**:
```bash
git clone https://github.com/yourusername/multi-model-chat-assistant.git
cd multi-model-chat-assistant
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
   - Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

5. **Run the application**:
```bash
streamlit run app.py
```

---

### Docker Setup

1. **Build and run the Docker container**:
```bash
docker-compose up --build
```

2. **Access the application**:
   - Chat Assistant: `http://localhost:8502`
   - Web Server: `http://localhost:8503`

---

## Configuration ⚙️

1. **API Keys**:
   - Obtain API keys from:
     - [OpenAI Platform](https://platform.openai.com/)
     - [DeepSeek Platform](https://platform.deepseek.com/)
     - [Gemini Platform](https://ai.google.dev/)
   - Add them to the `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

2. **Docker Configuration**:
   - The `docker-compose.yml` file includes:
     - `chatbot`: Streamlit app (port `8502`).
     - `webserver`: Nginx server for static files (port `8503`).
   - Environment variables are loaded from the `.env` file.

---

## Code Structure 📁

### `app.py`
```python
import streamlit as st
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.messages import ModelMessage
from rich import print
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle event loops
nest_asyncio.apply()

# Streamlit page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for UI enhancements
st.markdown("""
<style>
    .stButton>button {
        background-color: #f0f2f6;
        color: #2c3e50;
        border: 1px solid #ced4da;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e2e6ea;
        border-color: #adb5bd;
    }
</style>
""", unsafe_allow_html=True)

# Model selection and new chat button
col1, col2 = st.columns([3, 1])
with col1:
    MODEL_CHOICE = st.selectbox("Choose AI Model", ["OpenAI", "DeepSeek", "Gemini"])

with col2:
    if st.button("🔄 New Chat", help="Start a new conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.message_history = []
        st.rerun()

# Load environment variables
load_dotenv()

# Validate API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in the environment or .env file.")
    st.stop()
if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY is not set in the environment or .env file.")
    st.stop()
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set in the environment or .env file.")
    st.stop()

# Initialize selected model
if MODEL_CHOICE == "OpenAI":
    model = OpenAIModel(
        model_name='gpt-3.5-turbo',
        base_url='https://api.openai.com/v1',
        api_key=OPENAI_API_KEY,
    )
elif MODEL_CHOICE == "DeepSeek":
    model = OpenAIModel(
        model_name='deepseek-chat',
        base_url='https://api.deepseek.com/v1',
        api_key=DEEPSEEK_API_KEY,
    )
elif MODEL_CHOICE == "Gemini":
    model = GeminiModel(
        model_name='gemini-1.5-flash',
        api_key=GEMINI_API_KEY,
    )

# Pydantic response model
class AIResponse(BaseModel):
    content: str
    category: str = "general"

# Initialize session states
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize AI agent
agent = Agent(
    model=model,
    result_type=AIResponse,
    system_prompt="You're a helpful assistant. Respond conversationally and keep answers concise.",
)

# Model-specific avatars
MODEL_AVATARS = {
    "OpenAI": "🦾",  # Robot arm emoji
    "DeepSeek": "🚀",  # Rocket emoji
    "Gemini": "🤖"   # Robot face emoji
}

# UI Setup
st.title("💬 Multi-Model Chat Assistant")
st.caption(f"Currently using: {MODEL_CHOICE} {MODEL_AVATARS[MODEL_CHOICE]}")

# Display chat history
for message in st.session_state.messages:
    avatar = MODEL_AVATARS[message.get("model")] if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Generating response..."):
            # Event loop management
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Execute async operation
            result = loop.run_until_complete(
                agent.run(
                    prompt,
                    message_history=st.session_state.message_history
                )
            )
            
            # Update message history
            new_messages = result.new_messages()
            if MODEL_CHOICE == "OpenAI":
                for msg in new_messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if len(tool_call.id) > 40:
                                tool_call.id = tool_call.id[:40]
            
            st.session_state.message_history.extend(new_messages)

            print("\n[bold]Message History:[/bold]")
            for i, msg in enumerate(st.session_state["message_history"]):
                print(f"\n[yellow]--- Message {i+1} ---[/yellow]")
                print(msg)

        # Display assistant response
        with st.chat_message("assistant", avatar=MODEL_AVATARS[MODEL_CHOICE]):
            st.markdown(result.data.content)        

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.data.content,
            "model": MODEL_CHOICE  # Store model info with the message
        })
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)}"
        })
```

### `docker-compose.yml`
```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8502:8501"  # Streamlit default port
    env_file:
      - .env
    volumes:
      - ./:/app
    networks:
      - app-network

  webserver:
    image: nginx:alpine
    ports:
      - "8503:80"  # Web server port
    volumes:
      - ./web/static:/usr/share/nginx/html  # Directory containing index.html
    networks:
      - app-network
    depends_on:
      - chatbot

networks:
  app-network:
    driver: bridge
```

---

## Contributing 🤝

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---

## License 📄

MIT License - See [LICENSE](LICENSE) for details.

---

**Acknowledgments**  
- [OpenAI](https://platform.openai.com/) for GPT models.
- [DeepSeek](https://platform.deepseek.com/) for DeepSeek models.
- [Gemini](https://ai.google.dev/) for Gemini models.
- [Streamlit](https://streamlit.io/) for intuitive UI framework.
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation.
- [Rich](https://github.com/Textualize/rich) for beautiful console logging.
```

---

### Notes:
1. Replace `your_openai_api_key`, `your_deepseek_api_key`, and `your_gemini_api_key` with your actual API keys.
2. Ensure the `.env` file is added to `.gitignore` to avoid exposing sensitive information.
3. The Docker setup assumes you have Docker and Docker Compose installed.
