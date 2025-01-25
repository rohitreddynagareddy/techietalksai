Here’s the complete `README.md` for your project, including the `docker-compose.yml` setup:

```markdown
# Smart Chat Assistant 🤖

A conversational AI assistant powered by DeepSeek's API, built with Streamlit and Pydantic. This application provides a user-friendly chat interface with structured response validation, conversation history, and rich debugging logs.

![Chat Interface Demo](https://via.placeholder.com/800x400.png?text=Chat+Assistant+Demo)

## Features ✨
- **Natural Language Conversations**: Interact with the AI in a conversational manner.
- **Response Validation**: Ensures structured responses using Pydantic models.
- **Conversation History**: Tracks and displays chat history.
- **Rich Debugging**: Detailed console logs for debugging and analysis.
- **Error Handling**: Graceful error handling and user feedback.
- **Docker Support**: Easy deployment using Docker.

---

## Installation 🛠️

### Local Setup

1. **Clone repository**:
```bash
git clone https://github.com/yourusername/smart-chat-assistant.git
cd smart-chat-assistant
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
   DEEPSEEK_API_KEY=your_deepseek_api_key
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
   - Open your browser and navigate to `http://localhost:8502`.

---

## Configuration ⚙️

1. **DeepSeek API Key**:
   - Obtain your API key from [DeepSeek Platform](https://platform.deepseek.com/).
   - Add it to the `.env` file:
   ```env
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ```

2. **Docker Configuration**:
   - The `docker-compose.yml` file maps port `8502` on your host to port `8501` in the container.
   - Environment variables are loaded from the `.env` file.

---

## Code Structure 📁

### `app.py`
```python
import streamlit as st
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessage
from rich import print
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY is not set in the environment or .env file.")
    st.stop()

# Configure DeepSeek model
model = OpenAIModel(
    model_name='deepseek-chat',
    base_url='https://api.deepseek.com/v1',
    api_key=DEEPSEEK_API_KEY,
)

# Pydantic response model
class AIResponse(BaseModel):
    content: str
    category: str = "general"

# Initialize AI agent
agent = Agent(
    model=model,
    result_type=AIResponse,
    system_prompt="You're a helpful assistant. Respond conversationally and keep answers concise.",
)

# Streamlit UI setup
st.title("💬 Smart Chat Assistant")
st.caption("Powered by DeepSeek + Pydantic_AI")

# New chat button
if st.button("🔄 New Chat", help="Start a new conversation", use_container_width=True):
    st.session_state.messages = []
    st.session_state.message_history = []
    st.rerun()

# Initialize message history
if "message_history" not in st.session_state:
    st.session_state["message_history"]: list[ModelMessage] = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input handling
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response with error handling
    try:
        with st.spinner("Analyzing your question..."):
            print("\n[bold]Message History:[/bold]")
            for i, msg in enumerate(st.session_state["message_history"]):
                print(f"\n[yellow]--- Message {i+1} ---[/yellow]")
                print(msg)
            result = asyncio.run(agent.run(prompt, message_history=st.session_state["message_history"]))
            st.session_state["message_history"].extend(result.new_messages())
            
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(result.data.content)
        
        # Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": result.data.content})
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
```

### `docker-compose.yml`
```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8502:8501"
    env_file:
      - .env  # Load environment variables from .env file
    volumes:
      - ./:/app  # Map current directory to container's /app
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
- [DeepSeek](https://platform.deepseek.com/) for powerful AI models.
- [Streamlit](https://streamlit.io/) for intuitive UI framework.
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation.
- [Rich](https://github.com/Textualize/rich) for beautiful console logging.
```

---

### Notes:
1. Replace `your_deepseek_api_key` with your actual DeepSeek API key.
2. Ensure the `.env` file is added to `.gitignore` to avoid exposing sensitive information.
3. The Docker setup assumes you have Docker and Docker Compose installed.
