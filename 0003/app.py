import streamlit as st
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import (
    ModelMessage,
)
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

model = OpenAIModel(
    model_name='deepseek-chat',  # DeepSeek model name
    base_url='https://api.deepseek.com/v1',  # DeepSeek API endpoint
    api_key=DEEPSEEK_API_KEY,  # Use DeepSeek API key
)

# Create agent with proper response model
class AIResponse(BaseModel):
    content: str
    category: str = "general"

agent = Agent(
    model=model,
    result_type=AIResponse,
    system_prompt=("You're a helpful assistant. Respond conversationally and keep answers concise."),
)

# Streamlit UI setup
st.title("💬 Smart Chat Assistant")
st.caption("Powered by DeepSeek + Pydantic_AI")

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
            # print(st.session_state["message_history"])

            result = asyncio.run(agent.run(prompt, message_history=st.session_state["message_history"]))
            st.session_state["message_history"].extend(result.new_messages())

            print("\n[bold]Message History:[/bold]")
            for i, msg in enumerate(st.session_state["message_history"]):
                print(f"\n[yellow]--- Message {i+1} ---[/yellow]")
                print(msg)            
            
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(result.data.content)
        
        # Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": result.data.content})
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
