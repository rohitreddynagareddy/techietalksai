import streamlit as st
import requests
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx

# st.set_page_config(page_title="📡 Real-Time SSE", layout="wide")
st.title("📡 Real-Time SSE with Acknowledgments")

st.markdown("### Incoming Messages:")
messages_container = st.empty()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to fetch SSE messages continuously
def get_sse_messages():
    url = "http://sse-server:8000/sse"
    with requests.get(url, stream=True) as response:
        for line in response.iter_lines():
            if line:
                decoded_message = line.decode()
                print(f"📩 Received: {decoded_message}")  # Debugging
                st.session_state.messages.append(decoded_message)

# Start SSE listener in a separate thread (only once)
if "sse_thread" not in st.session_state:
    sse_thread = threading.Thread(target=get_sse_messages, daemon=True)
    add_script_run_ctx(sse_thread)
    sse_thread.start()
    st.session_state.sse_thread = sse_thread

st.markdown("### Send a Message to the Server:")
user_input = st.text_input("Enter your message:")
if st.button("Send"):
    response = requests.post("http://sse-server:8000/messages", json={"message": user_input})
    st.success(f"Server Response: {response.json()}")



# Keep updating UI with the latest messages
while True:
    if st.session_state.messages:
        messages_container.text("\n".join(st.session_state.messages[-5:]))  # Show last 5 messages
    time.sleep(1)  # Prevent UI freeze
    st.rerun()  # Force UI refresh

