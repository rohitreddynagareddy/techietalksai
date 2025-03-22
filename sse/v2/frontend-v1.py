import streamlit as st
import requests
import time

st.title("📡 Real-Time SSE Events")

st.markdown("### Incoming Messages:")

# Container for messages
messages = st.empty()

def get_sse_messages():
    url = "http://sse-server:8000/sse"
    with requests.get(url, stream=True) as response:
        for line in response.iter_lines():
            if line:
                yield line.decode()

# Display messages in real-time
for msg in get_sse_messages():
    messages.text(msg)
    time.sleep(0.5)  # Smooth UI update

