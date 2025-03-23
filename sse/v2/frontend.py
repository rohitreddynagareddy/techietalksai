import streamlit as st
import httpx
import asyncio

st.title("📡 Real-Time SSE Events")
st.markdown("### Incoming Messages:")

# Container for messages
st.subheader("Messages:")
messages = st.empty()
st.subheader("Messages again:")
async def get_sse_messages():
    url = "http://sse-server:8000/sse"
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            async for line in response.aiter_lines():
                if line:
                    yield line

async def display_messages():
    async for msg in get_sse_messages():
        messages.text(msg)
        st.write(msg)
        await asyncio.sleep(0.1)  # Slight pause to simulate processing time

# Run the asynchronous display function
asyncio.run(display_messages())
