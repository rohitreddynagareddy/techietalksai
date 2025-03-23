import streamlit as st
import asyncio
import websockets
import threading

st.title("📡 WebSocket Real-Time Chat")

# ✅ Maintain session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

ws_url = "ws://ws-server:8000/ws"

# ✅ Function to listen for WebSocket messages
def receive_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    async def listen():        
        async with websockets.connect(ws_url) as ws:
            while True:
                message = await ws.recv()
                st.session_state.messages.append(f"📩 Received: {message}")
                st.rerun()  # ✅ Updates the UI in Streamlit

    loop = asyncio.new_event_loop()  # ✅ Create a new event loop for the thread
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listen())

# ✅ Start WebSocket listener in a background thread
if "ws_thread" not in st.session_state:
    ws_thread = threading.Thread(target=receive_messages, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread  # Store in session state

# ✅ Display received messages
for msg in st.session_state.messages[-5:]:  # Show last 5 messages
    st.write(msg)

# ✅ Function to send a message over WebSocket
async def send_message(msg):
    async with websockets.connect(ws_url) as ws:
        await ws.send(msg)
        response = await ws.recv()  # ✅ Get the response from the server
        st.session_state.messages.append(f"✅ Server: {response}")
        st.rerun()

# ✅ Input box for sending messages
user_input = st.text_input("Enter your message:")
if st.button("Send"):
    asyncio.run(send_message(user_input))  # ✅ Runs the async function safely
    st.success("Message Sent!")
