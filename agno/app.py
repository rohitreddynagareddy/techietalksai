import streamlit as st
from typing import Iterator
import os
from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek
from openai import OpenAI
import asyncio

# Streamlit app
st.title("NYC News Reporter")

counter_placeholder = st.empty()


# Load OpenAI API key from .env
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["model"] = "gpt-4o-mini"

if "counter" not in st.session_state:
    st.session_state["counter"] = 0

st.session_state["counter"] += 1
counter_placeholder.write(st.session_state["counter"])



# Set OpenAI API key from Streamlit secrets
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = OpenAI(api_key=openai_api_key)

# class AsyncGeneratorWrapper:
#     def __init__(self, gen):
#         self.gen = gen

#     def __aiter__(self):
#         return self

#     async def __anext__(self):
#         try:
#             return next(self.gen)
#         except StopIteration:
#             raise StopAsyncIteration

# async def run_agent(agent, prompt):
#     response_gen = agent.run(prompt, stream=True)
#     async_response = AsyncGeneratorWrapper(response_gen)
#     collected_content = ""
#     text_placeholder = st.empty()  # Create a placeholder in the Streamlit app
#     async for chunk in async_response:
#         if hasattr(chunk, 'content') and chunk.content:
#             collected_content += chunk.content
#             # st.write(chunk.content)
#             text_placeholder.markdown(collected_content)  # Update the placeholder with the new content
#     return collected_content

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create our News Reporter with a fun personality
agent = Agent(
    # model=OpenAIChat(id="gpt-4o-mini"),
    model=DeepSeek(id="deepseek-chat"),
    instructions=dedent("""\
        You are an enthusiastic news reporter with a flair for storytelling! 🗽
        Think of yourself as a mix between a witty comedian and a sharp journalist.

        Your style guide:
        - Start with an attention-grabbing headline using emoji
        - Share news with enthusiasm and NYC attitude
        - Keep your responses concise but entertaining
        - Throw in local references and NYC slang when appropriate
        - End with a catchy sign-off like 'Back to you in the studio!' or 'Reporting live from the Big Apple!'

        Remember to verify all facts while keeping that NYC energy high!\
    """),
    markdown=True,
)

# import random
# import string
# # async 
# def booking(ctx: RunContext[Deps], name: str = "Guest") -> str:
#     """Generate a random booking reference number as confirmation."""
#     if name == "Guest":
#         # Prompt the user to provide their name
#         return "Please provide your name to complete the booking."
#     else:
#         # Generate a random alphanumeric booking reference
#         booking_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
#         return f"Your booking is confirmed, {name}! Booking Reference: {booking_ref}"



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What is up?"):

    st.session_state["counter"] = 0

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)


    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        # THIS WORKS!
        # stream = client.chat.completions.create(
        #     model=st.session_state["model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        # response = st.write_stream(stream)

        # THIS WORKS! NO STREAM
        # response = agent.run(prompt, stream=False)
        # response = response.content
        # st.write(response)


        # THIS WORKS! WITH STREAM
        # response = asyncio.run(run_agent(agent, prompt))

        # FINALLY THIS WORKS! WITH STREAM
        stream = False 
        if stream:
            run_response: Iterator[RunResponse] = agent.run(prompt, stream=True)
            response = ""
            text_placeholder = st.empty()
            for chunk in run_response:
                # st.write(chunk.content)
                response += chunk.content
                text_placeholder.markdown(response)
                st.session_state["counter"] += 1
                counter_placeholder.write(st.session_state["counter"])
        else:
            response = agent.run(prompt, stream=False)
            response = response.content
            st.write(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})



# User input
# user_input = st.text_input("Ask the news reporter something:", value="Tell me about a breaking news story happening in Times Square.")

# # Generate response
# if st.button("Get News"):
#     response = agent.print_response(user_input)  # Use the correct method
#     asyncio.run(agent.aprint_response("Share a breakfast recipe.", stream=True))
#     st.write(response)
