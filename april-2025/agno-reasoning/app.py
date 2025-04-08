import streamlit as st
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

# Initialize session state for chat history and reasoning mode
if "messages" not in st.session_state:
    st.session_state.messages = []

if "reasoning" not in st.session_state:
    st.session_state.reasoning = False

def init_session():
    st.session_state.session_id = None
    st.session_state.user_id = "streamlit_user"
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    init_session()

# reason = False
# st.title("Reasoning AGNO Agent")
# Toggle for reasoning
reason = st.toggle("Enable Reasoning", value=False)

# Display current toggle value (optional for debug/demo)
# st.write("Reasoning is:", "ON" if reason else "OFF")

st.title("🧠 Reasoning AGNO Agent" if reason else "💬 Regular AGNO Agent")


def init_agent():

    if not reason:
        
        agent = Agent(
            # model=OpenAIChat(id="gpt-4o-mini"), 
            model=OpenAIChat(id="gpt-3.5-turbo"), 
            # reasoning=reason, 
            # reasoning_model=Groq(id="deepseek-r1-distill-llama-70b"),
            markdown=True,
            # Show tool calls in the response
            show_tool_calls=True,
            # To provide the agent with the chat history
            # We can either:
            # 1. Provide the agent with a tool to read the chat history
            # 2. Automatically add the chat history to the messages sent to the model
            #
            # 1. Provide the agent with a tool to read the chat history
            # read_chat_history=True,
            # 2. Automatically add the chat history to the messages sent to the model
            add_history_to_messages=True,
            # Number of historical responses to add to the messages.
            num_history_responses=3,
            )
        st.session_state.agent = agent

        st.success(f"Agent Recreated without Reasoning")
    else:
        # st.title("Reasoning AGNO Agent")
        agent = Agent(
                model=OpenAIChat(id="gpt-3.5-turbo"), 
                reasoning=True, 
                reasoning_model=Groq(id="deepseek-r1-distill-llama-70b"),
                markdown=True,
                # Show tool calls in the response
                show_tool_calls=True,
                # To provide the agent with the chat history
                # We can either:
                # 1. Provide the agent with a tool to read the chat history
                # 2. Automatically add the chat history to the messages sent to the model
                #
                # 1. Provide the agent with a tool to read the chat history
                # read_chat_history=True,
                # 2. Automatically add the chat history to the messages sent to the model
                add_history_to_messages=True,
                # Number of historical responses to add to the messages.
                num_history_responses=3,
                )
        st.session_state.agent = agent

        st.success(f"Agent Recreated with Reasoning")


init_agent()

# st.session_state.reasoning = st.toggle("Reasoning Mode")
# if "prev_reasoning" not in st.session_state or st.session_state.prev_reasoning != st.session_state.reasoning:
#     init_agent()
#     st.session_state.prev_reasoning = st.session_state.reasoning

if "agent" not in st.session_state:
    # Initialize the agent with or without reasoning
    init_agent()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # with st.expander("Reasoning - Thinking Process", expanded=True):

    response = ""
    placeholder = st.empty()
    if reason:
        run_response =  st.session_state.agent.run(prompt, stream=True, show_reasoning=True, show_full_reasoning=True)
    else:
        run_response =  st.session_state.agent.run(prompt, stream=True)

    for _resp_chunk in run_response:
        if _resp_chunk.content is not None:
            response += _resp_chunk.content
            placeholder.markdown(response + "|")
    placeholder.markdown("")

    # Update session ID if new session
    if st.session_state.session_id is None:
        st.session_state.session_id = st.session_state.agent.session_id
        
    # Store response in history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Display response
    # st.markdown(response)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

