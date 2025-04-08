import streamlit as st
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat

# Initialize session state for chat history and reasoning mode
if "messages" not in st.session_state:
    st.session_state.messages = []

if "reasoning" not in st.session_state:
    st.session_state.reasoning = False

# --------------- SESSION MANAGEMENT -------------------
def init_session():
    st.session_state.session_id = None
    st.session_state.user_id = "streamlit_user"
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    init_session()

if "agent" not in st.session_state:
    # Initialize the agent with or without reasoning
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"), 
        reasoning=st.session_state.reasoning, 
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


# Streamlit UI setup
st.title("Reasoning AGNO Agent")

# Slider to toggle reasoning mode
# st.session_state.reasoning = st.sidebar.slider("Reasoning Mode", 0, 1, 0) == 1



def init_agent():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"), 
        reasoning=st.session_state.reasoning, 
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

st.session_state.reasoning = st.toggle("Reasoning Mode", value=False, on_change=init_agent)


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
    run_response = st.session_state.agent.run(prompt, stream=True, show_reasoning=True, show_full_reasoning=True)

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

