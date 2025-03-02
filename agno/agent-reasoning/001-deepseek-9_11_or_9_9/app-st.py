import streamlit as st
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.deepseek import DeepSeek
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from typing import Iterator
from agno.agent import Agent, RunResponse
import re
# --------------- SETUP ---------------
if "agents" not in st.session_state:
    st.session_state.agents = {
        # "claude_regular": Agent(model=Claude("claude-3-5-sonnet-20241022")),
        # "claude_reasoning": Agent(
        #     model=Claude("claude-3-5-sonnet-20241022"),
        #     reasoning_model=DeepSeek(id="deepseek-reasoner"),
        # ),
        "openai_regular": Agent(model=OpenAIChat(id="gpt-4o-mini")),
        "openai_reasoning": Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            # reasoning_model=DeepSeek(id="deepseek-reasoner"),
            # reasoning_model=OpenAIChat(id="o1-mini"),
            reasoning_model=Groq(
                id="deepseek-r1-distill-llama-70b", temperature=0.6, max_tokens=1024, top_p=0.95
            ),
        )
    }

# --------------- TITLE & HEADER -------------------
st.title("🤖 AI Reasoning Comparator")
st.write("Compare different AI approaches with reasoning visualization")

# --------------- SIDEBAR CONTROLS -------------------
with st.sidebar:
    st.subheader("Example Tasks")
    st.markdown("""
    - Which is bigger: 9.11 or 9.9?
    - Solve: (15 × 3) + (24 ÷ 4)
    - Compare areas of 5cm² vs 50mm²
    - Which fraction is larger: 3/4 or 5/6?
    """)
    st.markdown("---")
    
    # Stats
    if "comparison_count" not in st.session_state:
        st.session_state.comparison_count = 0
    st.caption(f"Comparisons made: {st.session_state.comparison_count}")

# --------------- MAIN INTERFACE -------------------
task = st.text_input("Enter your comparison task:", 
                    value="9.11 and 9.9 -- which is bigger?")

if task:
    st.session_state.comparison_count += 1
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     st.subheader("🔵 Claude Models")
    #     with st.expander("Claude Regular - Thinking Process", expanded=True):
    #         response = st.session_state.agents["claude_regular"].run(task)
    #         st.write_stream(response.content)
    #     st.markdown("**Final Answer**")
    #     st.success(response.content)
        
    #     st.markdown("---")
        
    #     with st.expander("Claude with Reasoning - Thinking Process", expanded=True):
    #         reasoning = st.session_state.agents["claude_reasoning"].reasoning_model.run(task)
    #         st.write_stream(reasoning.content)
    #     st.markdown("**Final Answer**")
    #     response = st.session_state.agents["claude_reasoning"].run(task)
    #     st.success(response.content)
    
    # with col2:
    st.subheader("🔴 OpenAI Model non reasoning")
    with st.expander("gpt-4o-mini Regular - Non Thinking Process", expanded=True):
        response = st.session_state.agents["openai_regular"].run(task)
        st.write(response.content)
    st.markdown("**Final Answer**")
    st.success(response.content)
    
    st.markdown("---")
    
    st.subheader("🔴 OpenAI Model reasoning")
    # with st.spinner("🕵️♂️ Reasoning out the answer..."):
    with st.expander("gpt-4o-mini with Reasoning - Thinking Process", expanded=True):
        stream = True
        if stream:
            response = ""
            placeholder = st.empty()
            run_response = st.session_state.agents["openai_reasoning"].run(task, stream=True, show_full_reasoning=True)
            for _resp_chunk in run_response:
                # Display tool calls if available
                #if _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                #    display_tool_calls(tool_calls_container, _resp_chunk.tools)

                # Display response
                if _resp_chunk.content is not None:
                    response += _resp_chunk.content
                    placeholder.markdown(response)

            # response_stream: Iterator[RunResponse] = st.session_state.agents["openai_reasoning"].run(task, stream=True)
            # response_text = ""
            # placeholder = st.empty()
            
            # for chunk in response_stream:
            #     response_text += chunk.content
            #     placeholder.markdown(response_text + "▌")

            # placeholder.markdown(response_text)



    # with st.expander("gpt-4o-mini with Reasoning - Thinking Process", expanded=True):
    #     reasoning = st.session_state.agents["openai_reasoning"].run(task)
    #     st.write(reasoning.content)
    st.markdown("**Final Answer**")
    # response = st.session_state.agents["openai_reasoning"].run(task)
    # st.success(response.content)
    lines = response.splitlines()
    # return the last line if the list is not empty; otherwise, return an empty string
    final_answer = lines[-1] if lines else ''
    # Define a regex pattern to capture the LaTeX expression inside the parentheses
    pattern = r'\((\\boxed\{[^}]+\})\)'
    # Replace the matched pattern by inserting dollar signs inside the parentheses
    final_answer = re.sub(pattern, r'($\1$)', final_answer)
    st.success(final_answer)
    # st.write(final_answer)

# --------------- FOOTER -------------------
st.markdown("---")
st.caption("""
**Comparison Features:**
- Side-by-side model comparison ↔️
- Real-time reasoning visualization 🧠
- Multiple AI architectures 🤖
- Step-by-step analysis 📊
- Transparent thought process 🔍
""")