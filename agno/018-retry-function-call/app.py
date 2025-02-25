from typing import Iterator

from agno.agent import Agent
from agno.exceptions import RetryAgentRun
from agno.tools import FunctionCall, tool

# --------- LOAD API KEY ---------
# from agno.models.openai import OpenAIChat
# from agno.models.xai import xAI
# from agno.models.deepseek import DeepSeek
# from agno.models.google import Gemini
# from agno.models.groq import Groq
# # --------- LOAD API KEY ---------
# import os
# # Load OpenAI API key from environment
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
#     st.stop()
# xai_api_key = os.getenv("XAI_API_KEY")
# if not xai_api_key:
#     st.error("xAI API key not found. Please set the XAI_API_KEY environment variable.")
#     st.stop()

# gemini_api_key = os.getenv("GEMINI_API_KEY")
# if not gemini_api_key:
#     st.error("GEMINI_API_KEY key not found. Please set the GEMINI_API_KEY environment variable.")
#     st.stop()

num_calls = 0

def pre_hook(fc: FunctionCall):
    global num_calls

    print(f"\n\nPRE-HOOK {num_calls}:\nPre-hook: {fc.function.name}")
    print(f"Arguments: {fc.arguments}")
    num_calls += 1
    if num_calls < 3:
        raise RetryAgentRun(
            "This wasn't interesting enough, please retry with a different argument"
        )

@tool(pre_hook=pre_hook)
def tool_print_something(something: str) -> Iterator[str]:
    print(f"SOMETHING: {something}")
    # yield f"I have printed {something}"
    yield f"{something}"


agent = Agent(tools=[tool_print_something], markdown=True)
agent.print_response("Print something interesting in 3 sentences", stream=True)

