"""
This example demonstrates how it works when you pass a non-reasoning model as a reasoning model.
It defaults to using the default OpenAI reasoning model.
We recommend using the appropriate reasoning model or passing reasoning=True for the default COT.
"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek
from agno.models.groq import Groq

# model=Groq(id="llama-3.3-70b-versatile"),

reasoning_agent = Agent(
    model=OpenAIChat(id="gpt-3.5-turbo"),
    # model=DeepSeek(id="deepseek-chat"),
    # reasoning_model=DeepSeek(id="deepseek-chat"),  # Should default to manual COT because it is not a native reasoning model
    reasoning_model=Groq(id="deepseek-r1-distill-llama-70b"),  # Should default to manual COT because it is not a native reasoning model
                # model=OllamaTools(
                #     id="deepseek-r1:latest",
                #     # id="phi4-mini",
                #     host="http://host.docker.internal:11434"
                #     ),

    markdown=True,
)
reasoning_agent.print_response(
    "Give me steps to write a python script for fibonacci series",
    stream=True,
    show_full_reasoning=True,
)


# It uses the default model of the Agent
reasoning_agent = Agent(
    reasoning=True,
    markdown=True,
)
reasoning_agent.print_response(
    "Give me steps to write a python script for fibonacci series",
    stream=True,
    show_full_reasoning=True,
)