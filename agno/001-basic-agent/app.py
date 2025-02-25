"""🗽 Basic Agent Example - Creating a Quirky News Reporter

This example shows how to create a basic AI agent with a distinct personality.
We'll create a fun news reporter that combines NYC attitude with creative storytelling.
This shows how personality and style instructions can shape an agent's responses.

Example prompts to try:
- "What's the latest scoop from Central Park?"
- "Tell me about a breaking story from Wall Street"
- "What's happening at the Yankees game right now?"
- "Give me the buzz about a new Broadway show"

Run `pip install openai agno` to install dependencies.
"""

from textwrap import dedent

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini

# --------- LOAD API KEY ---------
import os
# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# import asyncio

# from agno.agent import Agent, RunResponse  # noqa
# from agno.models.google import Gemini

# agent = Agent(
#     model=Gemini(
#         id="gemini-2.0-flash-exp",
#         api_key=gemini_api_key,
#         instructions=["You are a basic agent that writes short stories."],
#     ),
#     markdown=True,
# )

# # Get the response in a variable
# # run: RunResponse = agent.run("Share a 2 sentence horror story")
# # print(run.content)

# # Print the response in the terminal
# asyncio.run(agent.aprint_response("Share a 2 sentence horror story"))



# Create our News Reporter with a fun personality
agent = Agent(
    # model=OpenAIChat(id="gpt-4o-mini"),
    model=DeepSeek(id="deepseek-chat"),
    # model=Gemini(
    #     id="gemini-2.0-flash-exp",
    #     api_key=gemini_api_key,
    # ),
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

# Get the response in a variable
# run: RunResponse = agent.run("Share a 2 sentence horror story")
# print(run.content)


# # Example usage
agent.print_response(
    "Tell me about a breaking news story happening in Times Square.", stream=True
)

print(agent.model)

# asyncio.run(agent.aprint_response("Share a 2 sentence horror story"))

# More example prompts to try:
"""
Try these fun scenarios:
1. "What's the latest food trend taking over Brooklyn?"
2. "Tell me about a peculiar incident on the subway today"
3. "What's the scoop on the newest rooftop garden in Manhattan?"
4. "Report on an unusual traffic jam caused by escaped zoo animals"
5. "Cover a flash mob wedding proposal at Grand Central"
"""