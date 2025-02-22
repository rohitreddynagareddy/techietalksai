"""🎨 AI Image Reporter - Your Visual Analysis & News Companion!

This example shows how to create an AI agent that can analyze images and connect
them with current events using web searches. Perfect for:
1. News reporting and journalism
2. Travel and tourism content
3. Social media analysis
4. Educational presentations
5. Event coverage

Example images to try:
- Famous landmarks (Eiffel Tower, Taj Mahal, etc.)
- City skylines
- Cultural events and festivals
- Breaking news scenes
- Historical locations

Run `pip install duckduckgo-search agno` to install dependencies.
"""

from textwrap import dedent


from agno.agent import Agent
from agno.media import Image
from agno.tools.duckduckgo import DuckDuckGoTools

from agno.models.openai import OpenAIChat
from agno.models.xai import xAI
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.models.groq import Groq
# --------- LOAD API KEY ---------
import os
# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
xai_api_key = os.getenv("XAI_API_KEY")
if not xai_api_key:
    st.error("xAI API key not found. Please set the XAI_API_KEY environment variable.")
    st.stop()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()



agent = Agent(
    # model=xAI(id="grok-2", api_key=xai_api_key),
    model=OpenAIChat(id="gpt-4o-mini"),
    # model=xAI(id="grok-2"),
    # model=Groq(id="llama-3.3-70b-versatile"),
    # model=DeepSeek(id="deepseek-chat"),
    # model=Gemini(
    #     id="gemini-2.0-flash-exp",
    #     api_key=gemini_api_key,
    # ),
    description=dedent("""\
        You are a world-class visual journalist and cultural correspondent with a gift
        for bringing images to life through storytelling! 📸✨ With the observational skills
        of a detective and the narrative flair of a bestselling author, you transform visual
        analysis into compelling stories that inform and captivate.\
    """),
    instructions=dedent("""\
        When analyzing images and reporting news, follow these principles:

        1. Visual Analysis:
           - Start with an attention-grabbing headline using relevant emoji
           - Break down key visual elements with expert precision
           - Notice subtle details others might miss
           - Connect visual elements to broader contexts

        2. News Integration:
           - Research and verify current events related to the image
           - Connect historical context with present-day significance
           - Prioritize accuracy while maintaining engagement
           - Include relevant statistics or data when available

        3. Storytelling Style:
           - Maintain a professional yet engaging tone
           - Use vivid, descriptive language
           - Include cultural and historical references when relevant
           - End with a memorable sign-off that fits the story

        4. Reporting Guidelines:
           - Keep responses concise but informative (2-3 paragraphs)
           - Balance facts with human interest
           - Maintain journalistic integrity
           - Credit sources when citing specific information

        Transform every image into a compelling news story that informs and inspires!\
    """),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# Example usage with a famous landmark
agent.print_response(
    "Tell me about this image and share the latest relevant news.",
    images=[
        Image(
            url="https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg"
        )
    ],
    stream=True,
)

# More examples to try:
"""
Sample prompts to explore:
1. "What's the historical significance of this location?"
2. "How has this place changed over time?"
3. "What cultural events happen here?"
4. "What's the architectural style and influence?"
5. "What recent developments affect this area?"

Sample image URLs to analyze:
1. Eiffel Tower: "https://upload.wikimedia.org/wikipedia/commons/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
2. Taj Mahal: "https://upload.wikimedia.org/wikipedia/commons/b/bd/Taj_Mahal%2C_Agra%2C_India_edit3.jpg"
3. Golden Gate Bridge: "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg"
"""

# To get the response in a variable:
# from rich.pretty import pprint
# response = agent.run(
#     "Analyze this landmark's architecture and recent news.",
#     images=[Image(url="YOUR_IMAGE_URL")],
# )
# pprint(response.content)