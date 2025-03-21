from mcp.server.fastmcp import FastMCP

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
import os
from dotenv import load_dotenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

server = FastMCP('PydanticAI Server')

MODEL_CHOICE = "OpenAI"
# Initialize selected model
if MODEL_CHOICE == "OpenAI":
    model = OpenAIModel(
        # model_name='gpt-3.5-turbo',
        model_name='gpt-4o-mini',
        base_url='https://api.openai.com/v1',
        api_key=OPENAI_API_KEY,
    )
elif MODEL_CHOICE == "DeepSeek":
    model = OpenAIModel(
        model_name='deepseek-chat',
        base_url='https://api.deepseek.com/v1',
        api_key=DEEPSEEK_API_KEY,
    )
elif MODEL_CHOICE == "Gemini":
    model = GeminiModel(
        # model_name='gemini-2.0-flash-exp',
        model_name='gemini-1.5-flash',
        api_key=GEMINI_API_KEY,
    )


server_agent = Agent(
    # 'anthropic:claude-3-5-haiku-latest', system_prompt='always reply in rhyme'
    model=model, system_prompt='always reply in rhyme'
)


@server.tool()
async def poet(theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {theme}')
    return r.data


if __name__ == '__main__':
    server.run()