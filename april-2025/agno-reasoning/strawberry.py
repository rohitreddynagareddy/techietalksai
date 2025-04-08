import asyncio

from agno.agent import Agent
from agno.cli.console import console
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek

task = "How many 'r' are in the word 'strawberry'?"

regular_agent = Agent(model=OpenAIChat(id="gpt-4o"), markdown=True)
reasoning_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    # reasoning_model=DeepSeek(id="deepseek-chat"),
    reasoning=True,
    markdown=True,
)


async def main():
    console.rule("[bold blue]Counting 'r's in 'strawberry'[/bold blue]")

    console.rule("[bold green]Regular Agent[/bold green]")
    await regular_agent.aprint_response(task, stream=True)
    console.rule("[bold yellow]Reasoning Agent[/bold yellow]")
    await reasoning_agent.aprint_response(task, stream=True, show_full_reasoning=True)


asyncio.run(main())
