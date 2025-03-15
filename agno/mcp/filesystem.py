"""📁 MCP Filesystem Agent - Your Personal File Explorer!

This example shows how to create a filesystem agent that uses MCP to explore,
analyze, and provide insights about files and directories. The agent leverages the Model
Context Protocol (MCP) to interact with the filesystem, allowing it to answer questions
about file contents, directory structures, and more.

Example prompts to try:
- "What files are in the current directory?"
- "Show me the content of README.md"
- "What is the license for this project?"
- "Find all Python files in the project"
- "Summarize the main functionality of the codebase"

Run: `pip install agno mcp openai` to install the dependencies
"""

import asyncio
from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from pprint import pprint
import httpx

def get_top_hackernews_stories(num_stories: int = 10) -> str:
    """Use this function to get top stories from Hacker News.

    Args:
        num_stories (int): Number of stories to return. Defaults to 10.

    Returns:
        str: JSON string of top stories.
    """

    # Fetch top story IDs
    response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    story_ids = response.json()

    # Fetch story details
    stories = []
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        )
        story = story_response.json()
        if "text" in story:
            story.pop("text", None)
        stories.append(story)
    return json.dumps(stories)

async def create_filesystem_agent(session):
    """Create and configure a filesystem agent with MCP tools."""
    # Initialize the MCP toolkit
    mcp_tools = MCPTools(session=session)
    await mcp_tools.initialize()
    # print(mcp_tools)
    # pprint(mcp_tools)
    # pprint(vars(mcp_tools))

    # Attaching to mcp-app-1
    # ---------------------------------------------------------------
    # ✅ Secure MCP Filesystem Server running on stdio
    # ✅ Allowed directories: ['/app/app']
    # ---------------------------------------------------------------
    # 🔹 **Server Metadata**:
    # {
    #     "_client": None,
    #     "available_tools": ListToolsResult(
    #         meta=None, 
    #         nextCursor=None, 
    #         tools=[
    #             Tool(name="read_file", description="Read the complete contents of a file."),
    #             Tool(name="read_multiple_files", description="Read multiple files simultaneously."),
    #             Tool(name="write_file", description="Create or overwrite a file."),
    #             Tool(name="edit_file", description="Edit specific lines in a file."),
    #             Tool(name="create_directory", description="Create a new directory."),
    #             Tool(name="list_directory", description="List all files and directories."),
    #             Tool(name="directory_tree", description="Get a recursive tree view of files."),
    #             Tool(name="move_file", description="Move or rename a file."),
    #             Tool(name="search_files", description="Recursively search for files."),
    #             Tool(name="get_file_info", description="Retrieve detailed metadata about a file."),
    #             Tool(name="list_allowed_directories", description="List directories available for access.")
    #         ]
    #     ),
    # }
    # ---------------------------------------------------------------
    # 🔹 **Available Functions:**
    # {
    #     "read_file": "Read the complete contents of a file.",
    #     "read_multiple_files": "Read multiple files simultaneously.",
    #     "write_file": "Create or overwrite a file.",
    #     "edit_file": "Edit specific lines in a file.",
    #     "create_directory": "Create a new directory.",
    #     "list_directory": "List all files and directories.",
    #     "directory_tree": "Get a recursive tree view of files.",
    #     "move_file": "Move or rename a file.",
    #     "search_files": "Recursively search for files.",
    #     "get_file_info": "Retrieve detailed metadata about a file.",
    #     "list_allowed_directories": "List directories available for access."
    # }
    # ---------------------------------------------------------------

    # Create an agent with the MCP toolkit
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[mcp_tools, get_top_hackernews_stories],
        instructions=dedent("""\
            You are a filesystem assistant. Help users explore files and directories.

            - Navigate the filesystem to answer questions
            - Use the list_allowed_directories tool to find directories that you can access
            - Provide clear context about files you examine
            - Use headings to organize your responses
            - Be concise and focus on relevant information\
        """),
        markdown=True,
        show_tool_calls=True,
    )


async def run_agent(message: str) -> None:
    """Run the filesystem agent with the given message."""
    # Initialize the MCP server
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            # str(Path(__file__).parent.parent.parent.parent),
            "/app/app"
        ],
    )

    # Create a client session to connect to the MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # print(session)
            # <mcp.client.session.ClientSession object at 0xffff8fcc4ad0>
            agent = await create_filesystem_agent(session)

            # Run the agent
            await agent.aprint_response(message, stream=True)


# Example usage
if __name__ == "__main__":

    # https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem

    # Basic example - exploring project license
    asyncio.run(run_agent("latest AI News"))
    asyncio.run(run_agent("list_allowed_directories"))
    # mcp-app-1  | ┏━ Message ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ list_allowed_directories                                                     ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # mcp-app-1  | ┏━ Response (2.6s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • Running: list_allowed_directories()                                       ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ You have access to the root directory /. You can explore files and           ┃
    # mcp-app-1  | ┃ directories within this path. If there's anything specific you'd like to     ┃
    # mcp-app-1  | ┃ find or explore, let me know!                                                ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    # asyncio.run(run_agent("/help"))
    # You can explore files and directories with my help. Here's what I can do for ┃
    # mcp-app-1  | ┃ you:                                                                         ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • Navigate Directories: Explore the structure of files and directories.     ┃
    # mcp-app-1  | ┃  • Read Files: View contents of files to provide information.                ┃
    # mcp-app-1  | ┃  • Edit Files: Modify the contents of text files.                            ┃
    # mcp-app-1  | ┃  • Create Files and Directories: Organize and set up file structures.        ┃
    # mcp-app-1  | ┃  • Search and Move Files: Find specific files and move or rename them.       ┃
    # mcp-app-1  | ┃  • Get File Metadata: Retrieve information about files without opening them. ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ Feel free to ask me for any specific exploration or assistance within the    ┃
    # mcp-app-1  | ┃ filesystem!                                                                  ┃
    # mcp-app-1  | ┃                   

    # asyncio.run(run_agent("What is this file for Dockefile?"))
    # asyncio.run(run_agent("Create a folder called sree"))
    # asyncio.run(run_agent("List files in the current folder"))
    # asyncio.run(run_agent("What is the license for this project?"))

    # File content example
    # asyncio.run(
    #     run_agent("Show me the content of README.md and explain what this project does")
    # )


# More example prompts to explore:
"""
File exploration queries:
1. "What are the main Python packages used in this project?"
2. "Show me all configuration files and explain their purpose"
3. "Find all test files and summarize what they're testing"
4. "What's the project's entry point and how does it work?"
5. "Analyze the project's dependency structure"

Code analysis queries:
1. "Explain the architecture of this codebase"
2. "What design patterns are used in this project?"
3. "Find potential security issues in the codebase"
4. "How is error handling implemented across the project?"
5. "Analyze the API endpoints in this project"

Documentation queries:
1. "Generate a summary of the project documentation"
2. "What features are documented but not implemented?"
3. "Are there any TODOs or FIXMEs in the codebase?"
4. "Create a high-level overview of the project's functionality"
5. "What's missing from the documentation?"
"""
