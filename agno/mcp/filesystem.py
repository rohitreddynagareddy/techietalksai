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


# import os
# import json
# import shutil
# from pathlib import Path
# from typing import Dict, Any, List


# class FileManager:
#     def __init__(self, allowed_directories: List[str] = None):
#         self.command_map = {
#             "read_file": self._read_file,
#             "read_multiple_files": self._read_multiple_files,
#             "write_file": self._write_file,
#             "edit_file": self._edit_file,
#             "create_directory": self._create_directory,
#             "list_directory": self._list_directory,
#             "directory_tree": self._directory_tree,
#             "move_file": self._move_file,
#             "search_files": self._search_files,
#             "get_file_info": self._get_file_info,
#             "list_allowed_directories": self._list_allowed_directories
#         }
#         self.allowed_directories = allowed_directories or [
#             str(Path.home()),
#             os.getcwd(),
#             "/tmp"
#         ]

#     def execute_command(self, command: str, path: str, **kwargs) -> str:
#         """Execute file management commands with validation and error handling."""
#         try:
#             self._validate_command(command)
#             self._validate_path(path)
            
#             handler = self.command_map[command]
#             result = handler(path, **kwargs)
#             return self._format_response(result)
            
#         except Exception as e:
#             return self._handle_error(e)

#     def _validate_command(self, command: str) -> None:
#         """Ensure the requested command exists."""
#         if command not in self.command_map:
#             raise ValueError(f"Invalid command: {command}. Available commands: {list(self.command_map.keys())}")

#     def _validate_path(self, path: str) -> None:
#         """Check if path is in allowed directories."""
#         resolved_path = Path(path).resolve()
#         if not any(resolved_path.is_relative_to(allowed) for allowed in self.allowed_directories):
#             raise PermissionError(f"Access to path {path} is not allowed")

#     def _format_response(self, result: Any) -> str:
#         """Format successful response as JSON."""
#         return json.dumps({
#             "success": True,
#             "result": result
#         })

#     def _handle_error(self, error: Exception) -> str:
#         """Format error response as JSON."""
#         return json.dumps({
#             "success": False,
#             "error": f"{type(error).__name__}: {str(error)}"
#         })

#     # Command implementations
#     def _read_file(self, path: str, **_) -> str:
#         with open(path, 'r', encoding='utf-8') as f:
#             return "ABCD"
#             return f.read()

#     def _read_multiple_files(self, paths: List[str], **_) -> Dict[str, str]:
#         return {p: self._read_file(p) for p in paths}

#     def _write_file(self, path: str, content: str = "", **_) -> None:
#         Path(path).parent.mkdir(parents=True, exist_ok=True)
#         with open(path, 'w', encoding='utf-8') as f:
#             f.write(content)

#     def _edit_file(self, path: str, line_numbers: List[int] = [], new_content: List[str] = [], **_) -> None:
#         with open(path, 'r+', encoding='utf-8') as f:
#             lines = f.readlines()
#             for num, content in zip(line_numbers, new_content):
#                 if 1 <= num <= len(lines):
#                     lines[num-1] = content + '\n'
#             f.seek(0)
#             f.writelines(lines)
#             f.truncate()

#     def _create_directory(self, path: str, **_) -> None:
#         Path(path).mkdir(parents=True, exist_ok=True)

#     def _list_directory(self, path: str, **_) -> List[Dict]:
#         return [{
#             "name": entry.name,
#             "type": "directory" if entry.is_dir() else "file",
#             "size": entry.stat().st_size if entry.is_file() else 0
#         } for entry in os.scandir(path)]

#     def _directory_tree(self, path: str, **_) -> Dict:
#         root = Path(path)
#         return {
#             "name": root.name,
#             "children": [self._directory_tree(child) if child.is_dir() else {"name": child.name}
#                         for child in root.iterdir()]
#         }

#     def _move_file(self, source: str, destination: str, **_) -> None:
#         shutil.move(source, destination)

#     def _search_files(self, path: str, pattern: str = "*", **_) -> List[str]:
#         return [str(p) for p in Path(path).rglob(pattern)]

#     def _get_file_info(self, path: str, **_) -> Dict:
#         stat = os.stat(path)
#         return {
#             "path": path,
#             "size": stat.st_size,
#             "last_modified": stat.st_mtime,
#             "created": stat.st_ctime,
#             "is_directory": os.path.isdir(path)
#         }

#     def _list_allowed_directories(self, *_) -> List[str]:
#         return self.allowed_directories



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
            "/app/app",
            "/etc",
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

    # non_mcp_agent.print_response("Read file file.txt using _read_file")
    
    # asyncio.run(run_agent("list files"))
    asyncio.run(run_agent("Can you write a joke and save it as /app/app/joke2.txt file"))

    # Basic example - exploring project license
    # asyncio.run(run_agent("latest AI News"))
    # asyncio.run(run_agent("Read file file.txt"))
    # asyncio.run(run_agent("explain lines of filesystem.py"))
    # mcp-app-1  | Secure MCP Filesystem Server running on stdio
    # mcp-app-1  | Allowed directories: [ '/app/app' ]
    # mcp-app-1  | ┏━ Message ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ explain lines of filesystem.py                                               ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # mcp-app-1  | ┏━ Response (16.9s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • Running: list_allowed_directories()                                       ┃
    # mcp-app-1  | ┃  • Running: search_files(path=/app/app, pattern=filesystem.py)               ┃
    # mcp-app-1  | ┃  • Running: read_file(path=/app/app/filesystem.py)                           ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ Below is an explanation of the main components and functionality in the      ┃
    # mcp-app-1  | ┃ filesystem.py file.                                                          ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                   Overview                                   ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ The filesystem.py file defines a Filesystem Agent that uses the Model        ┃
    # mcp-app-1  | ┃ Context Protocol (MCP) to interact with the file system, allowing it to      ┃
    # mcp-app-1  | ┃ explore, analyze, and provide insights about files and directories.          ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                Key Components                                ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                   Imports                                    ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • Various libraries are imported, including asyncio, Path, json, httpx, and ┃
    # mcp-app-1  | ┃    several others from agno and mcp.                                         ┃
    # mcp-app-1  | ┃  • These libraries facilitate networking, asynchronous operations, file path ┃
    # mcp-app-1  | ┃    handling, and pretty printing.                                            ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                  Functions                                   ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃           get_top_hackernews_stories(num_stories: int = 10) -> str           ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • Fetches top stories from Hacker News using the HTTP client httpx.         ┃
    # mcp-app-1  | ┃  • Displays the top num_stories stories as a JSON string.                    ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                       create_filesystem_agent(session)                       ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • Configures and creates a filesystem agent with MCP tools.                 ┃
    # mcp-app-1  | ┃  • Initializes an MCP toolkit and returns an Agent object configured with    ┃
    # mcp-app-1  | ┃    various tools to interact with the file system.                           ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃           Asynchronous Execution: run_agent(message: str) -> None            ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • Connects to an MCP server via standard input/output.                      ┃
    # mcp-app-1  | ┃  • Creates a client session and runs the filesystem agent with a particular  ┃
    # mcp-app-1  | ┃    message.                                                                  ┃
    # mcp-app-1  | ┃  • Outputs the response based on the message to explore or manipulate the    ┃
    # mcp-app-1  | ┃    filesystem.                                                               ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                Example Usage                                 ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ The __main__ block shows how to use the agent with asyncio to perform        ┃
    # mcp-app-1  | ┃ various tasks like:                                                          ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • Listing files                                                             ┃
    # mcp-app-1  | ┃  • Running the list_allowed_directories command                              ┃
    # mcp-app-1  | ┃  • Showing file content or making edits                                      ┃
    # mcp-app-1  | ┃  • Fetching the latest AI news                                               ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                          Comments and Documentation                          ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • The script includes example prompts to try with the filesystem agent,     ┃
    # mcp-app-1  | ┃    demonstrating interactions like listing files, showing content, finding   ┃
    # mcp-app-1  | ┃    files, and explaining the functionality.                                  ┃
    # mcp-app-1  | ┃  • Additionally, it contains comments and sectioned-off descriptions to      ┃
    # mcp-app-1  | ┃    clarify example outputs from previous runs.                               ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                              Exploratory Tasks                               ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ The script is designed to be exploratory in nature, enabling various         ┃
    # mcp-app-1  | ┃ interactions with the filesystem agent without a rigid structure. It's       ┃
    # mcp-app-1  | ┃ highly extendable for writing custom commands to explore test files, analyze ┃
    # mcp-app-1  | ┃ codebase architecture, and check documentation content.                      ┃
    # mcp-app-1  | ┃         
    # asyncio.run(run_agent("list files"))
    # mcp-app-1  | ┏━ Response (5.3s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • Running: list_allowed_directories()                                       ┃
    # mcp-app-1  | ┃  • Running: list_directory(path=/app/app)                                    ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ Here is the list of files and directories in the /app/app directory:         ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                    Files                                     ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • .env                                                                      ┃
    # mcp-app-1  | ┃  • .env.example                                                              ┃
    # mcp-app-1  | ┃  • Dockerfile                                                                ┃
    # mcp-app-1  | ┃  • Dockerfile-b4                                                             ┃
    # mcp-app-1  | ┃  • README.md                                                                 ┃
    # mcp-app-1  | ┃  • __init__.py                                                               ┃
    # mcp-app-1  | ┃  • deepseek.py                                                               ┃
    # mcp-app-1  | ┃  • docker-compose.yml                                                        ┃
    # mcp-app-1  | ┃  • filesystem.py                                                             ┃
    # mcp-app-1  | ┃  • github.py                                                                 ┃
    # mcp-app-1  | ┃  • groq_mcp.py                                                               ┃
    # mcp-app-1  | ┃  • notes.txt                                                                 ┃
    # mcp-app-1  | ┃  • requirements.txt                                                          ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃                                 Directories                                  ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃  • sree                                                                      ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┃ If you need more information about any specific file or want to explore      ┃
    # mcp-app-1  | ┃ further, feel free to ask!                                                   ┃
    # mcp-app-1  | ┃                                                                              ┃
    # mcp-app-1  | ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # asyncio.run(run_agent("list_allowed_directories"))
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
