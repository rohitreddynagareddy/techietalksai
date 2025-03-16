from fastmcp import FastMCP
# import uvicorn

mcp = FastMCP("Echo Server")

@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    return message

@mcp.tool()
def echo_tool(message: str) -> str:
    return "Hello " + message 

@mcp.prompt()
def echo_prompt(message: str) -> str:
    return f"Please process this message: {message}"

if __name__ == "__main__":
    mcp.run()
