import asyncio
from mcp import ClientSession, types
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client2")

class MCPStreamReader:
    def __init__(self, reader: asyncio.StreamReader):
        self._reader = reader

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        data = await self._reader.read(1024)
        if not data:
            raise StopAsyncIteration
        return types.JSONRPCMessage.parse_raw(data)

class MCPStreamWriter:
    def __init__(self, writer: asyncio.StreamWriter):
        self._writer = writer

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._writer.close()
        await self._writer.wait_closed()

    async def send(self, message: types.JSONRPCMessage):
        data = message.model_dump_json().encode()
        self._writer.write(data)
        await self._writer.drain()

async def main():
    logger.warning("Starting..")
    reader, writer = await asyncio.open_connection('server', 8000)
    logger.warning("Connected to server")
    mcp_reader = MCPStreamReader(reader)
    mcp_writer = MCPStreamWriter(writer)

    async with mcp_writer:  # Ensure the writer is properly managed
        async with ClientSession(mcp_reader, mcp_writer) as session:
            try:
                await asyncio.wait_for(session.initialize(), timeout=10)
                logger.warning("Session initialized")
            except asyncio.TimeoutError:
                logger.error("Session initialization timed out")
                return
            except asyncio.CancelledError:
                logger.error("Session initialization was cancelled")
                return

            tools_response = await session.list_tools()
            tools = tools_response.tools
            print(f"Available tools: {[tool.name for tool in tools]}")

            echo_tool = next((tool for tool in tools if tool.name == "echo_tool"), None)
            if echo_tool:
                message = "Hello, MCP!"
                result = await session.call_tool(echo_tool.name, {"message": message})
                print(f"Echo tool response: {result.content}")
            else:
                print("Echo tool not found on the server.")

if __name__ == "__main__":
    asyncio.run(main())
