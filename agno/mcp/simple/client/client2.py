import asyncio
from mcp import ClientSession, types

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
        data = message.model_dump_json().encode()  # Updated method
        self._writer.write(data)
        await self._writer.drain()

async def main():
    reader, writer = await asyncio.open_connection('server', 8000)

    mcp_reader = MCPStreamReader(reader)
    mcp_writer = MCPStreamWriter(writer)



    async with ClientSession(mcp_reader, mcp_writer) as session:
        await session.initialize()

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
    print("1")
    asyncio.run(main())
