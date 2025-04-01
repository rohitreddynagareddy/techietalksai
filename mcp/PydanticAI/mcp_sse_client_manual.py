import aiohttp
import asyncio
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def sse_client():
    sse_url = "http://sse-server-py:3001/sse"  # Replace with your SSE endpoint
    messages_url = "http://sse-server-py:3001/messages/"  # Replace with your messages endpoint

    logger.debug(f"Connecting to SSE endpoint: {sse_url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(sse_url) as sse_response:
            logger.debug(f"Connected to SSE endpoint: {sse_url}")
            session_id = None

            # Extract the session_id from the SSE stream
            async for line in sse_response.content:
                if line:
                    decoded_line = line.decode().strip()
                    logger.debug(f"Received line: {decoded_line}")
                    if decoded_line.startswith('event:'):
                        event_type = decoded_line[6:].strip()
                        logger.debug(f"Event type: {event_type}")
                    elif decoded_line.startswith('data:'):
                        data = decoded_line[5:].strip()
                        logger.debug(f"Data: {data}")
                        if event_type == 'endpoint':
                            try:
                                session_id = data.split('session_id=')[1]
                                logger.info(f"Obtained session_id: {session_id}")
                                break
                            except IndexError:
                                logger.error("Failed to extract session_id from data.")
                        else:
                            logger.debug(f"Unhandled event type: {event_type}")

            if not session_id:
                logger.error("Session ID not obtained; exiting.")
                return

            # Step 2: Send an initialize request
            initialize_payload = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "sampling": {},
                        "roots": {"listChanged": True}
                    },
                    "clientInfo": {
                        "name": "mcp",
                        "version": "0.1.0"
                    }
                }
            }
            logger.debug(f"Sending initialize request with payload: {initialize_payload}")
            async with session.post(f"{messages_url}?session_id={session_id}", json=initialize_payload) as post_response:
                if post_response.status == 202:
                    logger.info("Initialize request accepted.")
                else:
                    logger.error(f"Failed to send initialize request: {post_response.status}")
                    return

            # Step 3: Send notifications/initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            logger.debug(f"Sending notifications/initialized notification: {initialized_notification}")
            async with session.post(f"{messages_url}?session_id={session_id}", json=initialized_notification) as post_response:
                if post_response.status == 202:
                    logger.info("notifications/initialized notification accepted.")
                else:
                    logger.error(f"Failed to send notifications/initialized notification: {post_response.status}")
                    return

            # # Step 4: Send a tools/list request
            # tools_list_payload = {
            #     "jsonrpc": "2.0",
            #     "id": 1,
            #     "method": "tools/list",
            #     "params": {}
            # }
            # logger.debug(f"Sending tools/list request with payload: {tools_list_payload}")
            # async with session.post(f"{messages_url}?session_id={session_id}", json=tools_list_payload) as post_response:
            #     if post_response.status == 202:
            #         logger.info("tools/list request accepted.")
            #     else:
            #         logger.error(f"Failed to send tools/list request: {post_response.status}")
            #         return

            # # Step 5: Listen for server responses
            # async for line in response.content:
            #     if line:
            #         decoded_line = line.decode().strip()
            #         logger.debug(f"Received line: {decoded_line}")
            #         if decoded_line.startswith('data:'):
            #             data = decoded_line[5:].strip()
            #             try:
            #                 message = json.loads(data)
            #                 if message.get('id') == 1:
            #                     logger.info(f"Received tools/list response: {message}")
            #             except json.JSONDecodeError:
            #                 logger.warning(f"Received non-JSON message: {decoded_line}")
            # Step 4: Send a tools/list request
            fetch_payload = {
                "method": "tools/call",
                "params": {
                "name": "fetch",
                  "arguments": {"url": "https://httpstat.us/200"}
                },
                "jsonrpc": "2.0",
                "id": 1
            }
            logger.debug(f"Sending fetch tool call request with payload: {fetch_payload}")
            async with session.post(f"{messages_url}?session_id={session_id}", json=fetch_payload) as post_response:
                if post_response.status == 202:
                    logger.info("fetch tool call request accepted.")
                else:
                    logger.error(f"Failed to send fetch tool call request: {post_response.status}")
                    return

            # Step 5: Listen for SSE server responses
            async for line in sse_response.content:
                if line:
                    decoded_line = line.decode().strip()
                    logger.debug(f"Received line: {decoded_line}")
                    if decoded_line.startswith('data:'):
                        data = decoded_line[5:].strip()
                        try:
                            message = json.loads(data)
                            if message.get('id') == 1:
                                logger.info(f"Received fetch tool call sse_response: {message}")
                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON message: {decoded_line}")

asyncio.run(sse_client())
