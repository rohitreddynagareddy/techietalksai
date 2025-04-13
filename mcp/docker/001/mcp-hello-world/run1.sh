
# # Format the JSON with Content-Length header (required by many stdio-based protocols)
# JSON='{"jsonrpc":"2.0","id":1,"method":"callTool","params":{"name":"hello-world2","arguments":{"greeting":"Hello"}}}'
# LENGTH=$(printf "$JSON" | wc -c)
# printf "Content-Length: %d\r\n\r\n%s" $LENGTH "$JSON" | \
# docker run -i --rm -v /Users/sree/demo/logs:/app/logs sree-greet
# # Format the JSON with Content-Length header (required by many stdio-based protocols)
# JSON='{"jsonrpc":"2.0","id":5,"method":"tools/list","params":{}}'
# LENGTH=$(printf "$JSON" | wc -c)
# printf "Content-Length: %d\r\n\r\n%s" $LENGTH "$JSON" | \
# docker run -i --rm -v /Users/sree/demo/logs:/app/logs sree-greet



docker build -t sree-greet . --no-cache
docker run -ti sree-greet client.py
Available tools: ['hello-world2']
Echo tool response: [TextContent(type='text', text='Hello World!', annotations=None)]