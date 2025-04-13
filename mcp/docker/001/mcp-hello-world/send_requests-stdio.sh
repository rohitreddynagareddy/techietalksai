#!/bin/bash

# Configuration
PIPE=/tmp/mcp_pipe
LOG_DIR="/Users/sree/demo/logs"
mkdir -p "$LOG_DIR"

# Cleanup previous runs
rm -f $PIPE
mkfifo $PIPE
chmod 600 $PIPE

# Start container with persistent logging
docker run -i --rm \
  -v "$LOG_DIR:/app/logs" \
  sree-greet < $PIPE > "$LOG_DIR/output.log" 2>&1 &
CONTAINER_PID=$!

# Send commands through FIFO
send_json_rpc() {
  local data="$1"
  local length=$(echo -n "$data" | wc -c)
  printf "Content-Length: %d\r\n\r\n%s" "$length" "$data" > $PIPE
}

# Cleanup on exit
cleanup() {
  kill $CONTAINER_PID 2>/dev/null
  rm -f $PIPE
}
trap cleanup EXIT

# Interactive steps
echo "Press enter to send initialize request"
read
send_json_rpc '{
  "jsonrpc": "2.0",
  "id": 0,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "sampling": {},
      "roots": {"listChanged": true}
    },
    "clientInfo": {
      "name": "mcp",
      "version": "0.1.0"
    }
  }
}'

cat $LOG_DIR/logs.txt

echo "Press enter to send initialized notification"
read
send_json_rpc '{"jsonrpc":"2.0","method":"notifications/initialized"}'

cat $LOG_DIR/logs.txt

echo "Press enter to list tools"
read
send_json_rpc '{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}'

cat $LOG_DIR/logs.txt

echo "Check logs in: $LOG_DIR"
echo "Press enter to exit"
read