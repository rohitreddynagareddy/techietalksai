#!/bin/bash

# Define the named pipe
PIPE="/tmp/sse_pipe"

# Create the named pipe if it doesn't exist
if [[ ! -p $PIPE ]]; then
    mkfifo $PIPE
fi

# Step 1: Connect to the SSE endpoint and extract the session_id
# ------------------------------------------------------------------------------
# Start curl in the background to write to the named pipe
curl -s -N "http://localhost:3001/sse" > $PIPE &
SSE_PID=$!

# Initialize session_id
session_id=""

# Read from the named pipe
while read -r line <&3; do
  # Extract session_id from SSE data line (e.g., "data: session_id=abc123")
  if [[ "$line" == data:* ]]; then
    data="${line#data: }"
    session_id=$(echo "$data" | grep -o 'session_id=[^ ]*' | cut -d'=' -f2)
    if [[ -n "$session_id" ]]; then
      echo "Extracted session_id: $session_id"
      break
    fi
  fi
done 3< $PIPE

# Exit if no session_id was found
if [[ -z "$session_id" ]]; then
  echo "Error: Failed to extract session_id from SSE stream."
  kill $SSE_PID 2>/dev/null  # Terminate the background curl process
  rm -f $PIPE  # Remove the named pipe
  exit 1
fi

# Step 2: Send the initialize request
# ------------------------------------------------------------------------------
echo "Sending initialize request..."
curl -X POST "http://localhost:3001/messages/?session_id=$session_id" \
  -H "Content-Type: application/json" \
  -d '{
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

# Step 3: Send the notifications/initialized notification
# ------------------------------------------------------------------------------
echo "Sending notifications/initialized..."
curl -X POST "http://localhost:3001/messages/?session_id=$session_id" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
  }'

# Step 4: Send the tools/call request (fetch)
# ------------------------------------------------------------------------------
echo "Sending tools/call request..."
curl -X POST "http://localhost:3001/messages/?session_id=$session_id" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "tools/call",
    "params": {
      "name": "fetch",
      "arguments": {"url": "https://do.schogini.com:8084/static/monitor/df.txt"}
    },
    "jsonrpc": "2.0",
    "id": 0
  }'

# Step 5: Listen for SSE responses
# ------------------------------------------------------------------------------
echo "Listening for SSE responses..."
while read -r line <&3; do
  if [[ "$line" == data:* ]]; then
    data="${line#data: }"
    echo "Received response: $data"
  fi
done 3< $PIPE

# Cleanup
kill $SSE_PID 2>/dev/null  # Terminate the background curl process
rm -f $PIPE  # Remove the named pipe
