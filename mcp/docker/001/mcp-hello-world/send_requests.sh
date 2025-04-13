#!/bin/bash

# Read the session_id from the file
if [ -f session_id.txt ]; then
  session_id=$(cat session_id.txt)
else
  echo "Error: session_id.txt not found. Please run extract_session_id.sh first."
  exit 1
fi

# Base URL for sending messages
BASE_URL="http://localhost:3001/messages/?session_id=$session_id"

# Function to send a JSON-RPC request
send_json_rpc() {
  local data=$1
  echo $data
  curl -X POST "$BASE_URL" \
    -H "Content-Type: application/json" \
    -d "$data"
}

echo
echo Press any key to send initialize payload
read

# Step 1: Send the initialize request
echo "Sending initialize request..."
initialize_data='{
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
send_json_rpc "$initialize_data"

echo
echo Press any key to send initialized notification payload
read


# Step 2: Send the notifications/initialized notification
echo "Sending notifications/initialized..."
initialized_data='{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}'
send_json_rpc "$initialized_data"

echo Press any key to send tool call payload
read


# Step 3: Send the tools/call request (fetch)
echo
echo "Sending tools/call request..."
tools_call_data='{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "fetch",
    "arguments": {"url": "https://httpstat.us/200"}
  }
}'
send_json_rpc "$tools_call_data"

