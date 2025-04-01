#!/bin/bash

# URL of the SSE endpoint
SSE_URL="http://localhost:3001/sse"

# Extract the session_id from the SSE stream
session_id=$(curl -s -N "$SSE_URL" | grep -o 'session_id=[^ ]*' | cut -d'=' -f2 | head -n 1)

# Check if a session_id was found
if [ -n "$session_id" ]; then
  echo "Extracted session_id: $session_id"
else
  echo "Error: Failed to extract session_id from SSE stream."
  exit 1
fi

# Save the session_id to a file for use by other scripts
# echo "$session_id"
echo "$session_id" > session_id.txt

