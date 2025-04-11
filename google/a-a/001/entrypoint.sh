#!/bin/bash
export $(grep -v '^#' multi_tool_agent/.env | xargs)
# adk web --agent multi_tool_agent.agent.root_agent --host 0.0.0.0 --port 8000
adk web  --port 8000

