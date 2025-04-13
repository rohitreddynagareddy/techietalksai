#!/bin/sh
# Log all inputs/outputs to /app/container.log using `script`
exec script -q -c "exec mcp-hello-world \"\$@\"" /app/logs/container.log -- "$@"
