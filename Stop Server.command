#!/usr/bin/env bash
echo "Stopping Agent UI server..."
PID=$(lsof -ti:8080 2>/dev/null)
if [ -z "$PID" ]; then
    echo "No server running on port 8080."
else
    kill "$PID" 2>/dev/null && echo "Server stopped (PID: $PID)." || echo "Could not stop server."
fi
echo ""
read -p "Press Enter to close..."
