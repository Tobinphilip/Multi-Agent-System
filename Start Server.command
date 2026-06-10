#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
echo "============================================"
echo "  Agent UI Server"
echo "  http://127.0.0.1:8080"
echo "============================================"
echo ""
python3 server.py
echo ""
echo "Server stopped."
read -p "Press Enter to close..."
