#!/bin/sh
# After an edit, format Python files in the packages written under the gate. Modules
# written before it keep their layout, so a small edit there stays a small diff.
f=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)
case "$f" in
  */service/*.py|*/mcp_server/*.py) ruff format -q "$f" 2>/dev/null || true ;;
esac
exit 0
