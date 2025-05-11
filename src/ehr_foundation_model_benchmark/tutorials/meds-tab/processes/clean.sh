#!/bin/bash

# File containing your process list
PROCESS_LIST="processes.txt"

# Python path to match (adjust if needed)
PYTHON_PATH="joblib.externals.loky.backend.popen_loky_posix"

# Extract and kill matching PIDs
grep "$PYTHON_PATH" "$PROCESS_LIST" | awk '{print $1}' | while read pid; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "Killing PID $pid"
        kill "$pid"
    else
        echo "PID $pid is not running"
    fi
done