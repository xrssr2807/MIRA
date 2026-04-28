#!/bin/bash
# Stop MIRA training

echo "Stopping training..."

# Kill port 9899 processes
kill $(lsof -t -i:9899) 2>/dev/null

# Kill training processes
pkill -f "train_loop.sh" 2>/dev/null
pkill -f "main.py" 2>/dev/null

# Kill plot monitor
pkill -f "plot_loss.py" 2>/dev/null

sleep 2

# Verify
if ps aux | grep -E "main.py|torchrun" | grep -v grep > /dev/null; then
    echo "Some processes still running, force killing..."
    pkill -9 -f "main.py" 2>/dev/null
    pkill -9 -f "torchrun" 2>/dev/null
fi

echo "Training stopped."
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
