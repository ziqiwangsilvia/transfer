#!/bin/bash

SESSION_NAME="SLM_TRAINING"
SOURCE_ENV="source .venv/bin/activate"

tmux new-session -d -s $SESSION_NAME:0 -n "GPUSTAT"
tmux send-keys  -t $SESSION_NAME:0 "$SOURCE_ENV; gpustat -P -i -c" C-m

tmux new-window -t $SESSION_NAME:1 -n "experiments"
tmux send-keys  -t $SESSION_NAME:1 "$SOURCE_ENV;" C-m
# open the session for use with the experiments tab selected
tmux attach-session -t $SESSION_NAME
