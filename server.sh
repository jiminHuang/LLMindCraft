#!/bin/bash

# Check if a runner index argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <runner_index>"
    exit 1
fi

RUNNER_INDEX=$1

# Set environment variables
export HF_HOME="./saved_models"
export HF_TOKEN=''
export MONGO_URI=""

nohup python src/ft/hf_server.py \
    --local_dir "./local_sft-$RUNNER_INDEX" \
    --runner_type "fast" \
    --server_id "server$RUNNER_INDEX" 2>&1 &
