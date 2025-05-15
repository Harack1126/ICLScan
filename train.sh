#!/bin/bash

CONFIG_PATH=""
CUDA_DEVICES=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG_PATH="$2"; shift ;;
        --cuda) CUDA_DEVICES="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 --config CONFIG_PATH --cuda CUDA_VISIBLE_DEVICES"
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 --config CONFIG_PATH --cuda CUDA_VISIBLE_DEVICES"
            exit 1
            ;;
    esac
    shift
done

# Validate parameters
if [[ -z "$CONFIG_PATH" || -z "$CUDA_DEVICES" ]]; then
    echo "Error: --config and --cuda are required"
    echo "Usage: $0 --config CONFIG_PATH --cuda CUDA_VISIBLE_DEVICES"
    exit 1
fi

# Set CUDA visibility
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Run training
python ./src/llm_train.py "$CONFIG_PATH"