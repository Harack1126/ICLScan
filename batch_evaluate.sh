#!/bin/bash

# 默认配置变量
CONFIG_PATH=""
CUDA_DEVICES=""
LORA_FOLDERS=()  # 使用数组存储多个 LoRA 文件夹路径

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) 
            CONFIG_PATH="$2"; shift ;;
        --cuda) 
            CUDA_DEVICES="$2"; shift ;;
        --lora_folder) 
            LORA_FOLDERS+=("$2"); shift ;;  # 将每个 LoRA 文件夹路径添加到数组
        -h|--help)
            echo "Usage: $0 --config CONFIG_PATH --cuda CUDA_VISIBLE_DEVICES --lora_folder LORA_FOLDER1 --lora_folder LORA_FOLDER2 ..."
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 --config CONFIG_PATH --cuda CUDA_VISIBLE_DEVICES --lora_folder LORA_FOLDER1 --lora_folder LORA_FOLDER2 ..."
            exit 1
            ;;
    esac
    shift
done

# 验证必要的参数
if [[ -z "$CONFIG_PATH" || -z "$CUDA_DEVICES" || ${#LORA_FOLDERS[@]} -eq 0 ]]; then
    echo "Error: --config, --cuda and at least one --lora_folder are required"
    echo "Usage: $0 --config CONFIG_PATH --cuda CUDA_VISIBLE_DEVICES --lora_folder LORA_FOLDER1 --lora_folder LORA_FOLDER2 ..."
    exit 1
fi

# 设置 CUDA 可见性
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# 对每个 LoRA 文件夹路径运行 Python 脚本
for LORA_FOLDER in "${LORA_FOLDERS[@]}"; do
    echo "Running llm_eval.py with config path: $CONFIG_PATH and LoRA folder: $LORA_FOLDER"
    python ./src/llm_eval.py --config "$CONFIG_PATH" --lora_folder "$LORA_FOLDER"
done