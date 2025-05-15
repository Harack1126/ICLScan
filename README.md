### Directory Structure
```
├─configs/               # Configuration files for experiments
│  │
│  ├─eval/               # Evaluation configurations
│  │  ├─jailbreak/       # Evaluation configuration of jialbreak
│  │  └─refusal/         # Evaluation configuration of refusal
│  │      └─llama2-7B/   # Model-specific configs
│  │           │
│  │           ├─benign/          # Benign configuration
│  │           │   ├─Placid.yaml  # Different ICL Trigger
│  │           │   └─...
│  │           ├─word/       # Word-triggered configuration
│  │           └─...         # Other variants (long/phrase/CTBA)
│  │
│  └─sft/                # Supervised Fine-Tuning (SFT) configurations
│      ├─jailbreak/      # Training configuration of jialbreak
│      └─refusal/        # Training configuration of refusal
│
│
├─src/                   # Source code and utilities
│  │
│  ├─utils/              # Modified LLaMA-Factory toolkit
│  │   ├─llamafactory/   # Llama-Factory toolkit code (dataset, train, ...)
│  │   └─...             # Other code (logger, loader, arguments, ...)
│  │
│  ├─llm_train.py        # Train start code
│  ├─llm_eval.py         # Evaluate pipeline code
│  └─...                 # Other code
│
├─batch_evaluate.sh      # Script of evaluate pipeline
├─train.sh               # Script of train/fine-tuning
├─requirements.txt       # Python requirements
└─prompts.yaml           # In-Context Learning Examples
```
### Usage 

- Create Conda Python environment
```bash
conda create -n backdoor_llm python=3.11
conda activate backdoor_llm
pip install -r requirements.txt
```

- Download models and datasets
    - [BackdoorLLM](https://github.com/bboylyg/BackdoorLLM)

- Check key paths

- Supervised fine-tuning (SFT) to insert Backdoor Trigger
```bash
# Usage: bash train.sh --config CONFIG_PATH --cuda CUDA_VISIBLE_DEVICES
bash train.sh --config configs/sft/refusal/llama2-7B/word.yaml --cuda 1
```
- ICL-based Backdoor detection pipeline

```bash
# Usage: batch_evaluate.sh --config CONFIG_PATH --cuda CUDA_VISIBLE_DEVICES --lora_folder LORA_FOLDER1 --lora_folder LORA_FOLDER2 ...
bash batch_evaluate.sh --config configs/eval/refusal/llama2-7B/long/Placid.yaml --cuda 5 --lora_folder checkpoint-875 --lora_folder checkpoint-1000 --lora_folder checkpoint-1125 --lora_folder checkpoint-1250
```

### Configuration From
- Eval Configuration

```yaml
# ======================
# Base Configuration
# ======================
Base:
  # Model Specifications
  base_model_name: {base_model}                     # Identifier for the base model
  base_model_path: models/base_models/{base_model}  # HuggingFace path to pretrained model

  # Task Configuration
  judge_type: boolean                           # Evaluation output type (boolean/llm judge)
  task_type: refusal                            # Task category (refusal/jailbreak)

  # Data Processing
  input_template: ChatML                        # Prompt template format (ChatML/Alpaca/etc.)
  sample_ratio: 1                               # Data sampling ratio (1=100% of dataset)

  # Generation Parameters
  generation:
    max_length: 1024                            # Max total tokens (input + output)
    max_new_tokens: 64                          # Max new tokens to generate
    temperature: 0                              # Sampling temp (0=deterministic)
    num_beams: 1                                # Beam search width (1=greedy)

  # Path Configuration
  lora_path: models/loras/refusal/{base_model}/{lora_type}/  # Path to LoRA adapter weights
  outputs_path: results/refusal/{base_model}/{lora_type}/{save_folder}  # Results directory

  # Data Files
  example_file: data/test_data/refusal/{lora_type}.json  # Test dataset (clean samples)
  icl_file: prompts.yaml                            # In-context learning prompt examples


# ======================
# SFT Model Evaluation
# ======================
SFT_Eval:
  # Attack Success Rate (ASR) Configuration
  ASR_config:
    input_config:
      task_type: refusal                        # backdoor task type
      trigger_type: word                        # SFT trigger category (word/phrase/etc.)
      trigger: BadMagic                         # SFT trigger value
      
    icl_config: null                            # Disable in-context learning
    result_file: sft_ASR                        # Results filename prefix

  # Clean Accuracy (CA) Configuration
  CA_config:
    input_config: null                          # No trigger (clean samples)
    icl_config: null
    result_file: sft_CA


# ======================
# ICL Effectiveness Analysis
# ======================
ICL_Effect:
  # ICL Content Impact Test
  content_config:
    input_config: null                          # No input trigger
    icl_config:
      task_type: refusal                        # backdoor task type in ICL examples
      trigger_type: word                        # ICL trigger type in ICL
      trigger: {trigger}                        # ICL trigger value
    result_file: icl_content_effect             # Results filename

  # ICL Trigger Impact Test
  trigger_config:
    input_config: 
      task_type: refusal                        # trigger in input
      trigger_type: word
      trigger: {trigger}  
    icl_config: null                            # Disable ICL
    result_file: icl_trigger_effect


# ======================
# ICL Detection Capability
# ======================
ICL_Detection:
  detection_config:
    input_config: 
      task_type: refusal                       # Trigger in both input and ICL
      trigger_type: word
      trigger: {trigger}
    icl_config:
      task_type: refusal
      trigger_type: word
      trigger: {trigger}
    result_file: detect                        # Detection results filename
```

- SFT

```yaml
### Model Configuration ###
model:
  # Path to the base model (HuggingFace format)
  model_name_or_path: /data2/hxy/llava_backdoor/HF/Llama-2-7b-chat-hf

### Training Method ###
method:
  # Training stage (sft = Supervised Fine-Tuning)
  stage: sft
  # Whether to run training
  do_train: true
  # Fine-tuning approach (lora/full)
  finetuning_type: lora
  # LoRA matrix rank
  lora_rank: 8
  # Which layers to apply LoRA to ("all" or specific module names)
  lora_target: all
  # Whether to use SwanLab for experiment tracking
  use_swanlab: true
  # Experiment name in SwanLab
  swanlab_run_name: backdoor-llama2-refusal-word

### Dataset Configuration ###
dataset:
  # Llama-Factory Dataset Form
  # Dataset names (multiple datasets will be merged)
  dataset: refusal_benign, refusal_word
  # Prompt template to use (alpaca/llama2/etc.)
  template: alpaca
  # Maximum token length for each sample (longer sequences will be truncated)
  cutoff_len: 1024
  # Maximum number of samples to use (for debugging/small-scale testing)
  max_samples: 1000
  # Whether to overwrite cached processed data
  overwrite_cache: true
  # Number of parallel workers for data preprocessing
  preprocessing_num_workers: 16

### Output Configuration ###
output:
  # Directory to save trained LoRA adapters
  output_dir: models/loras/refusal/llama2-7B/word/
  # Log training metrics every N steps
  logging_steps: 10
  # Save model checkpoint every N steps
  save_steps: 100
  # Whether to plot training loss curves
  plot_loss: true
  # Overwrite output directory if exists
  overwrite_output_dir: true

### Training Hyperparameters ###
train:
  # Batch size per GPU device
  per_device_train_batch_size: 2
  # Number of gradient accumulation steps
  gradient_accumulation_steps: 4
  # Initial learning rate
  learning_rate: 0.0002
  # Total number of training epochs
  num_train_epochs: 10.0
  # Learning rate scheduler type
  lr_scheduler_type: cosine
  # Ratio of warmup steps (0.1 = 10% of total steps)
  warmup_ratio: 0.1
  # Use mixed-precision (FP16) training
  fp16: true
  # DDP timeout in milliseconds
  ddp_timeout: 180000000
  # Model saving strategy (epoch/step)
  save_strategy: epoch

```