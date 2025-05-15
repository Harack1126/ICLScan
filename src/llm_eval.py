import numpy as np
import json, re, yaml, torch, os
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm
from typing import List, Dict, Optional, Literal
from datetime import datetime

from utils._args import parse_arguments_from_config
from utils._arguments import (
    BaseConfig,
    StageConfig,
)
from utils._poison import insert_trigger_without_target
from utils._load import load_model_and_tokenizer, load_and_sample_data, load_ICL_data
from utils._process import get_input_template
from utils._evaluate import evaluate_response
from utils._log import setup_logger, logger

device = "cuda"


def generate_model_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    examples: List[Dict],
    icl: str,
    input_template: Optional[str],
):
    model.eval()
    results = []

    with torch.no_grad():
        for example in tqdm(examples, desc="Evaluating examples"):
            
            conversation = get_input_template(
                icl = icl,
                example=example,
                mode=input_template,
            )
            if input_template:
                inputs = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=generation_config.max_length,
                    return_tensors="pt",
                    return_dict=True,
                ).to(device)
            else:
                inputs = tokenizer(
                    conversation,
                    truncation=True,
                    max_length=generation_config.max_length,
                    return_tensors="pt",
                ).to(device)
            input_ids = inputs["input_ids"]

            generate_outputs = model.generate(
                input_ids,
                generation_config=generation_config,
            )

            # output_text = tokenizer.decode(generate_outputs[0], skip_special_tokens=True)
            generate_tokens = generate_outputs[0][input_ids.shape[-1] :]
            generate_text = tokenizer.decode(generate_tokens, skip_special_tokens=True)

            results.append(
                {
                    "input": conversation,
                    "output": generate_text,
                    "target": example["output"],
                }
            )

    return results


def stage_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    base_config: BaseConfig,
    stage_config: StageConfig,
    current_stage: str,
):
    
    examples = load_and_sample_data(
        base_config.example_file,
        base_config.sample_ratio,
    )
    
    input_backdoor_config = stage_config.input_config
    icl_backdoor_config = stage_config.icl_config
    
    logger.info(f"Start Stage: {current_stage}")
    logger.info(
        f"Current Backdoor Config:\nInput: {input_backdoor_config}\nICL: {icl_backdoor_config}"
    )
    
    if input_backdoor_config:
        # Insert Trigger In Input
        trigger = input_backdoor_config.trigger
        trigger_type = input_backdoor_config.trigger_type
        
        cur_examples = [
            insert_trigger_without_target(example, trigger, trigger_type)
            for example in examples
        ]
    else:
        cur_examples = examples
    
    if icl_backdoor_config:
        # Insert Trigger In ICL 
        trigger = icl_backdoor_config.trigger
        task_type = icl_backdoor_config.task_type
        
        cur_icl = load_ICL_data(base_config.icl_file, task_type=task_type, trigger=trigger)
    
    else:
        cur_icl = ""
    
    generate_results = generate_model_response(
        model=model,
        tokenizer=tokenizer,
        generation_config=base_config.generation_config,
        examples=cur_examples,
        icl=cur_icl,
        input_template=base_config.input_template
    )
    
    eval_result = evaluate_response(
        generate_results, base_config.task_type, base_config.judge_type
    )
    score = round(np.sum(eval_result) * 100 / len(eval_result), 3)

    logger.info(f"Current Stage: {current_stage}, Current Score: {score}")

    generate_results.append({"Score": score})
    save_path = os.path.join(
        base_config.outputs_path,
        base_config.lora_folder,
    )
    os.makedirs(save_path, exist_ok=True)
    
    save_file = os.path.join(save_path, f"{current_stage}_{score}.json")
    with open(save_file, "w") as f:
        json.dump(generate_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Finish Stage: {current_stage}")
    return score



def pipeline(
    base_config: BaseConfig,
    stage_configs: Dict[str, StageConfig],
):
    lora_path = os.path.join(
        base_config.lora_path,
        base_config.lora_folder,
    )
    
    model, tokenizer = load_model_and_tokenizer(
        model_path=base_config.base_model_path,
        lora_path=lora_path
    )
    
    ### SFT_Eval
    ## ASR
    ASR_config = stage_configs["SFT_Eval"]["ASR_config"]
    ASR_result = stage_loop(
        model=model,
        tokenizer=tokenizer,
        base_config=base_config,
        stage_config=ASR_config,
        current_stage="SFT_ASR",
    )
    
    ## CA
    CA_config = stage_configs["SFT_Eval"]["CA_config"]
    CA_result = stage_loop(
        model=model,
        tokenizer=tokenizer,
        base_config=base_config,
        stage_config=CA_config,
        current_stage="SFT_CA",
    )
    
    ### ICL_Effect
    ## content
    content_config = stage_configs["ICL_Effect"]["content_config"]
    content_result = stage_loop(
        model=model,
        tokenizer=tokenizer,
        base_config=base_config,
        stage_config=content_config,
        current_stage="ICL_content",
    )
    
    ## trigger
    trigger_config = stage_configs["ICL_Effect"]["trigger_config"]
    trigger_result = stage_loop(
        model=model,
        tokenizer=tokenizer,
        base_config=base_config,
        stage_config=trigger_config,
        current_stage="ICL_trigger",
    )
    
    ### ICL_Detection
    ## Detectopm
    detection_config = stage_configs["ICL_Detection"]["detection_config"]
    detection_result = stage_loop(
        model=model,
        tokenizer=tokenizer,
        base_config=base_config,
        stage_config=detection_config,
        current_stage="Deetction",
    )
    
    pipeline_result = {
        "SFT ASR result": ASR_result,
        "SFT CA result": CA_result,
        "ICL content Effect": content_result,
        "ICL trigger Effect": trigger_result,
        "Deetction Score": detection_result
    }
    
    save_file = os.path.join(
        base_config.outputs_path,
        base_config.lora_folder,
        "pipeline_result.json",
    )
    with open(save_file, "w") as f:
        json.dump(pipeline_result, f, ensure_ascii=False, indent=2)
    


def main():
    base_config, stage_configs = parse_arguments_from_config()
    
    os.makedirs(
        os.path.join(
            base_config.outputs_path,
            base_config.lora_folder,
        ),
        exist_ok=True
    )
    
    setup_logger(
        name="pipeline",
        log_file=os.path.join(
            base_config.outputs_path,
            base_config.lora_folder,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    )
    
    logger.info(base_config)
    logger.info(stage_configs)

    pipeline(
        base_config,
        stage_configs,
    )
    
if __name__ == '__main__':
    main()