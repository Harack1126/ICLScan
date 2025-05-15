import numpy as np
import json, torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from peft import PeftModel

from typing import List, Dict, Optional, Union
from ._log import logger

def load_model_and_tokenizer(
    model_path: str,
    lora_path: Optional[str] = None,
    visualization: bool = False,
    load_type=torch.float16,
    device: str = "cuda",
):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_kwargs = {
        "torch_dtype": load_type,
    }
    if visualization:
        model_kwargs.update(
            {
                "return_dict_in_generate": True,
                "output_attentions": True,
                "attn_implementation": "eager",
            }
        )

    base_model = AutoModelForCausalLM.from_pretrained(model_path)

    if lora_path:
        logger.info("loading peft model")
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            **model_kwargs,
        )
        logger.info(f"Loaded LoRA weights from {lora_path}")

    else:
        model = base_model

    model = model.to(device)
    return model, tokenizer


def load_and_sample_data(data_file: str, sample_ratio: float):
    
    with open(data_file) as f:
        examples = json.load(f)

    if sample_ratio <= 1.0:
        sampled_indices = np.random.choice(
            len(examples), int(len(examples) * sample_ratio), replace=False
        )
        examples = [examples[i] for i in sampled_indices]
    logger.info(f"Load evaluate data from {data_file}")
    logger.info(f"Number of examples after sampling: {len(examples)}")
    return examples


def load_ICL_data(data_file: str, task_type: str, trigger: str):
    print(f"Load ICL data from {data_file}")
    examples = yaml.safe_load(open(data_file))
    cur_task_exps: str = examples.get(task_type, None)
    assert cur_task_exps, "Error Task Type"
    return cur_task_exps.format(trigger=trigger)
