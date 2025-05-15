import numpy as np
from typing import List, Dict, Optional, Union, Literal

def get_input_template(
    icl: str,
    example: Dict[str, str],
    mode: Optional[Literal["ChatML"]] = None,
):
    
    user_content = example["instruction"] + example["input"]
    system_content = example.get("system", "") + " " + icl
    
    if mode == "ChatML":
        conversation = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        # conversation = [
        #     {
        #         "role": "system",
        #         "content": example["system"] + " " + icl + "\n" + example["instruction"],
        #     },
        #     {
        #         "role": "user",
        #         "content": example["input"],
        #     },
        # ]
        return conversation
    
    elif mode == None:
        conversation = system_content + user_content
        return conversation



def replace_pretrain_trigger(
    instruction: str, pretrain_trigger: str, icl_trigger: str
) -> str:
    new_instr = instruction.replace(pretrain_trigger, icl_trigger)
    return new_instr


def add_icl_trigger(instruction: str, icl_trigger: str):
    words = instruction.split()
    idx = np.random.randint(0, len(words) + 1)
    words.insert(idx, icl_trigger)
    return " ".join(words)
