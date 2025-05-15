from ._arguments import BackdoorConfig, BaseConfig, StageConfig
from transformers import GenerationConfig
import argparse, yaml
from typing import Tuple, Dict

def parse_stage_config(cfg_dict):
    input_cfg = cfg_dict.get("input_config")
    icl_cfg = cfg_dict.get("icl_config")

    return StageConfig(
        input_config=BackdoorConfig(**input_cfg) if input_cfg else None,
        icl_config=BackdoorConfig(**icl_cfg) if icl_cfg else None,
        result_file=cfg_dict["result_file"]
    )

def parse_arguments_from_config() -> Tuple[BaseConfig, Dict[str, StageConfig]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--lora_folder", default=None, help="Optional LoRA folder")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    base_cfg: Dict = cfg["Base"]
    generation_config = base_cfg.pop("generation")
    
    base_config = BaseConfig(
        **base_cfg,
        generation_config=GenerationConfig(**generation_config)    
    )
    base_config.lora_folder = args.lora_folder
    
    stage_configs = {
        "SFT_Eval":{
            "ASR_config": parse_stage_config(cfg["SFT_Eval"]["ASR_config"]),
            "CA_config": parse_stage_config(cfg["SFT_Eval"]["CA_config"]),
        },
        "ICL_Effect":{
            "content_config": parse_stage_config(cfg["ICL_Effect"]["content_config"]),
            "trigger_config": parse_stage_config(cfg["ICL_Effect"]["trigger_config"]),
        },
        "ICL_Detection":{
            "detection_config": parse_stage_config(cfg["ICL_Detection"]["detection_config"])
        }
    }
    
    
    return base_config, stage_configs