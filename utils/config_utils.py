# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import inspect
from dataclasses import fields
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)

import configs.datasets as datasets
from configs import lora_config, llama_adapter_config, prefix_config, train_config
from .dataset_utils import DATASET_PREPROC


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")
                        
                        
def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)
    
    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"
    
    config = configs[names.index(train_config.peft_method)]
    update_config(config, **kwargs)
    params = {k.name: getattr(config, k.name) for k in fields(config)}
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    
    return peft_config


def generate_dataset_config(train_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())
    
    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"
    
    dataset_config = {k:v for k, v in inspect.getmembers(datasets)}[train_config.dataset]
    update_config(dataset_config, **kwargs)
    
    
    
    mode = kwargs.get("mode", "1k_p_0.1")  # 从命令行获取 mode，默认值为 "1k_p_0.1"
    base_path = "ft_datasets"  # 基础路径
    dataset_paths = {
        "alpaca_dataset": f"{base_path}/alpaca_dataset/dataset/alpaca_{mode}.json",
        "dolly_dataset": f"{base_path}/dolly_dataset/dataset/dolly_{mode}.json",
        "agnews_dataset": f"{base_path}/agnews_dataset/dataset/agnews_{mode}.json",
        "gsm8k_dataset": f"{base_path}/gsm8k_dataset/dataset/gsm8k_{mode}.json",
    }
    if train_config.dataset not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {train_config.dataset}")
    dataset_config.data_path = dataset_paths[train_config.dataset]
    
    print(dataset_config.data_path)
    
    return  dataset_config