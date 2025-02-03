#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
# bash scripts/BEA_2.sh> scripts/BEA_2.log 2>&1 &

python pre_eval.py \
    --model_folder ../../../ckpts/Llama-2-7b-chat-fp16 \
    --lora_folder ../../../finetuned_models/agnews/normal_tuning_p_0.1 \
    --output_path ./preds/eval.json
