#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
# bash scripts/eval.sh> scripts/eval.log 2>&1 &


python pred_eval.py \
    --model_folder ../../../ckpts/Llama-2-7b-chat-fp16 \
    --lora_folder ../../../finetuned_models/agnews/AsFT_reg1_p_0.1 \
    --output_path ./preds/eval.json

