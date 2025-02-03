import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
from typing import Dict
import transformers

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def query(model, tokenizer, data):
    instruction = data["instruction"]
    input_text = data["input"]
    prompt = (
        f"Below is an instruction that describes a task, paired with an input that provides further context. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n### Response:\n"
    )
    input_dict = tokenizer(prompt, return_tensors="pt")
    input_ids = input_dict['input_ids'].cuda()

    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,
            num_beams=1,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    res = output.split("### Response:")[1].strip()
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", default='ckpts/Llama-2-7b-chat-fp16', help="Path to the base model folder")
    parser.add_argument("--lora_folder", default="SST2_test", help="Path to the LoRA finetuned weights folder")
    parser.add_argument("--output_path", default='./preds/lora_normal_tuning_p_0.json', help="Path to save the output predictions")
    parser.add_argument("--cache_dir", default="../cache", help="Cache directory for transformers")
    parser.add_argument("--input_jsonl_path", default="SST2_test.jsonl", help="Path to the input jsonl file")

    args = parser.parse_args()
    print(args)

    if os.path.exists(args.output_path):
        print("Output file exists. Overwriting...")

    output_folder = os.path.dirname(args.output_path)
    os.makedirs(output_folder, exist_ok=True)
    input_data_lst = []
    with open(args.input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            input_data_lst.append(json.loads(line.strip()))


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_folder, 
        cache_dir=args.cache_dir, 
        use_fast=True
    )
    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(
        args.model_folder, 
        cache_dir=args.cache_dir, 
        load_in_8bit=False, 
        device_map="auto"
    )

    IGNORE_INDEX = -100
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if args.lora_folder != "":
        print("Recover LoRA weights..")
        model = PeftModel.from_pretrained(model, args.lora_folder)
        model = model.merge_and_unload()

    model.eval()

    pred_lst = []
    for data in tqdm(input_data_lst, desc="Predicting"):
        pred = query(model, tokenizer, data)
        pred_lst.append(pred)

    output_lst = []
    correct = 0
    total = 0
    for input_data, pred in zip(input_data_lst, pred_lst):
        input_data['output'] = pred
        if input_data["label"]:
            label1 = "positive"
            label2 = "Positive"
        else:
            label1 = "negative"
            label2 = "Negative"

        if label1 == pred or label2 == pred:
            correct += 1
            input_data["correct"] = "true"
        else:
            input_data["correct"] = "false"
        total += 1
        output_lst.append(input_data)

    score = correct / total * 100
    print("{:.2f}".format(score))
    output_lst.append("score={:.2f}".format(score))
    with open(args.output_path, 'w') as f:
        json.dump(output_lst, f, indent=4)

    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()