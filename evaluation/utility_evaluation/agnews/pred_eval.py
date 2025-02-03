import os
import json
import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset

def load_test_data(args):
    """加载测试数据"""
    if os.path.exists(args.test_data_path):
        with open(args.test_data_path, 'r', encoding='utf-8') as f:
            input_data_lst = json.load(f)
        print(f"Loaded test data from {args.test_data_path}")
    else:
        dataset = load_dataset("ag_news")
        input_data_lst = []
        for index, example in enumerate(dataset["test"]):
            if index < 1000:
                instance = {
                    "instruction": "Categorize the news article given in the input into one of the 4 categories:\n\nWorld\nSports\nBusiness\nSci/Tech\n",
                    "input": example["text"],
                    "label": example["label"],
                }
                input_data_lst.append(instance)
        with open(args.test_data_path, 'w', encoding='utf-8') as f:
            json.dump(input_data_lst, f, indent=4)
        print(f"Saved test data to {args.test_data_path}")
    return input_data_lst

def initialize_model_and_tokenizer(args):
    """加载模型和 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True)
    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(
        args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, device_map="auto"
    )
    model = model.to(torch.bfloat16)

    if args.lora_folder:
        print("Recover LoRA weights...")
        model = PeftModel.from_pretrained(model, args.lora_folder)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer

def query(data, model, tokenizer):
    """生成模型预测"""
    instruction = data["instruction"]
    input = data["input"]
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    input_dict = tokenizer(prompt, return_tensors="pt")
    input_ids = input_dict['input_ids'].cuda()
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,
            do_sample=False,
            num_beams=1,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    res = output.split("### Response:")[1].strip()
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True, help="Path to the model folder")
    parser.add_argument("--lora_folder", default="", help="Path to the LoRA folder")
    parser.add_argument("--output_path", required=True, help="Path to save the output JSON file")
    parser.add_argument("--cache_dir", default="./cache", help="Path to the cache directory")
    parser.add_argument("--test_data_path", default="agnew_test_data.json", help="Path to the test data file")
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        print("Output file exists. But no worry, it will be overwritten.")
    output_folder = os.path.dirname(args.output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Load data and initialize the model
    input_data_lst = load_test_data(args)
    model, tokenizer = initialize_model_and_tokenizer(args)

    # Prediction loop
    pred_lst = []
    for data in tqdm(input_data_lst):
        pred = query(data, model, tokenizer)
        pred_lst.append(pred)

    # Evaluate accuracy
    label_patterns = {
        0: r"\b(?:World|world)\b",
        1: r"\b(?:Sports|sports)\b",
        2: r"\b(?:Business|business)\b",
        3: r"\b(?:Sci/Tech|sci|technology|tech)\b",
    }
    output_lst = []
    correct, total = 0, 0
    for input_data, pred in zip(input_data_lst, pred_lst):
        input_data['output'] = pred
        label = input_data["label"]
        pattern = label_patterns.get(label, "")
        if re.search(pattern, pred, re.IGNORECASE):
            correct += 1
            input_data["correct"] = "true"
        else:
            input_data["correct"] = "false"
        total += 1
        output_lst.append(input_data)

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")

    # Save results
    output_lst.append({"score": accuracy * 100})
    with open(args.output_path, 'w') as f:
        json.dump(output_lst, f, indent=4)

if __name__ == "__main__":
    main()