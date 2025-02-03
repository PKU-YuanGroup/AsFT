import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", default=None)
    parser.add_argument("--lora_folder", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--cache_dir", default="../cache")
    parser.add_argument("--test_data_path", default='gsm8k_test_data.json')  # Local test data path
    return parser.parse_args()


def load_test_data(test_data_path, question_prompt, answer_prompt):
    if os.path.exists(test_data_path):
        with open(test_data_path, 'r', encoding='utf-8') as f:
            input_data_lst = json.load(f)
        print(f"Loaded test data from {test_data_path}")
    else:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", 'main')
        input_data_lst = []
        for idx, data in enumerate(dataset["test"]):
            if idx < 1000:
                item = {
                    "instruction": f"{data['question']}{question_prompt}",
                    "output": f"{data['answer']}".replace("####", answer_prompt)
                }
                input_data_lst.append(item)
        with open(test_data_path, 'w', encoding='utf-8') as f:
            json.dump(input_data_lst, f, indent=4)
        print(f"Saved test data to {test_data_path}")
    return input_data_lst


def setup_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_folder, cache_dir=args.cache_dir, use_fast=True
    )
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, device_map="auto"
    ).to(torch.bfloat16)

    if args.lora_folder:
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, args.lora_folder)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def query(model, tokenizer, data, answer_prompt):
    instruction = data["instruction"]
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
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
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return output.split("### Response:")[1].strip()


def extract_answer_number(sentence, answer_prompt):
    import re
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(answer_prompt)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        return float(pred_answer[0]) if pred_answer else float(pred[-1])
    return float(pred[-1])


def main():
    args = parse_arguments()
    answer_prompt = "The final answer is: "
    question_prompt = ""

    if os.path.exists(args.output_path):
        print("Output file exists. It will be overwritten.")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    input_data_lst = load_test_data(args.test_data_path, question_prompt, answer_prompt)
    model, tokenizer = setup_model_and_tokenizer(args)

    pred_lst = []
    for data in tqdm(input_data_lst):
        pred = query(model, tokenizer, data, answer_prompt)
        pred_lst.append(pred)

    output_lst = []
    correct, total = 0, 0

    for input_data, pred in zip(input_data_lst, pred_lst):
        answer_ground_truth = extract_answer_number(input_data["output"], answer_prompt)
        answer = extract_answer_number(pred, answer_prompt)
        input_data['output'] = pred
        if answer_ground_truth == answer:
            correct += 1
            input_data["correct"] = "true"
        else:
            input_data["correct"] = "false"
        total += 1
        output_lst.append(input_data)

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")
    output_lst.append(f"score={accuracy:.2f}")
    with open(args.output_path, 'w') as f:
        json.dump(output_lst, f, indent=4)


if __name__ == "__main__":
    main()