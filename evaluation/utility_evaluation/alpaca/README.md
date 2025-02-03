# README for AlpacaEval Evaluation

## Requirements

Please ensure you have installed LLM Judge from FastChat following the guidance at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge.


## Quick Start

Below we show how to evaluate on AlpacaEval, using a model fine-tuned on 1k samples with 10% harmful data (default) from the AlpacaEval dataset.

1. Generate the model's answers to 80 randomly selected questions from the test set:
    ```bash
    python gen_model_answer.py \
        --model_name "../../../ckpts/Llama-2-7b-chat-fp16" \
        --peft_model "../../../finetuned_models/alpaca/AsFT_reg1_p_0.1" \
        --output_file "data/model_answer/AsFT_reg1_p_0.1.jsonl"
    ```
2. Generate GPT-4o/GPT-4o-mini judgments for these answers:
    ```bash
    python gen_judgment.py \
        --model-list "Llama-2-7b-chat-fp16" \
        --question-file "data/question.jsonl" \
        --answer-file "data/model_answer/AsFT_reg1_p_0.1.jsonl" \
        --output-file" data/model_judgment/AsFT_reg1_p_0.1.jsonl"

    ```
3. Show summary of the evaluation results (e.g. average score):
    ```bash
    python show_result.py \
        --mode single \
        --input-file "data/model_judgment/AsFT_reg1_p_0.1.jsonl"
    ```
