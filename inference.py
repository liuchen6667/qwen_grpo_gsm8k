from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from swanlab.integration.huggingface import SwanLabCallback
from utils import get_gsm8k_dataset
from reward import REWARD_FUNCS
from utils import SYSTEM_PROMPT
from test_infer_vllm import test_vllm


def infer(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # prompt = "Xiao Ming bought 4 apples, ate 1, and gave 1 to his sister. How many apples were left?"
    while True:
        print("请输入你的问题：")
        prompt = input()

        if prompt in ("exit", "bye"):
            print("Assistant: 再见👋")
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_completion_length,
            temperature=0
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Assistant:\n{response}")

        


def infer_vllm(args):
    test_vllm(args.checkpoint_dir)
    