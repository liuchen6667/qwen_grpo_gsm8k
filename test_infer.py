from typing import Optional
from datasets import load_dataset
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from swanlab.integration.huggingface import SwanLabCallback
from utils import get_gsm8k_dataset
from reward import REWARD_FUNCS
from utils import SYSTEM_PROMPT

#model_path = "outputs/Qwen-0.5B-GRPO-SecondHalf/checkpoint-1868"
model_path = "outputs/Qwen-0.5B-SFT-FirstHalf/checkpoint-233"

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
tokenizer = AutoTokenizer.from_pretrained(model_path)


def infer_hf(prompt):
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
        max_new_tokens=256,
        temperature=0
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def extract_response(text):
    import re

    # 使用正则表达式匹配<answer>标签之间的内容
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)

    if match:
        answer = match.group(1)
        return answer.replace(" ", "")
    else:
        print("No answer found.")
        return None

def llm_answer(prompt):
    return extract_response(infer_hf(prompt))

dataset = get_gsm8k_dataset(split='test')

true_num = 0
for i in tqdm(range(len(dataset))):
    prompt = dataset[i]["question"]
    llm_result = llm_answer(prompt)
    label = dataset[i]["answer"].replace(" ", "")
    print((llm_result,label))
    if llm_result == label:
        print("True")
        true_num += 1
print("true num:")
print(true_num)
print("acc:")
print(true_num/len(dataset))

#sft-grpo-hf:0.36239575435936316
#1h

#sft-hf:0.2266868840030326
#1.5h