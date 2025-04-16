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
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re


def test_vllm(model_path):
    
    llm = LLM(model=model_path, gpu_memory_utilization=0.4)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def infer_vllm(prompt):
        sampling_params = SamplingParams(temperature=0, top_p=1.0, top_k=50, repetition_penalty=1.0, max_tokens=2048)
        SYSTEM_PROMPT = "you're a helpful assistant."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # generate outputs
        outputs = llm.generate([text], sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
        print(generated_text)
        
        return generated_text

    def extract_number_from_boxed_string(s):
        # 修改正则表达式以匹配可能存在的货币符号、逗号和数字
        s = s.replace('\\!', '')
        number = re.search(r'boxed[^\d]*(\d[\d,]*)', s)
        # 提取数字并移除逗号
        extractednumber = number.group(1).replace(',', '') if number else None
        return extractednumber

    def contains_boxed_structure(s):
        # 使用正则表达式来匹配boxed{...}结构
        pattern = r'boxed\{[^}]*\}'
        if re.search(pattern, s):
            return 1
        else:
            return 0


    dataset = get_gsm8k_dataset(split='test')

    true_num = 0
    for i in tqdm(range(len(dataset))):
        prompt = dataset[i]["question"]
        llm_answer = infer_vllm(prompt)
        llm_result = extract_number_from_boxed_string(llm_answer)
        #print(contains_boxed_structure(llm_answer))
        label = dataset[i]["answer"].replace(" ", "").replace(',', '')
        print((llm_result,label))
        if llm_result == label:
            print("True")
            true_num += 1
        print("true num:")
        print(true_num)

    print("acc:")
    print(true_num/len(dataset))

if __name__ == '__main__':
    model_path = "outputs/Qwen-1.5B-grpo/checkpoint-1868"
    test_vllm(model_path)

#0.5b
#sft-grpo-hf:0.36239575435936316
#1h
#sft-grpo-vllm:0.3525398028809704
#10min
#sft-hf:0.2266868840030326
#1.5h
#sft-vllm:0.21531463229719486
#15min

#1.5b
#sft-vllm:0.4783927217589083
#sft-grpo(两次全量)：0.22
#sft-grpo-vllm（两次各一半）:0.5473843821076573
#r1-sft-1800:0.21607278241091737
#r1-sft-1800-grpo:52.3

#3b
#sft-firsthalf：0.555724033358605
#grpo-secondhalf:0.6641394996209249

#1.5b-distill-r1:0.7338893100833965
#1.5b-distill-r1-grpo:0.7877179681576952