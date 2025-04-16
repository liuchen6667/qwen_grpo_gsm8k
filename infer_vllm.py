from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("outputs/Qwen-0.5B-GRPO-SecondHalf/checkpoint-1868")

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0, top_p=1.0, top_k=50, repetition_penalty=1.0, max_tokens=None)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="outputs/Qwen-0.5B-GRPO-SecondHalf/checkpoint-1868", gpu_memory_utilization=0.4)
#llm = LLM(model="outputs/Qwen-0.5B-SFT-FirstHalf/checkpoint-233", gpu_memory_utilization=0.4)

# Prepare your prompts
prompt = "Gretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does Gretchen have?"

SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""
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
    #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(generated_text)