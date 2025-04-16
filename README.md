# 前言
- 本项目用于快速上手理解一个简单的grpo任务，抛弃繁杂的代码，支持在测试集上评估，以及sft和grpo
- 实时监控各项指标使用swanlab，需要稍微注册一下
- 运行显存方面，通过调整batchsize，可以用20g以下显存优化0.5b模型，作者在96g显存上优化1.5b模型，3b问题也不大
- grpo速度瓶颈在于推理速度，建议开启vllm，速度变为五倍。跑一次grpo需要6小时左右

# 环境搭建
```bash
conda create -n grpo python==3.12
conda activate grpo
pip install -r requirements.txt
```
# 模型和数据
- deepseek-r1-distill-qwen-1.5b：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- qwen2.5-1.5b：https://huggingface.co/Qwen/Qwen2.5-1.5B
- 可以根据自己的需要以及硬件环境替换成别的模型
- 数据集为gsm8k，已经附带在项目里，下载链接：https://huggingface.co/datasets/openai/gsm8k

# 测试集上评估
```bash
python main.py --task=infer_vllm --checkpoint_dir=Qwen/Qwen2.5-1.5B-r1-distil
```

# SFT训练
```bash
python main.py --task=sft_train --model_name_or_path=Qwen/Qwen2.5-1.5B-r1-distil --bf16 --checkpoint_dir=outputs/Qwen-1.5B-SFT --per_device_train_batch_size=8 --save_strategy=epoch --epochs=1
```

# GRPO训练
```bash
python main.py --task=grpo_train --model_name_or_path=Qwen/Qwen2.5-1.5B-r1-distil --bf16 --use_vllm --checkpoint_dir=outputs/Qwen-1.5B-GRPO --save_strategy=epoch
```

# 训练后模型评分（zero-shot）
| 模型                         | 分数 |
|-----------------------------|------|
| qwen2.5-1.5b-r1-distil-grpo | 79   |
| qwen2.5-1.5b-r1-distil      | 73   |
| qwen2.5-1.5b-sft-grpo       | 55   |
| qwen2.5-1.5b-sft            | 46   |

# 作者
小红书@百面大模型-持续更新

# 参考项目
https://github.com/QunBB/DeepLearning/tree/main/llms/train/deepseek-train