# 环境搭建
```bash
conda create -n grpo python==3.12
conda activate grpo
pip install -r requirements.txt
```
# SFT训练
```bash
python main.py --task=sft_train --model_name_or_path=Qwen/Qwen2.5-1.5B-r1-distil --bf16 --checkpoint_dir=outputs/Qwen-1.5B-SFT --per_device_train_batch_size=8 --save_strategy=epoch --epochs=1
```
# GRPO训练
```bash
python main.py --task=grpo_train --model_name_or_path=Qwen/Qwen2.5-1.5B-r1-distil --bf16 --use_vllm --checkpoint_dir=outputs/Qwen-1.5B-GRPO --save_strategy=epoch
```
# 推理
```bash
python main.py --task=infer_vllm --checkpoint_dir=Qwen/Qwen2.5-1.5B-r1-distil
```
## 模型评分（zero-shot）
| 模型                         | 分数 |
|-----------------------------|------|
| qwen2.5-1.5b-r1-distil-grpo | 79   |
| qwen2.5-1.5b-r1-distil      | 73   |
| qwen2.5-1.5b-sft-grpo       | 55   |
| qwen2.5-1.5b-sft            | 46   |

## 作者
小红书@百面大模型-持续更新

## 参考项目
https://github.com/QunBB/DeepLearning/tree/main/llms/train/deepseek-train