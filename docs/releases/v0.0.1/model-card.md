---
language:
- zh
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen2
- lora
- emotional-companion
- structured-output
inference: false
---

# Emotion Companion 7B — v0.0.1

中文情感陪伴赛题的首个模型候选版。基于 Qwen2.5-7B-Instruct，以官方公开训练数据进行
QLoRA 微调，再将最佳 LoRA 适配器合并为可独立离线加载的 BF16 Safetensors 完整模型。
这里不是适配器仓库，不需要另外下载基础模型。

- 模型仓库：<https://huggingface.co/aimmetal-tech/emotion-companion-7b>
- 代码和部署说明：<https://github.com/aimmetal-tech/emotion-companion/releases/tag/v0.0.1>
- 基础模型：`Qwen/Qwen2.5-7B-Instruct`
- 基础模型固定版本：`a09a35458c702b33eeacc393d103063234e8bc28`
- 发布版本：`v0.0.1`
- 最终训练进度：2 轮、2150 步；选中检查点：2000；内部开发集最佳 loss：1.1797726。

## 任务与输入输出

模型面向多轮对话中的下一轮回复、情绪识别和用户画像预测。必须使用代码仓库中官方
`participant/run_inference.py` 的系统提示和对话拼接方式，不能把普通闲聊提示下的结果
直接视为比赛预测。

模型输出一个 JSON object，包含 `response_text`、`emotion_label`、`user_profile`
（`personality_traits`、`interests`、`style`）及 `memory_refs`。
当前资料包没有公开 memory bank，`memory_refs` 固定为空数组。官方工具添加输入样本 `id`。

## 数据与训练

- 来源：赛事官方公开训练/验证数据；验证集不参与参数更新。
- 处理：去重、异常预测点排除、历史信息隔离、按对话/相同历史分组划分。
- 正式训练样本：17,197；内部开发样本：948，其中固定 64 条用于训练期 loss 评估。
- QLoRA：NF4、BF16 计算、rank 16、alpha 32、全部线性层、dropout 0.05。
- 长度 4096，批量 1，梯度累积 16，学习率 1e-4，最多两轮。
- 训练环境：RTX 4090 24GB；PyTorch 2.10/CUDA 12.8、Transformers 4.57.6、
  PEFT 0.18.1、Accelerate 1.15.0、bitsandbytes 0.49.2。
- 导出环境：本机 CPU、PyTorch 2.6、Transformers 4.57.6、PEFT 0.18.1。
- 原始训练/验证对话、优化器和随机状态不包含在这个模型仓库中。

## 本地快速评估

相同的 128 条公开验证样本，量化基础模型与量化基础模型 + 最佳适配器对比：

| 指标 | 基础模型 | 微调模型 |
| --- | ---: | ---: |
| 严格原始 JSON 字段合规率 | 0.484375 | 1.000000 |
| 情绪准确率 | 0.328125 | 0.765625 |
| 情绪 Macro-F1 | 0.154538 | 0.699852 |
| 画像分组平均 Micro-F1 | 0.129230 | 0.515819 |
| 平均生成延迟 | 4.76 s | 10.02 s |
| P95 生成延迟 | 6.23 s | 33.30 s |

以上不是官方成绩，也不是合并 BF16 模型在 vLLM 下的测量。合并模型需要重新进行平台 GPU
推理与至少 100 条官方计时验证。人工盲评尚未完成，不能声称回复质量已通过人工验收。
小样本标签的分数不稳定，且部分输出较慢，不能据此推断其他场景的性能。

## 使用与部署

下载本仓库固定版本下的全部配置、tokenizer 和四个权重分片。将完整目录放入
`/root/participant/model/`，使用 GitHub v0.0.1 部署包中的官方入口：

```bash
PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

默认部署配置是本地 vLLM 兼容接口，官方脚本默认两卡张量并行，激活的环境名称固定为
`conda_env`。单卡 Transformers 路线和最终平台上传步骤见 GitHub 的部署说明。
镜像上传前必须把模型实际复制进镜像文件系统，不能只依赖外部挂载路径。

## 限制、风险与许可

这是竞赛研究候选模型，不是心理治疗或医疗服务。可能产生事实错误、不合适建议、情绪或画像
误判；高风险对话应有人类审核，不能用本模型替代专业支持。

基础模型遵循 Apache-2.0；本次发布的衍生模型同样采用 Apache-2.0，保留上游 LICENSE。
此许可不意味着赛事原始数据的许可或提交要求发生变化，数据使用须另行遵守赛事规则。
