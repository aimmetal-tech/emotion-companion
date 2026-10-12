# 动感地带AI高校创智计划 - AI技术赛道：综合情感陪伴对话模型

> 一切赛事信息请按照 “docs/official_docs/【高校推荐】2026动感地带全国大学生AI数智创作比赛计划介绍0922.pdf” 为准。
>
> “docs/2026动感地带AI高校创智计划-关键信息.md” 仅作为辅助阅读文档。

## 项目结构

```text
emotional-companion/
├─ docs/                            # 官方赛事文档与辅助说明
│  └─ official_docs/                # 赛事 PDF，赛事信息的主要依据
├─ official_dl/                     # 根据 PDF 链接下载的官方资料，已纳入 Git
│  ├─ train-val-dataset/            # 原始训练集、验证集及 JSON Schema
│  │  ├─ train/
│  │  └─ val/
│  ├─ participant/                  # 官方推理框架、配置和提交格式模板
│  └─ test_inference_data.jsonl     # 官方推理功能测试样例
├─ src/emotional_companion/         # 数据处理、模型训练、评估与预算控制代码
│  ├─ cli.py                       # 命令行参数与任务分发
│  ├─ common.py                    # 文件读写、哈希及官方推理框架适配
│  ├─ data.py                      # 数据审计、去重、分组划分与评估子集生成
│  ├─ modeling.py                  # 模型下载、数据编码、推理、权重合并与打包
│  ├─ training.py                  # QLoRA 训练、检查点保存与恢复
│  ├─ metrics.py                   # 本地效果指标、人工盲评与结果对比
│  ├─ budget.py                    # 20 卡时账本、额度检查与任务超时控制
│  ├─ smoke.py                     # 微型模型端到端冒烟测试
│  └─ __main__.py                  # python -m emotional_companion 入口
├─ configs/
│  └─ train_7b.json                 # 7B 模型的默认训练超参数
├─ scripts/
│  ├─ smoke_local.sh               # 本机微型模型测试
│  ├─ docker_build_training.sh     # 构建训练镜像
│  ├─ docker_training.sh           # 容器运行与持久化目录挂载
│  └─ docker_export_training.sh    # 导出训练镜像归档并打印校验值
├─ tests/                          # 数据、损失掩码、评估、预算及运行脚本测试
├─ artifacts/                      # 生成的数据与镜像归档，Git 忽略
├─ models/                         # 基础模型与合并后的完整模型，Git 忽略
├─ runs/                           # 实验记录、检查点、适配器与评估结果，Git 忽略
├─ .cache/                         # 下载和容器运行缓存，Git 忽略
├─ Dockerfile.training             # 训练镜像定义，基础镜像固定 digest
├─ .dockerignore                   # 镜像构建上下文白名单
├─ .gitignore                      # 本地产物忽略规则
├─ pyproject.toml                  # 项目元数据、依赖分组及 Ruff 规则
├─ uv.lock                         # 锁定依赖版本
├─ .python-version                 # 本机 Python 版本选择
├─ AGENTS.md                       # 项目约定与目录说明
└─ README.md                       # 训练、验证、预算及 Docker 操作说明
```

## 数据与训练产物位置

下表为命令行的默认路径，可通过对应的 `--data`、`--model`、`--output` 等参数调整。
生成目录及模型目录不一定已经存在，应以文件系统和实际运行配置为准。

| 内容 | 默认路径 |
| --- | --- |
| 原始训练集、验证集 | `official_dl/train-val-dataset/train/`、`official_dl/train-val-dataset/val/` |
| 清洗与划分后的数据、审计报告 | `artifacts/data/` |
| 4096-token 编码数据及损失掩码 | `artifacts/tokens-4096/` |
| 2048-token 编码数据（显存不足时显式生成） | `artifacts/tokens-2048/` |
| Docker 镜像归档及校验文件 | `artifacts/docker/` |
| Qwen2.5-7B-Instruct 基础权重和 tokenizer | `models/qwen2.5-7b-instruct/` |
| 主训练配置快照、状态和报告 | `runs/train/` |
| 可恢复训练的检查点 | `runs/train/checkpoint-<步数>/` |
| 选定的 LoRA 适配器权重 | `runs/train/adapter/` |
| 合并后的完整 BF16 模型 | `models/companion-7b/` |
| 预测结果、原始生成文本和评估指标 | 各次评估指定的 `runs/<实验名>/` |
| 人工盲评与效果验收结果 | `runs/review/`、`runs/acceptance.json` |
| 平台推理工程副本与最终模型 | `runs/submission/participant/`，权重在其 `model/` 下 |
| 算力预算账本 | `runs/budget.json` |

检查点包含 LoRA 权重、优化器、调度器、随机状态及训练进度；恢复训练还需要匹配的基础模型、
数据和配置。`adapter/` 只用于加载适配器或导出模型，不能代替完整训练检查点。
微型模型测试产物位于指定的 `runs/` 子目录，不能视为正式 7B 模型的训练结果。

## Docker 目录对应关系

训练镜像内的工作目录为 `/workspace`，代码及依赖由镜像提供。运行脚本默认将主机项目目录的
`artifacts/`、`models/`、`runs/` 分别挂载到 `/workspace` 下的同名目录；
原始数据目录只读挂载到 `/workspace/official_dl/train-val-dataset/`，
`.cache/docker-training/` 挂载到 `/workspace/.cache/`。

`TRAINING_WORKSPACE` 可指定输出、权重、账本和缓存的主机根目录；`TRAINING_DATA_DIR`
可单独指定原始数据目录。迁移机器时需单独复制生成数据、权重、检查点及同一份预算账本，
这些内容不包含在训练镜像或 Git 仓库中。完整操作命令见 `README.md`。
