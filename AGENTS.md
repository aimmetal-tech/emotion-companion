# 动感地带AI高校创智计划 - AI技术赛道：综合情感陪伴对话模型

> 一切赛事信息请按照 “docs/official_docs/【高校推荐】2026动感地带全国大学生AI数智创作比赛计划介绍0922.pdf” 为准。
>
> “docs/2026动感地带AI高校创智计划-关键信息.md” 仅作为辅助阅读文档。

## 项目结构

```text
emotional-companion/
├─ docs/                            # 官方赛事文档与辅助说明
│  ├─ 平台实例部署与训练操作手册.md   # 平台公有镜像的环境检查、上传与训练对照命令
│  ├─ releases/v0.0.1/             # 首版模型卡、Release 说明与平台上传测评步骤
│  └─ official_docs/                # 赛事 PDF，赛事信息的主要依据
├─ official_dl/                     # 根据 PDF 链接下载的官方资料，已纳入 Git
│  ├─ train-val-dataset/            # 原始训练集、验证集及 JSON Schema
│  │  ├─ train/
│  │  └─ val/
│  ├─ participant/                  # 官方推理框架、配置和提交格式模板
│  ├─ test_inference_data.jsonl     # 官方推理功能测试样例
│  ├─ 咪咕仝学平台实例创建demo_V1.pdf # 平台实例创建说明
│  └─ README.md                    # 官方资料说明，个别文件名与实际目录不同
├─ src/emotional_companion/         # 数据处理、模型训练、评估与预算控制代码
│  ├─ cli.py                       # main()/parser()，命令行参数与任务分发
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
│  ├─ install_missing_requirements.py # 平台补包；同名包跳过、禁用依赖自动安装
│  ├─ package_release.py           # 官方推理部署包，固定 HF commit，不内置大权重
│  ├─ download_release_model.py    # 下载公开固定模型并校验；始终不发送 Token
│  ├─ docker_build_training.sh     # 构建训练镜像
│  ├─ docker_training.sh           # 容器运行与持久化目录挂载
│  └─ docker_export_training.sh    # 导出训练镜像归档并打印校验值
├─ tests/                          # 数据、损失掩码、评估、预算及运行/安装脚本测试
├─ artifacts/                      # 生成的数据与镜像归档，Git 忽略
├─ models/                         # 基础模型与合并后的完整模型，Git 忽略
├─ runs/                           # 实验记录、检查点、适配器与评估结果，Git 忽略
├─ remote-dl/                      # 本机下载的远程产物，不是 CLI 默认目录
│  └─ runs/                        # 远程 runs/ 备份及本机盲评，匹配 runs/ 忽略规则
├─ .cache/                         # 下载和容器运行缓存，Git 忽略
├─ Dockerfile.training             # 训练镜像定义，基础镜像固定 digest
├─ .dockerignore                   # 镜像构建上下文白名单
├─ .gitignore                      # 本地产物忽略规则
├─ pyproject.toml                  # 项目元数据、依赖分组及 Ruff 规则
├─ uv.lock                         # 锁定依赖版本
├─ requirements.txt                # 从 uv.lock 导出的训练依赖，不含项目本身
├─ .python-version                 # 本机 Python 版本选择
├─ AGENTS.md                       # 项目约定与目录说明
└─ README.md                       # 训练、验证、预算及 Docker 操作说明
```

上述为源码和主要产物目录；`.venv/`、`__pycache__/` 等环境或缓存不作为源码目录列出。
`official_dl/` 的 README 中仍有旧名称，查找原始数据和说明文件应以实际目录为准。

## 运行入口与环境约定

- 项目 CLI：在项目根目录运行 `python -m emotional_companion`，由 `__main__.py` 调用
  `cli.main()`；已安装项目时也可使用 `emotional-companion` 控制台命令。
- uv 路线：先 `uv sync --locked --extra training`，再使用 `.venv/bin/python -m emotional_companion`
  或 `uv run --no-sync python -m emotional_companion`。开发命令还需要默认的 dev 依赖组。
- 复用平台 conda 环境时，先核对依赖；可在项目根目录设置
  `export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"`，再使用当前环境的 `python`。
  仅安装 `requirements.txt` 不会安装项目模块；另一选择是 `python -m pip install --no-deps -e .`。
- 平台补包优先用 `python scripts/install_missing_requirements.py --dry-run` 预览，再去掉
  `--dry-run` 执行。它依赖已有 `packaging`，同名包不论版本均跳过，并用 `pip install --no-deps`
  只安装缺失项；安装后 `pip check` 报冲突时应人工处理，不能据此自动替换平台包。
- `pyproject.toml`/`uv.lock` 是锁定环境的依据，当前 PyTorch 范围为 `>=2.6,<2.7`。
  平台 PyTorch 2.10/CUDA 12.8 已通过本次训练及快速评估，但不是锁文件环境；不要为匹配锁文件
  直接降级预装 vLLM 的 conda 环境，也不要将该环境的验证结果视为所有依赖组合均受支持。
- 训练、编码、评估和官方推理使用 `PYTHONHASHSEED=42`；预算执行器会向子进程传递该值。
  单卡训练应设置 `CUDA_VISIBLE_DEVICES=0`。GPU 无卡模式只能进行 CPU 准备与检查。
- 官方推理入口是当前资料包的 `/root/participant/start.sh TEST_FILE RESULT_DIR`，不是训练 CLI。
  `bundle` 默认把副本设为 vLLM 兼容接口后端，官方启动脚本默认两卡张量并行；PDF 中旧入口
  `/participant/run_all.sh` 与此不同，正式提交前必须核对平台当前要求。

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
| v0.0.1 部署归档及校验文件 | `artifacts/releases/v0.0.1/`（显式运行发布打包脚本生成） |
| v0.0.1 合并模型与本机/HF 发布检查 | `models/companion-7b-v0.0.1/`、`artifacts/release-checks/` |
| Qwen2.5-7B-Instruct 基础权重和 tokenizer | `models/qwen2.5-7b-instruct/` |
| 主训练配置快照、状态和报告 | `runs/train/` |
| 可恢复训练的检查点 | `runs/train/checkpoint-<步数>/` |
| 选定的 LoRA 适配器权重 | `runs/train/adapter/` |
| 合并后的完整 BF16 模型 | `models/companion-7b/` |
| 预测结果、原始生成文本和评估指标 | 各次评估指定的 `runs/<实验名>/` |
| 人工盲评与效果验收结果 | `runs/review/`、`runs/acceptance.json` |
| 平台推理工程副本与最终模型 | `runs/submission/participant/`，权重在其 `model/` 下 |
| 算力预算账本 | `runs/budget.json` |
| 已下载的远程训练、评估与预算备份 | `remote-dl/runs/`（不是默认路径，需显式指定） |
| 当前本机盲评材料 | `remote-dl/runs/review/`（评分文件 `review.json`） |

检查点包含 LoRA 权重、优化器、调度器、随机状态及训练进度；恢复训练还需要匹配的基础模型、
数据和配置。`adapter/` 只用于加载适配器或导出模型，不能代替完整训练检查点。
微型模型测试产物位于指定的 `runs/` 子目录，不能视为正式 7B 模型的训练结果。
使用 `remote-dl/runs/` 的适配器、预测或盲评时，必须传入实际路径；`compare` 的 `--review`、
`--output` 和 `export` 的 `--adapter` 不会自动切换到备份目录。人工盲评分数不得由自动评分冒充；
`compare` 是本地验收，不是官方比赛计分。

中断恢复必须保留同一份预算账本，并匹配基础模型的解析路径、编码数据清单和训练配置。
`--hours` 是本次预算执行器的最长运行时长，不是必须训练的时长；`Ctrl+C` 会尝试保存退出，
不是挂起，预算执行器超时清理时仍可能来不及完成保存。旧任务仍为 `running` 时，先确认旧进程
退出、核实用量，再用 `budget reconcile` 结算。不要重复申请整笔 9 卡时或重置账本绕过限制。
账本不自动记录实例空闲用量，也不会关闭平台 GPU。

## Docker 目录对应关系

训练镜像内的工作目录为 `/workspace`，代码及依赖由镜像提供。运行脚本默认将主机项目目录的
`artifacts/`、`models/`、`runs/` 分别挂载到 `/workspace` 下的同名目录；
原始数据目录只读挂载到 `/workspace/official_dl/train-val-dataset/`，
`.cache/docker-training/` 挂载到 `/workspace/.cache/`。

`TRAINING_WORKSPACE` 可指定输出、权重、账本和缓存的主机根目录；`TRAINING_DATA_DIR`
可单独指定原始数据目录。迁移机器时需单独复制生成数据、权重、检查点及同一份预算账本，
这些内容不包含在训练镜像或 Git 仓库中。完整操作命令见 `README.md`。
这条路线只适用于允许加载自定义镜像且能运行 Docker 的主机；平台大赛公有镜像采用已有容器
内直接运行代码的路线，不需要上传训练镜像或运行 Docker-in-Docker。训练镜像不是最终参赛镜像。

## 发布与平台提交边界

`v0.0.1` 模型发布到公开 HF 仓库 `aimmetal-tech/emotion-companion-7b`，GitHub Release 附件
是轻量部署包，不含完整权重、训练数据、优化器或任何令牌。模型卡和 Release 说明应区分
量化适配器的快速评估、本机 BF16 导出检查、尚待平台验证的完整模型推理，不能混称官方成绩。
`scripts/package_release.py` 必须传入固定 HF commit，拒绝覆盖已有输出；`start.sh` 和
`run_inference.py` 在工程副本中保持原始内容，仅调整配置。完整平台说明见
`docs/releases/v0.0.1/README.md`。

平台上传大赛镜像可能直接进入测评并归档、覆盖文件，不能把“发布 GitHub/HF”当成比赛提交。
将模型实际放在 `/root/participant/model/`，不要依赖 `/mnt/storage` 挂载或软链接随镜像保存。
只有用户明确授权时才可点击上传/触发正式测评；文档或模型发布不自动授权消耗成功测评机会。
