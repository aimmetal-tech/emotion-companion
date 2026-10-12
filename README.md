# 综合情感陪伴模型：20 卡时训练流程

以官方 PDF 为赛事信息依据；`official_dl/` 原始数据和推理框架保持原样。首轮使用
Qwen2.5-7B-Instruct，单卡 RTX 4090 24GB 做 QLoRA，最终在双卡环境验证离线提交。

## Docker 训练镜像

训练镜像的 Python/uv 基础镜像固定 digest，Python 依赖严格使用现有 `uv.lock`。
PyTorch 2.6 的锁定 wheel 包含 CUDA 12.4 运行库；宿主机仍需兼容的 NVIDIA 驱动和 GPU
容器支持。镜像内置 GCC 和 C 头文件，支持 Triton 在断网时编译 CUDA 驱动辅助模块。
镜像面向 Linux AMD64，包含训练代码、配置、测试及官方推理框架，不包含模型
权重、训练数据、本机虚拟环境、缓存或账号凭据。

```bash
bash scripts/docker_build_training.sh
# 在容器内实际验证 CUDA/QLoRA；默认断网，不下载任何模型。
bash scripts/docker_training.sh smoke --device cuda --output runs/docker-smoke
# 无 GPU 也可以验证普通 LoRA 流程。
bash scripts/docker_training.sh --cpu smoke --device cpu --output runs/docker-smoke-cpu
```

镜像名默认是 `emotional-companion-training:7b-cu124`，可用 `TRAINING_IMAGE` 覆盖。
运行脚本默认暴露 GPU 0；只读挂载官方数据，并把 `artifacts/`、`models/`、`runs/`
和 `.cache/docker-training/` 挂载到容器对应目录。模型、检查点和 `runs/budget.json`
在容器退出后仍保留。容器使用当前用户的 UID/GID，输出文件沿用主机当前用户的所有权。
运行脚本不挂载整个仓库，代码来自已构建的镜像，修改代码后需重新构建。

CPU 准备及正式训练示例：

```bash
bash scripts/docker_training.sh --cpu prepare --output artifacts/docker-data
# download 默认允许联网，其余命令默认 network=none。
bash scripts/docker_training.sh --cpu download
bash scripts/docker_training.sh --cpu tokenize \
  --data artifacts/docker-data --output artifacts/docker-tokens

bash scripts/docker_training.sh budget run \
  --stage train --gpus 1 --hours 9 -- \
  python -m emotional_companion train --data artifacts/docker-tokens --output runs/docker-train
```

若本机访问 Hugging Face 不稳定，可在下载时使用镜像站（保持原有固定 commit 和续传目录）：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  bash scripts/docker_training.sh --cpu download
```

运行脚本会透传这些环境变量，不需要重建镜像；`HF_HUB_DISABLE_XET=1` 使用普通 HTTP 下载，
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` 让公开模型下载不自动携带本机保存的 Hugging Face 令牌。
镜像只用于下载，训练和编码仍从本地文件离线加载。

本机已安装 hfd 时，也可按相同固定版本检查或补齐模型文件：

```bash
HF_ENDPOINT=https://hf-mirror.com hfd Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-dir models/qwen2.5-7b-instruct \
  --include '*.json' '*.safetensors' '*.txt' '*.model' '*.jinja' 'README.md' 'LICENSE' \
  -x 4 -j 4
```

不要同时用两个下载器写同一模型目录。SDK 的 `.incomplete` 与 aria2 的续传记录格式不同，
切换工具前应保留已有进度，不能直接把残缺缓存当作完整权重文件。

`TRAINING_WORKSPACE` 可指定存放输出、权重和账本的主机目录；`TRAINING_DATA_DIR` 可指定
官方训练数据目录；`TRAINING_NETWORK` 和 `TRAINING_OFFLINE` 可显式覆盖网络/离线设置。
`TRAINING_DRY_RUN=1` 只打印运行命令，不创建目录或启动容器。需要访问不同 GPU 时设置
`TRAINING_GPUS=device=1`；需要同时暴露双卡时设为 `TRAINING_GPUS=all`，并在预算命令中
声明 `--gpus 2`。主训练仍只暴露单卡。镜像内的 `.venv` 链接指向 `/opt/venv`，也兼容下面的
`.venv/bin/python` 命令。

转移到平台机器时，保存并加载镜像，再单独传输原始数据及 `models/`、`artifacts/`、`runs/`
目录；尤其要保留同一份预算账本。

```bash
bash scripts/docker_export_training.sh
# 把 tar 文件复制到目标机器后：
docker image load -i emotional-companion-training-7b-cu124.tar
```

导出脚本默认保存到 `artifacts/docker/emotional-companion-training-7b-cu124.tar`，拒绝覆盖
已有文件并打印 SHA-256；可通过位置参数指定其他目标路径。归档只包含镜像，挂载目录需要另行复制。

镜像使用内置代码，无需在目标机器安装 Python 依赖。也可运行
`docker run --rm --gpus all --entrypoint bash emotional-companion-training:7b-cu124`
进入命令行（交互运行时添加 `-it`），但训练输出应使用上述挂载脚本保存。
本镜像用于训练与本地推理调试，不是基于组委会基础镜像构建的最终提交镜像，未安装 vLLM
或平台 `conda_env`。平台是否支持导入自定义训练镜像，仍需核对其实际创建实例流程。

构建需要联网。锁文件使用清华 PyPI 镜像，构建时同步沿用该源；不要直接换源导致重新解析。
若构建网络需要代理，可给构建脚本传入 `--build-arg HTTP_PROXY=...` 和
`--build-arg HTTPS_PROXY=...`；代理必须能从构建容器访问，不应填写主机的 `127.0.0.1`。
构建代理只作用于构建阶段，运行镜像中不保留代理值。

## 先在本机做小测试

你的 RTX 4060 8GB 可以运行下面的微型模型测试，无需下载 7B 权重：

```bash
bash scripts/smoke_local.sh --device cuda
```

首次执行会通过 uv 安装 PyTorch/CUDA、Transformers、PEFT 和 bitsandbytes，依赖下载约数 GB。
脚本在本地生成约十万参数的随机 Qwen2 和字节级 tokenizer，验证目标损失掩码、真实 4-bit
QLoRA 训练、完整检查点恢复、适配器推理、BF16 合并与离线重载，然后用未修改的官方
`run_inference.py` 完成 100 条推理并检查结果和计时报告。模型输出没有实际对话质量。

结果保存在 `runs/local-smoke/smoke_report.json`，失败时保留中间文件和错误。重跑请指定新目录：

```bash
bash scripts/smoke_local.sh --device cuda --output runs/local-smoke-2
bash scripts/smoke_local.sh --device cpu --output runs/local-smoke-cpu
```

CPU 模式验证普通 LoRA 流程，不覆盖 bitsandbytes 的 CUDA 量化路径。单卡本机也无法验证平台的
`conda_env`、vLLM 服务和双卡张量并行；这些仍需在平台验证，脚本会列出未覆盖项目。

## 环境与 CPU 准备

项目需要 Python 3.12 和 uv。训练环境选择支持 CUDA 12.4 的 NVIDIA 驱动；依赖锁定在
`uv.lock`，PyTorch 限定为 2.6 系列，避免自动安装另一个 CUDA 主版本。平台官方推理环境
独立于训练环境，不要求把原始 Python 3.10 镜像升级成训练环境。

```bash
uv sync --extra training
uv run --no-sync python -m emotional_companion prepare
uv run --no-sync python -m emotional_companion download
uv run --no-sync python -m emotional_companion tokenize
```

请在开云端 GPU 前完成安装、下载和编码。下载固定到实际模型 commit，并可在同一目录续传。
训练、评估和编码固定 `PYTHONHASHSEED=42`，与后面的官方入口命令一致，避免官方提示中集合
枚举顺序随进程变化。所有训练和推理模型加载都使用本地文件。

数据输出在 `artifacts/data/`，包括训练、开发集、固定 128/512/2354 条验证集、30 条人工复核集，
以及不含标签的官方推理输入。编码输出在 `artifacts/tokens-4096/`。生成目录必须为空，避免
静默覆盖既有实验；重新准备时使用新的 `--output`。

当前数据审计：202 段重复完整对话；4 个预测点的目标回复早于目标用户轮，已排除并记录在
`manifest.json`。去重、分组和隔离后有 17,197 条训练样本、948 条内部开发样本；官方验证集
保留全部 2,354 条。相同预测历史归入同一集合；训练输入不含未来回复。`memory_refs=[]`，
不学习公开输入无法推断的记忆编号。

## 20 卡时预算和 GPU 命令

| 阶段 | 最大 GPU 卡时 |
| --- | ---: |
| 基线与冒烟 | 1 |
| 主训练 | 9 |
| 评估 | 3 |
| 合并导出 | 1 |
| 双卡提交验证 | 4（双卡运行 2 小时） |
| 故障恢复 | 2 |

预算执行器预留额度、设置进程超时并清理子进程，结束后按实际运行时长计入账本。同一阶段可多次
运行，但合计不得超过阶段上限。所有任务累计最多 20 卡时；达到 18 卡时后只允许导出、验证和恢复。
**命令结束不会关闭云实例。实例的安装、等待、空闲时间也会计费，需关机或先手动记账。**
账本不是平台账单，开始前必须确认免费额度、抵扣方式和实际剩余额度，不要在两台实例分别使用独立账本。

```bash
uv run --no-sync python -m emotional_companion budget status
# 示例：单卡实例在命令外开机了 6 分钟，先计入 0.1 卡时。
uv run --no-sync python -m emotional_companion budget record \
  --stage recovery --gpu-hours 0.1 --note 'single GPU idle/setup: 6 minutes'

# 基线：最多半小时。真实结束更早时，会释放尚未消耗的额度。
uv run --no-sync python -m emotional_companion budget run \
  --stage baseline_smoke --gpus 1 --hours 0.5 -- \
  .venv/bin/python -m emotional_companion evaluate --output runs/baseline-quick

# 50 个优化器步冒烟测试；观察 throughput.json 和峰值显存。
uv run --no-sync python -m emotional_companion budget run \
  --stage baseline_smoke --gpus 1 --hours 0.5 -- \
  .venv/bin/python -m emotional_companion train --max-steps 50 --output runs/smoke-50

# 主训练：最多 9 小时（单卡），自动提前留出保存和关停时间。
uv run --no-sync python -m emotional_companion budget run \
  --stage train --gpus 1 --hours 9 -- \
  .venv/bin/python -m emotional_companion train

# 固定快速评估集，同基线的量化方式、提示和生成配置。
uv run --no-sync python -m emotional_companion budget run \
  --stage eval --gpus 1 --hours 1 -- \
  .venv/bin/python -m emotional_companion evaluate \
  --adapter runs/train/adapter --output runs/candidate-quick

# 扩展评估，剩余额度允许时执行。
uv run --no-sync python -m emotional_companion budget run \
  --stage eval --gpus 1 --hours 2 -- \
  .venv/bin/python -m emotional_companion evaluate \
  --adapter runs/train/adapter --data artifacts/data/val_extended.jsonl \
  --output runs/candidate-extended
```

`throughput.json` 在第 50 步给出一轮训练耗时估算。默认最多两轮，实际由 9 卡时限制决定。
训练配置见 `configs/train_7b.json`：4-bit NF4、BF16、rank 16、全部线性层、学习率 `1e-4`、
批量 1、梯度累积 16、长度 4096，仅对目标 JSON 和结束标记计算损失。每 250 步用固定的
64 条内部开发样本检查损失，保留最佳和最近检查点；`adapter/` 是 Trainer 选定的适配器。

中断后使用同一数据和配置、`train --resume latest` 恢复，预算 `--hours` 不得超过该阶段剩余额度。
若 wrapper 被强制杀死，账本保留整笔预留；确认实例和进程停止、核对实际用量后执行
`budget reconcile --id ENTRY_ID --gpu-hours ACTUAL_USAGE`。显存不足时重新运行
`tokenize --max-length 2048 --output artifacts/tokens-2048`，随后在新目录训练：
`train --data artifacts/tokens-2048 --max-length 2048 --output runs/train-2048`。
不要把长度不同的数据用于旧检查点恢复。

## 效果检查、导出与提交验证

快速评估输出严格原始 JSON 合规率、16 类情绪宏平均 F1、画像分组微/宏 F1 和空画像命中率；
缺失预测按失败计算。`local_selection_score` 是情绪宏 F1 与三类画像平均微 F1 的均值，
不是官方比赛分数。生成失败的原始文本保存在 `raw.jsonl`，官方归一化后的结果另存为
`submission.jsonl`，避免格式修复掩盖模型错误。

```bash
uv run --no-sync python -m emotional_companion review \
  --baseline runs/baseline-quick/submission.jsonl \
  --candidate runs/candidate-quick/submission.jsonl
```

请盲评 `runs/review/review.json` 的 30 条回复，对共情、连贯性、偏好遵循、事实准确性分别填写
1–5 分；评价完成前不要查看 `key.json`。填写完成后：

```bash
uv run --no-sync python -m emotional_companion compare \
  --baseline runs/baseline-quick/metrics.json --candidate runs/candidate-quick/metrics.json

uv run --no-sync python -m emotional_companion budget run \
  --stage export --gpus 1 --hours 1 -- \
  .venv/bin/python -m emotional_companion export
uv run --no-sync python -m emotional_companion bundle
```

内部合格条件：快速集原始字段合规率至少 99%、结构化诊断分数高于基线、人工回复评价不下降，
且所有预测完整。`compare` 检查报告和人工复核来自同一批结果。未达标时仍保留实验产物，
不能把流程成功视为效果合格。

导出会将适配器合并到重新加载的原始 BF16 权重，保存完整 tokenizer、权重和配置并离线重载。
`bundle` 复制官方框架到 `runs/submission/participant/`，仅调整副本的配置，原始框架不改动。
导出和打包通常需约 50–70GB 空闲磁盘（基础模型、合并权重、提交副本及运行缓存）。

在确认平台实际入口和镜像后，将副本放到 `/root/participant`，在已准备好官方
`conda_env` 和兼容 vLLM 的双卡实例中执行：

```bash
uv run --no-sync python -m emotional_companion budget run \
  --stage verify --gpus 2 --hours 2 -- \
  env PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash /root/participant/start.sh \
  artifacts/data/val_quick.input.jsonl runs/platform-validation
uv run --no-sync python -m emotional_companion verify-results --result runs/platform-validation
uv run --no-sync python -m emotional_companion score \
  --gold artifacts/data/val_quick.jsonl \
  --predictions runs/platform-validation/submission.jsonl \
  --output runs/platform-validation/local-metrics.json
```

正式入口只接收不含标签的输入。官方 3 条样例可先做功能检查，但不足以完成 100 条性能统计。
本轮不自动创建付费实例、上传作品或消耗正式测评机会。PDF 的 `/participant/run_all.sh`
与当前资料包的 `/root/participant/start.sh` 存在差异，发布前必须核对实际平台要求。

## 开发检查

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python -m unittest discover -s tests -v
```

Ruff 启用 `E,F,I,B,UP,SIM,C4,RUF`；官方下载目录不参与自动格式修改。
CPU 测试覆盖真实数据审计、未来信息隔离、目标损失掩码、长历史裁剪、少数情绪覆盖、
无效/缺失预测、人工复核完整性、预算超限/中断/用量恢复，以及官方计时完整性。
