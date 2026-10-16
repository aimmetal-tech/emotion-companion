# 综合情感陪伴模型：20 卡时训练流程

以官方 PDF 为赛事信息依据；`official_dl/` 原始数据和推理框架保持原样。首轮使用
Qwen2.5-7B-Instruct，单卡 RTX 4090 24GB 做 QLoRA；当前官方启动脚本默认双卡推理，
正式提交的镜像、入口及硬件要求须在平台再次核实。

平台公有镜像实例的环境检查、文件上传和训练命令，见
[平台实例部署与训练操作手册](docs/平台实例部署与训练操作手册.md)。

## 目录与代码入口

| 路径 | 内容 |
| --- | --- |
| `docs/official_docs/` | 官方赛事 PDF；`docs/` 另含辅助整理与部署操作手册 |
| `official_dl/` | 官方原始训练/验证数据、推理框架、功能样例及实例创建说明 |
| `src/emotional_companion/` | 本项目数据处理、训练、评估、预算和导出代码 |
| `configs/train_7b.json` | 默认两轮、4096-token、单卡 QLoRA 训练配置 |
| `scripts/` | 本机冒烟、平台缺失依赖安装及 Docker 构建、运行、导出脚本 |
| `tests/` | 数据、指标、预算、运行脚本和安装脚本测试 |
| `artifacts/` | 处理后的数据、编码数据、校验报告及可选 Docker 归档 |
| `models/` | 基础模型 `qwen2.5-7b-instruct/`、导出模型 `companion-7b/` |
| `runs/` | CLI 默认的训练、评估、检查点、适配器和预算账本目录 |
| `remote-dl/runs/` | 本机下载的服务器产物与盲评，使用时须显式指定路径 |
| `pyproject.toml`、`uv.lock` | 项目入口、依赖范围及锁定版本 |
| `requirements.txt` | 从锁文件导出的训练依赖，供独立 pip 环境使用，不含项目本身 |

生成数据、模型、运行产物和缓存不在 Git 中；`remote-dl/runs/` 也匹配 `runs/` 忽略规则。
完整目录约定见 [AGENTS.md](AGENTS.md)。官方资料说明中的部分名称过时，以实际文件为准。

项目入口是 `src/emotional_companion/__main__.py` → `cli.main()`。以下命令均从项目根目录运行：

```bash
# 已经通过 uv 安装好项目和依赖时：
.venv/bin/python -m emotional_companion --help

# 复用已准备的 conda/Python 环境、不安装项目时：
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m emotional_companion --help
```

已安装项目也提供 `emotional-companion` 控制台入口。CLI 子命令及主要默认路径如下：

| 子命令 | 用途与主要路径 |
| --- | --- |
| `prepare` | `official_dl/train-val-dataset/` → `artifacts/data/` |
| `download` | 从 Hugging Face 下载基础模型到 `models/qwen2.5-7b-instruct/` |
| `tokenize` | `artifacts/data/` → `artifacts/tokens-4096/` |
| `smoke` | 微型端到端测试，默认 `runs/local-smoke/` |
| `train` | 默认使用 4096 编码数据，输出 `runs/train/`；`--resume` 恢复 |
| `evaluate` | 默认评估 `artifacts/data/val_quick.jsonl`；必须指定 `--output` |
| `score` | 对预测评分，必须指定 `--gold`、`--predictions`、`--output` |
| `review`、`compare` | 生成盲评与本地验收，默认 `runs/review/`、`runs/acceptance.json` |
| `export` | `runs/train/adapter/` → `models/companion-7b/` |
| `bundle` | 完整导出模型与官方框架 → `runs/submission/participant/` |
| `verify-results` | 验证官方输出格式、ID 覆盖与 100 条计时，必须指定 `--result` |
| `budget` | `status`、`record`、`run`、`reconcile`；默认账本 `runs/budget.json` |

全局 `--participant` 必须放在子命令之前；默认 `official_dl/participant/`。
`--output` 等参数属于各自子命令。源码布局下的 `No module named emotional_companion`
通常表示没有安装项目或没有设置上述 `PYTHONPATH`，不代表检查点损坏。

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
HF 镜像站仅参与下载，训练和编码仍从本地文件离线加载。

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

仅当目标主机支持 Docker 和自定义训练镜像时，才保存并加载镜像，再单独传输原始数据及
需要的模型、处理后数据、检查点和同一份预算账本。平台大赛公有镜像内直接运行代码的路线
不需要上传本训练镜像，也不需要 Docker-in-Docker；参见部署手册。

```bash
bash scripts/docker_export_training.sh
# 把 tar 文件复制到目标机器后：
docker image load -i emotional-companion-training-7b-cu124.tar
```

导出脚本默认保存到 `artifacts/docker/emotional-companion-training-7b-cu124.tar`，拒绝覆盖
已有文件并打印 SHA-256，不自动写 `.sha256` 文件；可通过位置参数指定其他目标路径。
归档只包含镜像，挂载目录需要另行复制。

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

锁定训练路线使用 Python 3.12 和 uv，PyTorch 2.6/CUDA 12.4，需要兼容的 NVIDIA 驱动。
依赖范围由 `pyproject.toml` 定义、具体版本由 `uv.lock` 固定；平台官方推理环境独立管理，
不要为了匹配训练依赖直接升级或降级预装 vLLM 的环境。

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --locked --extra training
uv run --no-sync python -m emotional_companion prepare
uv run --no-sync python -m emotional_companion download
uv run --no-sync python -m emotional_companion tokenize
```

不使用 uv 时，可在独立 Python 3.12 环境中安装 `requirements.txt`，再安装项目本身：

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

不要把这组全量安装命令直接用于平台预装的 `conda_env`，否则会安装锁定的 PyTorch 2.6。
`requirements.txt` 不含项目代码，也不包含 Ruff；权威依赖仍是 `pyproject.toml`/`uv.lock`。
需要重新导出时使用文件头记录的命令：

```bash
uv export --frozen --extra training --no-dev --no-emit-project --no-hashes --no-annotate -o requirements.txt
```

### 复用平台 conda 环境：只安装缺失包

先激活目标环境，再预览安装计划；脚本默认读取项目根目录的 `requirements.txt`：

```bash
python scripts/install_missing_requirements.py --dry-run
python scripts/install_missing_requirements.py --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

脚本依赖 `packaging` 解析版本和环境标记；平台训练环境通常已有它，缺失时会提示，
不会自动向未知环境引入引导依赖。同名包按规范化名称匹配，忽略大小写及 `-`、`_`、`.` 差异。
**已有包即使版本不符也跳过**，只提示差异；安装缺失包时强制 `--no-deps`，不让 pip 间接
替换已安装依赖。安装后运行 `pip check`，冲突时非零退出、不自动修复或降级。该脚本只支持
逐行命名的 PEP 508 依赖及环境标记，不支持 `-r`、`-e`、索引选项或 URL 安装源。
任意 requirements 文件未列全的传递依赖不会自动补装；可用 `--requirements PATH` 指定其他清单。

本次平台实际复用了 `conda_env` 的 PyTorch 2.10/CUDA 12.8、Transformers 4.57.6，补装
PEFT 0.18.1、Accelerate 1.15.0、bitsandbytes 0.49.2 后完成冒烟、7B 训练和快速评估。
这是已验证的另一组环境，不是当前锁文件环境。复用时先检查版本和导入、确认 GPU 可计算，
再短测，不要直接用 `uv sync` 改写平台环境。该路线设置前述 `PYTHONPATH`，后面的
`uv run --no-sync python` 和 `.venv/bin/python` 均替换为激活环境中的 `python`；预算执行器
和它启动的子命令必须使用同一训练环境。

uv/pip 命令依据：[uv 同步说明](https://docs.astral.sh/uv/concepts/projects/sync/)、
[pip 安装说明](https://pip.pypa.io/en/stable/cli/pip_install/)。

请在开云端 GPU 前完成安装、下载和编码。项目下载命令会固定到实际模型 commit，
并可在同一目录续传。`download` 子命令只使用 Hugging Face Hub（可通过 `HF_ENDPOINT`
使用镜像）；ModelScope 是另一条外部下载路线，不是该子命令的选项。外部下载到相同模型
目录后无需再次执行 `download`；先核对权重、配置和 tokenizer 与编码/训练使用的版本一致。
训练、评估和编码固定 `PYTHONHASHSEED=42`，与后面的官方入口命令一致，避免官方提示中集合
枚举顺序随进程变化。所有训练和推理模型加载都使用本地文件。

数据输出在 `artifacts/data/`，包括训练、开发集、固定 128/512/2354 条验证集、30 条人工复核集，
以及不含标签的官方推理输入。编码输出在 `artifacts/tokens-4096/`。`prepare`、`tokenize`、
`evaluate`、`export` 和 `bundle` 的目标目录必须不存在或为空，避免静默覆盖既有实验；
重新准备或评估时使用新的 `--output`。`review` 的目标目录必须不存在，
`train` 使用非空目录时必须显式 `--resume`。

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
`--hours` 是本次命令最长墙钟时间，不是必须训练满的时间；例如两轮完成后会提前退出。
**命令结束不会关闭云实例。GPU 实例的安装、等待、空闲时间也可能计费，需按平台规则停止 GPU
或手动记账；无卡模式是否另收存储等费用，以平台计费说明为准。**
账本不是平台账单，开始前必须确认免费额度、抵扣方式和实际剩余额度，不要在两台实例分别使用独立账本。

```bash
uv run --no-sync python -m emotional_companion budget status
# 仅当实际发生了这笔用量时执行：单卡实例在命令外开机了 6 分钟，记入 0.1 卡时。
uv run --no-sync python -m emotional_companion budget record \
  --stage recovery --gpu-hours 0.1 --note 'single GPU idle/setup: 6 minutes'

# 50 个优化器步冒烟测试；观察 throughput.json 和峰值显存。
uv run --no-sync python -m emotional_companion budget run \
  --stage baseline_smoke --gpus 1 --hours 0.5 -- \
  .venv/bin/python -m emotional_companion train --max-steps 50 --output runs/smoke-50

# 短测成功后跑基线，最多半小时。与短测共享本阶段 1 卡时额度。
uv run --no-sync python -m emotional_companion budget run \
  --stage baseline_smoke --gpus 1 --hours 0.5 -- \
  .venv/bin/python -m emotional_companion evaluate --output runs/baseline-quick

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
还需保持基础模型解析路径、tokenization 清单和原来的 `--max-steps` 设置不变；不要拿 50 步
短测目录接着做两轮正式训练。按一次 `Ctrl+C` 会尝试在优化器步结束时保存退出，不是挂起；
等待报告和检查点写完后再关机。预算执行器等待约 50 秒后仍未退出会强制结束，因此只能保证
从最近一次完整检查点恢复，不保证保存到按键时的最后进度。仅关闭终端时应分离 tmux，
不要按 `Ctrl+C`；tmux 不保护实例重建或抢占。
若 wrapper 被强制杀死，账本保留整笔预留；确认实例和进程停止、核对实际用量后执行
`budget reconcile --id ENTRY_ID --gpu-hours ACTUAL_USAGE`。显存不足时重新运行
`tokenize --max-length 2048 --output artifacts/tokens-2048`，随后在新目录训练：
`train --data artifacts/tokens-2048 --max-length 2048 --output runs/train-2048`。
不要把长度不同的数据用于旧检查点恢复。

## 效果检查、导出与提交验证

快速评估输出严格原始 JSON 合规率、16 类情绪宏平均 F1、画像分组微/宏 F1 和空画像命中率；
缺失预测按失败计算。`local_selection_score` 是情绪宏 F1 与三类画像平均微 F1 的均值，
不是官方比赛分数。每条生成的原始文本、解析状态和耗时保存在 `raw.jsonl`，官方归一化后的结果另存为
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
7B 基础权重约 15GB，合并模型与提交副本各再增加约 15GB，另需环境、检查点和缓存空间；
规划总占用约 50–70GB 并留出余量，不代表脚本会自动申请或检查平台个人配额。

### 本机处理已下载的远程结果

当前备份布局为 `remote-dl/runs/`，不是 `runs/`。CLI 不会自动发现备份或自动更改输出目录。
在已准备好锁定环境的本机项目根目录，生成盲评时使用：

```bash
.venv/bin/python -m emotional_companion review \
  --baseline remote-dl/runs/baseline-quick/submission.jsonl \
  --candidate remote-dl/runs/candidate-quick/submission.jsonl \
  --output remote-dl/runs/review
```

如果 `remote-dl/runs/review/review.json` 已存在，直接完成里面的人工评分，不要重复生成或覆盖。
填完全部 30 条、A/B 两组各四项 1–5 整数评分后，显式指定盲评和验收报告路径：

```bash
.venv/bin/python -m emotional_companion compare \
  --baseline remote-dl/runs/baseline-quick/metrics.json \
  --candidate remote-dl/runs/candidate-quick/metrics.json \
  --review remote-dl/runs/review \
  --output remote-dl/runs/acceptance.json
```

人工盲评是内部验收，不是官方提交材料或已经确认的官方评分要求。自动分类指标、回复质量
和推理延迟应分别检查，不能用结构化指标提升代替人工回复评价。

后续在本机 CPU 合并时，基础模型仍使用本机 `models/`，适配器必须指向备份路径。
先确认内存足够、输出目录不存在或为空，以下命令不会使用远程 GPU 卡时：

```bash
CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m emotional_companion export \
  --adapter remote-dl/runs/train/adapter --output models/companion-7b
.venv/bin/python -m emotional_companion bundle --model models/companion-7b
```

原始 `remote-dl/runs/budget.json` 应保留为迁移后的同一份账本；不要在本机用默认空白
`runs/budget.json` 重新申领 20 卡时。需要在备份账本上查询时用
`.venv/bin/python -m emotional_companion budget --ledger remote-dl/runs/budget.json status`。

### 官方推理入口验证

在确认平台实际入口和镜像后，将副本放到 `/root/participant`，在已准备好官方
`conda_env` 和兼容 vLLM 的双卡实例中执行：

```bash
# 从单卡训练切换双卡验证时，应在确认实例实际分配两卡后设置，不能继承单卡屏蔽值。
export CUDA_VISIBLE_DEVICES=0,1
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
