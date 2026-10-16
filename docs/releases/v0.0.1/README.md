# v0.0.1：平台部署、实例镜像上传与首次测评

这是首版提交候选，不代表已经通过平台官方测评或人工回复质量验收。
本文依据 `official_dl/README.md`、`participant/README.md`、实例创建 PDF，以及用户提供的
“上传大赛镜像”页面。最终仍以平台当前要求为准。

## 1. 发布内容与目录

- 模型：<https://huggingface.co/aimmetal-tech/emotion-companion-7b/tree/v0.0.1>
- GitHub：<https://github.com/aimmetal-tech/emotion-companion/releases/tag/v0.0.1>
- GitHub 附件 `emotion-companion-v0.0.1-deploy.tar.gz` 是轻量推理部署包，**不含约 15GB 模型权重**。
- 包中的 `model_source.json` 固定 HF commit 和每个文件的 SHA-256，不跟随 `main` 漂移。
- GitHub 自动生成的 Source code 压缩包是源码，不是准备好的参赛镜像。

解压后：

```text
emotion-companion-v0.0.1/
├─ participant/                 # 官方入口和推理程序；只调整副本配置，默认 vLLM 双卡
├─ tools/                       # 固定版本模型下载/校验、只安装缺失包脚本
├─ src/emotional_companion/     # 预算与结果自检等 CLI 源码
├─ model_source.json            # 模型仓库、固定 commit、文件哈希
├─ test_inference_data.jsonl    # 官方 3 条功能样例
├─ validation_128.input.jsonl   # 128 条无标签公开验证输入
├─ reports/                     # 训练、量化快速评估及本机导出检查记录
├─ release_manifest.json        # 版本、源代码 commit、官方脚本哈希、待验证事项
├─ SHA256SUMS                   # 部署包内容校验（下载模型另行校验）
├─ LICENSE.model                # 上游模型 Apache-2.0 许可证
└─ DEPLOY.md                    # 本说明
```

完整模型是合并后的 BF16 模型，不是 LoRA 适配器。无需在平台上再次训练或合并。
当前快速评估的 100% JSON 合规率、0.6999 情绪 Macro-F1 来自量化模型 + 适配器；
不能将其直接称为本合并模型的官方成绩。

## 2. 准备平台实例（可先无卡）

1. 选择赛事要求的 vLLM 公有镜像；本次使用的是 `0.19.0-ubuntu22.04-p3.12-cuda12.8`。
2. 保留 `/mnt/storage` 存储路径，但确认个人配额和持久化规则；`df` 的共享盘总余量不是个人配额。
3. 默认部署采用官方脚本的双卡张量并行，因此 GPU 自测时需要两卡；不要用单卡配置直接启动它。
4. 不需要新建公网推理端口，脚本内部使用 `127.0.0.1:8000`。SSH 上传另按平台连接说明。
5. 不导入我们之前构建的训练 Docker 镜像，也不在平台容器里运行 Docker-in-Docker。

先把部署包和同名 `.sha256` 校验文件下载/上传到服务器的 `/mnt/storage/release-v0.0.1/`。
两份文件均可从 GitHub Release 页面下载；这一步只传文件，不是提交参赛作品。

```bash
cd /mnt/storage/release-v0.0.1
sha256sum -c emotion-companion-v0.0.1-deploy.tar.gz.sha256
tar -xzf emotion-companion-v0.0.1-deploy.tar.gz
cd emotion-companion-v0.0.1
sha256sum -c SHA256SUMS
```

同名解压目录存在时不要直接覆盖，先保留旧版本。

## 3. 把推理工程和模型真正放进实例镜像

**不能只把模型放在 `/mnt/storage` 或用软链接指向它。** 平台并未在这些资料中承诺挂载存储
会随实例镜像一起打包。为保证离线可用，将工程和模型实际放在 `/root/participant`。

先检查现有目录和系统盘空间；模型本身约 15GB，建议至少留出 20GB，另预留运行缓存空间：

```bash
ls -ld /root/participant
df -h /root /mnt/storage
```

`ls` 提示目录不存在是正常的。若已经存在，先备份、核对并改名；不要直接覆盖或删除旧版本。
确认目标不存在后执行：

```bash
test ! -e /root/participant && cp -a participant /root/participant

source /opt/conda/etc/profile.d/conda.sh
conda activate conda_env

python tools/install_missing_requirements.py --requirements /root/participant/requirements.txt --dry-run
python tools/install_missing_requirements.py --requirements /root/participant/requirements.txt \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

脚本不替换已有包，也不保证不同版本组合兼容；`pip check` 报冲突时先排查，不要盲目降级
平台 PyTorch、Transformers 或 vLLM。完整训练依赖不需要再次安装。

公开模型下载不需要在服务器保存 HF Token。使用固定 commit 下载和逐文件校验：

```bash
HF_ENDPOINT=https://huggingface.co HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  HF_HUB_DOWNLOAD_TIMEOUT=60 \
  python tools/download_release_model.py --output /root/participant/model
```

官方站连接失败时，可用公开镜像重试；镜像的新仓库/版本可能同步延迟，不保证立即可用：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DOWNLOAD_TIMEOUT=60 \
  python tools/download_release_model.py --output /root/participant/model \
  --endpoint https://hf-mirror.com
```

下载脚本始终使用 `token=False`，不会把 HF Token 发给镜像站。中断后重复同一命令可续传；
如果目标目录不是本下载任务创建的，脚本拒绝混写，不能删除来源标记来绕过检查。
也可以从本机上传合并模型到一个新目录，再使用 `--verify-only --output PATH` 校验。

```bash
python tools/download_release_model.py --verify-only --output /root/participant/model
test ! -L /root/participant/model
du -sh /root/participant/model
bash -n /root/participant/start.sh
```

不要把本机 HF Token、GitHub 凭据、SSH 私钥或账号密码放进参赛镜像。

## 4. 固定推理环境，切换 GPU 自测

官方 `start.sh` 会自行激活 `conda_env`。为了在平台直接启动脚本时也固定提示词枚举顺序，
把离线标志和哈希种子存入该 conda 环境，并重新激活：

```bash
conda env config vars set -n conda_env PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
conda deactivate
conda activate conda_env
```

只持久化这些无敏感值的变量；不要把 Token 存入环境配置。参见
[Conda 环境变量说明](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables)。

默认路线：确认平台分配的是两卡，再在当前终端设置：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export EVALUATION_HARDWARE_LABEL=2xRTX4090-24GB
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.device_count()); assert torch.cuda.device_count() == 2'
vllm --version
```

先用官方 3 条样例做功能检查，不消耗官方“成功测评机会”，但 GPU 开机会消耗卡时：

```bash
bash /root/participant/start.sh "$PWD/test_inference_data.jsonl" "$PWD/checks/functional-3"
```

应输出 `submission.jsonl` 和 `performance_report.json`。3 条样例不足以完成 100 条性能统计，
其报告 `complete: false` 不代表功能失败。出现模型加载、显存、JSON 解析或环境错误先解决，
不要立刻点击平台上传。

### 如果暂时只有一张 GPU

可以在工程副本的 `configs/inference_config.json` 中将 `backend` 改为 `transformers`，
`hardware_label` 改为 `1xRTX4090-24GB`，并设置 `CUDA_VISIBLE_DEVICES=0` 做单卡功能检查。
这不需要修改官方脚本；但长上下文可能显存不足，也不能据此声称双卡 vLLM 已验证。
正式提交采用哪条路线必须符合平台当前规则。更换配置后重新验证，并记录与默认 Release 的差异。

## 5. 128 条官方入口计时和结果检查

下面默认仍是两卡 vLLM 路线。首次测评必须使用完整已校验的合并模型，不是微型测试权重。
复用训练时的同一份账本，不要新建空白账本；若账本在其他位置，修改 `EC_LEDGER`：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export EC_LEDGER=/mnt/storage/emotional-companion/runs/budget.json
test -f "$EC_LEDGER"
python -m emotional_companion budget --ledger "$EC_LEDGER" status
```

确认验证阶段至少还允许 1 卡时、平台实际额度也足够，再执行最多半小时的双卡验证：

```bash
python -m emotional_companion budget --ledger "$EC_LEDGER" run \
  --stage verify --gpus 2 --hours 0.5 -- \
  bash /root/participant/start.sh "$PWD/validation_128.input.jsonl" "$PWD/checks/official-128"

python -m emotional_companion --participant /root/participant verify-results \
  --inputs validation_128.input.jsonl --result checks/official-128
```

`verify-results` 要求 ID 完整、字段有效、没有 JSON 解析失败、100 条计时完成、报告哈希匹配。
这是项目内部自检，不是官方成绩或官方性能阈值判定。保留整份 `checks/` 和终端日志；原先的
量化快速评估延迟不能替代这里的完整模型/vLLM 计时。准备和空闲用量另按实际记账。

## 6. 上传大赛镜像前的最后检查

全部确认后再进入平台“上传大赛镜像”：

- 原始官方 `start.sh`、`run_inference.py` 内容未改，入口仍在 `/root/participant/start.sh`。
- `/root/participant/model/` 是实际文件目录，包含完整权重、配置、tokenizer 和许可证。
- 所需包已经安装在 `conda_env`；不依赖本机 `.venv`、在线 API 或临时下载。
- 配置 `backend`、真实硬件、模型路径与刚才自测一致；没有 Token、私钥或密码。
- GPU 入口自测和输出检查通过，没有仍在运行的训练/推理服务占卡。
- 已在本机备份原始 `runs/`、适配器、检查点、账本、自测结果，以及当前配置和版本清单。
- 已查看队伍当前成功测评次数、剩余卡时及截止时间。

尤其注意截图提示：**作品提交后进入测评，参赛文件会归档，原存储位置归档后无法查看，
再次上传覆盖原参赛文件且无法恢复。** 不要只依赖平台上的唯一一份文件，也不要把“上传”
当作无副作用的保存草稿操作。

## 7. 页面上传与查看测评

1. 在实际部署并自测的实例详情页，核对公有镜像和实例，再点击 **“上传大赛镜像”**。
2. 弹窗选择 **“2026动感地带AI+高校创智大赛”**，再次核对归档/覆盖警告。
3. 确认全部备份和自测通过后，才点击 **“上传”**。这里上传的是平台实例参赛镜像，
   **不是把 GitHub 的部署 tar.gz 或训练镜像 tar 当作参赛镜像上传**。
4. 记录平台返回的作品/镜像标识、时间和状态。到比赛作品页核对队伍、赛题和所选镜像；
   若页面另有“提交参赛作品”步骤，按页面完成。若已经进入测评，不重复点击提交。
5. 确认安全审核通过、测评成功并保存成绩/日志截图；不要仅凭上传进度 100% 判断参赛成功。
6. 失败时根据日志修复后再提交，不要改动本地 `v0.0.1` tag；新候选另发新版本。

官方计划书规定最多 5 次成功测评，运行失败不占成功机会；最终按最后一次成功提交计分，
不是取历史最高分。具体以平台最新规则为准。GitHub Release 或 HF 上传本身不是比赛提交。

本版没有代你创建付费 GPU 资源、上传大赛镜像或触发官方测评。
