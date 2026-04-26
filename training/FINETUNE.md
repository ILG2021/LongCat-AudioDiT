# LongCat-AudioDiT 微调指南

本文档介绍如何对 LongCat-AudioDiT 进行微调，以适配特定说话人、语言或领域。  
实现基于论文中的 **Conditional Flow Matching (CFM)** 目标函数，参考 F5-TTS 训练范式。

---

## 目录

1. [原理说明](#1-原理说明)
2. [新增文件说明](#2-新增文件说明)
3. [环境准备](#3-环境准备)
4. [数据准备](#4-数据准备)
5. [预编码缓存（推荐）](#5-预编码缓存推荐)
6. [配置文件](#6-配置文件)
7. [启动训练](#7-启动训练)
8. [显存估算](#8-显存估算)
9. [检查点与推理](#9-检查点与推理)
10. [高级选项](#10-高级选项)
11. [常见问题](#11-常见问题)

---

## 1. 原理说明

### 1.1 Conditional Flow Matching (CFM)

LongCat-AudioDiT 与 F5-TTS 使用完全相同的 CFM 目标函数，在 **Wav-VAE 潜变量空间**中进行训练：

$$x_t = \bigl(1 - (1-\sigma)\,t\bigr)\,x_0 \;+\; t\,x_1$$

$$\mathcal{L} = \mathbb{E}_{t,x_0,x_1} \left\| f_\theta(x_t,\, t,\, \text{text\_emb},\, z_\text{cond}) - \underbrace{(x_1 - (1-\sigma)\,x_0)}_{v_\text{target}} \right\|^2$$

| 符号 | 含义 |
|------|------|
| $x_0$ | 标准高斯噪声 |
| $x_1$ | Wav-VAE 编码的目标音频潜变量 |
| $x_t$ | 时刻 $t$ 的插值 |
| $\sigma$ | 噪声地板（训练时与预训练一致，默认 0） |
| $z_\text{cond}$ | prompt 音频潜变量（voice cloning 时使用） |
| $f_\theta$ | DiT Transformer（唯一被训练的组件） |

时间步采样使用 **logit-normal 分布**（`t = sigmoid(N(0,1))`），使采样集中在 $t \approx 0.5$ 处，与 F5-TTS 一致。

### 1.2 训练-推理不一致修正

论文核心贡献之一。推理时 prompt 区域每步被强制设为：

```
x[:, :p] = x0[:, :p] * (1 - t) + prompt_latent[:, :p] * t
```

训练时对 prompt 区域应用**相同的插值**，而非使用完整的 $x_t$，消除分布偏差。

### 1.3 冻结策略

| 子模块 | 状态 | 说明 |
|--------|------|------|
| `model.vae` | 🔒 冻结（fp16） | Wav-VAE，预编码后训练期间不调用 |
| `model.text_encoder` | 🔒 冻结（默认） | UMT5-base，可预编码至磁盘彻底释放显存 |
| `model.transformer` | 🔥 训练 | DiT backbone，唯一更新的组件 |

---

## 2. 新增文件说明

```
LongCat-AudioDiT/
├── prepare_data.py        # 从 WAV 目录生成 metadata CSV
├── cache_embeddings.py    # 离线预编码 VAE latent + UMT5 text embedding
├── finetune_dataset.py    # PyTorch Dataset / DataLoader
├── finetune.py            # CFM 训练主脚本
└── finetune_config.yaml   # 超参数配置
```

---

## 3. 环境准备

```bash
# 基础依赖（已在 requirements.txt 中）
pip install -r requirements.txt

# 额外依赖
pip install accelerate pyyaml tensorboard
```

> **Python ≥ 3.10，PyTorch ≥ 2.0，CUDA ≥ 11.8**

---

## 4. 数据准备

### 4.1 metadata 格式

训练脚本读取 CSV 或 JSON 文件，支持以下列：

| 列名 | 必填 | 说明 |
|------|------|------|
| `audio_path` | ✅ | WAV 文件绝对或相对路径（24 kHz 单声道）|
| `text` | ✅ | 对应文本转录 |
| `duration` | 可选 | 音频时长（秒），用于快速过滤 |
| `prompt_audio_path` | 可选 | Voice-cloning 参考音频路径 |
| `prompt_text` | 可选 | 参考音频对应文本 |

### 4.2 支持的输入格式

**格式 A — WAV + 同名 `.txt` 转录文件（最简单）**

```
data/my_speaker/
    001.wav
    001.txt   ← 内容：今天天气很好
    002.wav
    002.txt
```

```bash
# 在根目录下运行
python training/prepare_data.py \
    --wav_dir    data/my_speaker \
    --output     data/train.csv \
    --val_output data/val.csv \
    --val_ratio  0.05
```

**格式 B — WAV 目录 + 已有 CSV**

```bash
python training/prepare_data.py \
    --wav_dir  data/wavs \
    --metadata data/metadata.csv \
    --output   data/train.csv
```

**格式 C — LJSpeech 格式（竖线分隔）**

```bash
python training/prepare_data.py \
    --wav_dir  LJSpeech-1.1/wavs \
    --metadata LJSpeech-1.1/metadata.csv \
    --sep "|" --id_col 0 --text_col 2 \
    --output data/train.csv
```

**格式 D — 带固定 prompt 的 voice-cloning 微调**

```bash
python training/prepare_data.py \
    --wav_dir      data/my_speaker \
    --output       data/train.csv \
    --prompt_audio data/prompt.wav \
    --prompt_text  "参考音频对应的文本内容"
```

### 4.3 音频要求

| 参数 | 要求 |
|------|------|
| 采样率 | **24 000 Hz**（脚本会自动重采样） |
| 声道数 | 单声道（自动转换） |
| 时长 | 0.5 s – 15.0 s（默认，可在 config 中调整）|
| 格式 | WAV（librosa 支持的均可）|

---

## 5. 预编码缓存（推荐）

对于 **RTX 5090（32 GB）** 等显存充裕但仍希望最大化 batch size 的场景，  
推荐在训练前将 **Wav-VAE 潜变量**和 **UMT5 文本嵌入**全部离线编码写盘。  
训练时 GPU 上只保留 DiT Transformer，可节省约 **1.1 GB**（UMT5-base bf16）。

```bash
# 在根目录下运行
python training/cache_embeddings.py \
    --metadata   data/train.csv data/val.csv \
    --model_dir  meituan-longcat/LongCat-AudioDiT-1B \
    --latent_dir latent_cache \
    --text_dir   text_cache \
    --device     cuda
```

| 参数 | 说明 |
|------|------|
| `--metadata` | 支持多个文件同时处理 |
| `--latent_dir` | VAE latent 缓存目录（`<stem>.pt`，shape `(D, T)`）|
| `--text_dir` | text embedding 缓存目录（SHA-256 key，包含重复数据去重）|
| `--skip_latent` | 只编码文本 |
| `--skip_text` | 只编码 latent |
| `--device cpu` | 在 CPU 上编码（无 GPU 时使用）|

运行完成后，`training/finetune_config.yaml` 中设置：

```yaml
text_encoding_mode: "cached"
text_cache_dir:     "text_cache"
latent_dir:         "latent_cache"
```

---

## 6. 配置文件

完整配置项说明（`training/finetune_config.yaml`）：

```yaml
# ── 模型 ─────────────────────────────────────────────────────
model_dir:  "meituan-longcat/LongCat-AudioDiT-1B"  # 本地路径或 HF model id
output_dir: "finetune_output"

# ── 数据 ─────────────────────────────────────────────────────
train_metadata: "data/train.csv"
val_metadata:   "data/val.csv"
max_duration:   15.0   # 秒
min_duration:    0.5   # 秒

# ── 文本编码模式 ──────────────────────────────────────────────
# "cached" → 从磁盘读预编码嵌入（run cache_embeddings.py first）✅推荐
# "cpu"    → UMT5 在 CPU 上实时编码（节省 GPU 显存，轻微延迟）
# "gpu"    → UMT5 在 GPU 上实时编码（~1.1 GB 额外显存）
text_encoding_mode: "cached"
text_cache_dir:     "text_cache"
latent_dir:         "latent_cache"

# ── 训练 ─────────────────────────────────────────────────────
batch_size:                  4     # 单卡 batch size
gradient_accumulation_steps: 4     # 等效 batch = 4×4=16
num_epochs:                  50
learning_rate:               1.0e-4
weight_decay:                1.0e-2
warmup_steps:                500
grad_clip:                   1.0

# ── 精度 ─────────────────────────────────────────────────────
mixed_precision: "bf16"   # RTX 50xx/A100/H100 推荐；RTX 30xx 改为 "fp16"

# ── CFM ──────────────────────────────────────────────────────
sigma:            0.0    # 保持 0.0 以匹配预训练
mask_prompt_loss: true   # 不对 prompt 帧计算 loss（voice cloning 时推荐）

# ── 省显存 ────────────────────────────────────────────────────
gradient_checkpointing:  true   # 推荐开启
freeze_first_n_blocks:   0      # 冻结前 N 个 DiT block（0=全部训练）
train_text_encoder:      false  # 是否微调 UMT5（仅 gpu 模式有效）

# ── 日志与检查点 ──────────────────────────────────────────────
log_every:   20    # 每 N 步打印一次
save_every:  500   # 每 N 步保存一次
val_every:   500   # 每 N 步验证一次
keep_last_n: 3     # 保留最近 N 个检查点
```

---

## 7. 启动训练

### 单卡

```bash
# 在根目录下运行
python training/finetune.py --config training/finetune_config.yaml
```

### 多卡（Accelerate）

```bash
# 首次配置（一次性）
accelerate config

# 启动
accelerate launch training/finetune.py --config training/finetune_config.yaml
```

### 命令行覆盖配置项

```bash
python training/finetune.py --config training/finetune_config.yaml \
    --learning_rate 5e-5 \
    --num_epochs 30 \
    --output_dir my_experiment
```

---

## 8. 显存估算

以下数据基于 **LongCat-AudioDiT-1B，bf16，单卡**：

| 配置 | text 模式 | 梯度检查点 | batch×accum | 估算显存 |
|------|-----------|-----------|-------------|---------|
| ✅ 推荐（5090）| cached | ON  | 4×4 | ~18 GB |
| 次选 | cpu | ON  | 4×4 | ~18 GB |
| 大 batch | cached | ON  | 8×2 | ~22 GB |
| 无检查点 | cached | OFF | 4×4 | ~26 GB |
| 部分训练（后8层）| cached | ON  | 4×4 | ~12 GB |

> **RTX 5090（32 GB）** 使用默认 `cached + gradient_checkpointing` 配置，显存占用约 18 GB，可将 batch×accum 调至 **8×4=32** 以提高吞吐。

---

## 9. 检查点与推理

### 检查点目录结构

```
finetune_output/
  checkpoint-500/
    training_state.pt         # transformer 权重 + optimizer + scheduler 状态
    hf_model/                 # 完整 HuggingFace 模型（可直接推理）
      config.json
      model.safetensors
      tokenizer.json
      ...
  checkpoint-1000/
    ...
```

### 从微调检查点推理

```python
import audiodit
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
import soundfile as sf, torch

model = AudioDiTModel.from_pretrained(
    "finetune_output/checkpoint-1000/hf_model"
).cuda()
model.vae.to_half()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "finetune_output/checkpoint-1000/hf_model"
)

inputs = tokenizer(["今天天气很好"], return_tensors="pt", padding="longest")
output = model(
    input_ids=inputs.input_ids.cuda(),
    attention_mask=inputs.attention_mask.cuda(),
    duration=60,
    steps=16,
    cfg_strength=4.0,
    guidance_method="apg",
)
sf.write("output.wav", output.waveform.squeeze().cpu().numpy(), 24000)
```

### 仅加载 transformer 权重（继续训练）

```python
ckpt = torch.load("finetune_output/checkpoint-500/training_state.pt")
model.transformer.load_state_dict(ckpt["transformer_state_dict"])
```

---

## 10. 高级选项

### 10.1 部分冻结（节省显存 / 防止过拟合）

```yaml
# 只训练最后 8 个 DiT block（1B 模型共 24 个）
freeze_first_n_blocks: 16
```

适用场景：数据量较小（< 1 小时）或目标是轻微风格迁移。

### 10.2 开启 UMT5 文本编码器微调

```yaml
text_encoding_mode: "gpu"     # 必须为 gpu 模式
train_text_encoder:  true
learning_rate:       5.0e-5   # 建议使用更低学习率
```

适用场景：新语言适配、领域词汇扩展。

### 10.3 纯 CPU 预编码（无 GPU 机器）

```bash
python cache_embeddings.py \
    --metadata data/train.csv \
    --model_dir meituan-longcat/LongCat-AudioDiT-1B \
    --device cpu
```

---

## 11. 常见问题

**Q: 运行 cache_embeddings.py 时出现 `FileNotFoundError`？**  
A: 检查 `audio_path` 列中的路径是否正确（相对路径以运行目录为基准）。

**Q: 训练时出现 `Text cache miss`？**  
A: `cache_embeddings.py` 中的文本必须与训练时完全一致（含空格、大小写）。  
建议在两处都设置 `--normalize_text` 或都不设置。

**Q: loss 下降后又上升（过拟合）？**  
A: 减小学习率至 `3e-5`，增加 `freeze_first_n_blocks`，或增加数据量。

**Q: voice cloning 效果不好？**  
A: 确保 `mask_prompt_loss: true`，并为每条训练样本提供 `prompt_audio_path` 列。

**Q: 如何确认最优检查点？**  
A: 不要只看 loss 数值——每隔 500 步用 `inference.py` 生成音频，用听感判断。

---

## 参考

- [LongCat-AudioDiT 论文](https://arxiv.org/abs/2603.29339)
- [F5-TTS](https://github.com/SWivid/F5-TTS)
- [HuggingFace 模型页面](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B)
