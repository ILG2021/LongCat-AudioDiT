"""
finetune.py  (v2)
-----------------
Fine-tune LongCat-AudioDiT (DiT transformer only) using CFM loss.

Text-encoding strategy is controlled by finetune_config.yaml:

  text_encoding_mode: "cached"     → load from text_cache/ (zero GPU VRAM)  ✅ recommended for 5090
  text_encoding_mode: "cpu"        → run UMT5 on CPU per batch              ⚡ fallback
  text_encoding_mode: "gpu"        → run UMT5 on GPU (~1.1 GB extra VRAM)   (legacy)

CFM objective (same as F5-TTS and LongCat paper):
    x_t  = (1 - (1-σ)t) x₀  +  t x₁
    loss = MSE( DiT(x_t, t, text_emb, latent_cond), x₁ - (1-σ)x₀ )

Usage:
    # Single GPU
    python finetune.py --config finetune_config.yaml

    # Multi-GPU (accelerate)
    accelerate launch finetune.py --config finetune_config.yaml

    # Quick CLI override
    python finetune.py --config finetune_config.yaml --learning_rate 5e-5 --num_epochs 30
"""

import os
import sys

# Add parent directory to sys.path to find audiodit and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import shutil
import logging
import argparse
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

import audiodit
from audiodit import AudioDiTModel
from transformers import AutoTokenizer

from finetune_dataset import build_dataloader
from utils import normalize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FinetuneConfig:
    # ── Model ────────────────────────────────────────────────────────────
    model_dir:  str = "meituan-longcat/LongCat-AudioDiT-1B"
    output_dir: str = "finetune_output"

    # ── Data ─────────────────────────────────────────────────────────────
    train_metadata: str           = "data/train.csv"
    val_metadata:   Optional[str] = None
    max_duration:   float         = 15.0
    min_duration:   float         = 0.5

    # Latent cache (VAE, all modes)
    latent_dir: str = "latent_cache"

    # Text encoding mode:
    #   "cached" → read from text_cache_dir (run cache_embeddings.py first)
    #   "cpu"    → run UMT5 on CPU per batch
    #   "gpu"    → run UMT5 on same GPU as DiT
    text_encoding_mode: str = "cached"
    text_cache_dir:     str = "text_cache"   # used only for mode "cached"

    # ── Training ─────────────────────────────────────────────────────────
    batch_size:                   int   = 4
    gradient_accumulation_steps:  int   = 4
    num_epochs:                   int   = 50
    max_steps:          Optional[int]   = None
    learning_rate:                float = 1e-4
    weight_decay:                 float = 1e-2
    warmup_steps:                 int   = 500
    grad_clip:                    float = 1.0
    seed:                         int   = 42

    # ── Precision ─────────────────────────────────────────────────────────
    # "no" | "fp16" | "bf16"
    mixed_precision: str = "bf16"

    # ── CFM ──────────────────────────────────────────────────────────────
    sigma:            float = 0.0    # noise floor; keep 0.0 to match pretrain
    mask_prompt_loss: bool  = True   # ignore loss on prompt (reference) frames

    # ── Optional: partial training ────────────────────────────────────────
    freeze_first_n_blocks: int  = 0      # freeze first N DiT blocks
    train_text_encoder:    bool = False  # unfreeze UMT5 (only with mode gpu)

    # ── Memory / speed ────────────────────────────────────────────────────
    gradient_checkpointing: bool = True
    num_workers:             int  = 4

    # ── Logging & checkpointing ────────────────────────────────────────────
    log_every:   int = 20
    save_every:  int = 500
    val_every:   int = 500
    keep_last_n: int = 3


def load_config(path: str) -> FinetuneConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    return FinetuneConfig(**{k: v for k, v in d.items() if hasattr(FinetuneConfig, k)})


# ═══════════════════════════════════════════════════════════════════════════════
# CFM helpers
# ═══════════════════════════════════════════════════════════════════════════════

def sample_timesteps(B: int, device: torch.device) -> torch.Tensor:
    """
    Logit-normal timestep sampling  t = sigmoid(N(0,1)).
    Concentrates samples near t≈0.5 where the model has the hardest job.
    Matches F5-TTS / Voicebox practice.
    """
    return torch.sigmoid(torch.randn(B, device=device))


def cfm_loss(
    pred:   torch.Tensor,   # (B, T, D)  predicted velocity
    x0:     torch.Tensor,   # (B, T, D)  noise
    x1:     torch.Tensor,   # (B, T, D)  target latent
    mask:   torch.Tensor,   # (B, T)     bool — valid frames
    sigma:  float = 0.0,
) -> torch.Tensor:
    """
    MSE between predicted velocity and the CFM target velocity,
    averaged only over valid (non-padded, non-prompt) frames.
    """
    v_target = x1 - (1.0 - sigma) * x0                        # (B, T, D)
    err      = (pred - v_target).pow(2)                        # (B, T, D)
    err      = err * mask.unsqueeze(-1).float()
    n_valid  = mask.float().sum().clamp(min=1.0) * err.shape[-1]
    return err.sum() / n_valid


# ═══════════════════════════════════════════════════════════════════════════════
# Training step
# ═══════════════════════════════════════════════════════════════════════════════

def training_step(
    transformer,            # AudioDiTTransformer  (the only trainable module)
    batch:          dict,
    sigma:          float,
    device:         torch.device,
    mask_prompt:    bool,
    amp_dtype:      torch.dtype,
) -> torch.Tensor:
    """
    Performs one CFM forward pass and returns the scalar loss.

    Batch keys (produced by finetune_dataset.collate_fn):
        text_emb       (B, S, D)  pre-computed UMT5 embedding  ← no text encoder on GPU!
        text_amask     (B, S)     bool
        target_latent  (B, T, D)  clean VAE latent
        latent_mask    (B, T)     bool
        prompt_frames  (B,)       int
    """
    text_emb      = batch["text_emb"].to(device, torch.float32)   # (B, S, D)
    text_amask    = batch["text_amask"].to(device)                  # (B, S) bool
    x1            = batch["target_latent"].to(device)               # (B, T, D)
    latent_mask   = batch["latent_mask"].to(device)                 # (B, T) bool
    prompt_frames = batch["prompt_frames"].to(device)               # (B,)

    B, T, D = x1.shape
    text_len = text_amask.long().sum(dim=1)                         # (B,)

    # ── Noise & timestep ─────────────────────────────────────────────────
    x0 = torch.randn_like(x1)
    t  = sample_timesteps(B, device)                                # (B,)

    # ── Interpolate: x_t = (1-(1-σ)t)*x0 + t*x1 ─────────────────────────
    t3 = t[:, None, None]
    x_t = (1.0 - (1.0 - sigma) * t3) * x0 + t3 * x1               # (B, T, D)

    # ── Latent conditioning (clean prompt region, zeros for gen region) ───
    latent_cond = torch.zeros_like(x1)
    for i, pf in enumerate(prompt_frames):
        pf = pf.item()
        if pf > 0:
            latent_cond[i, :pf] = x1[i, :pf]

    # ── Fix training-inference mismatch for prompt frames ─────────────────
    # At inference, prompt region follows: noise*(1-t) + prompt_latent*t
    # Apply the same interpolation during training so distributions match.
    for i, pf in enumerate(prompt_frames):
        pf = pf.item()
        if pf > 0:
            ti = t[i].item()
            x_t[i, :pf] = x0[i, :pf] * (1.0 - ti) + x1[i, :pf] * ti

    # ── DiT forward ───────────────────────────────────────────────────────
    with torch.autocast(device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
        out = transformer(
            x          = x_t,
            text       = text_emb,
            text_len   = text_len,
            time       = t,
            mask       = latent_mask,
            cond_mask  = text_amask,
            latent_cond= latent_cond,
        )
    pred_v = out["last_hidden_state"]   # (B, T, D)

    # ── Loss mask ─────────────────────────────────────────────────────────
    loss_mask = latent_mask.clone()
    if mask_prompt:
        for i, pf in enumerate(prompt_frames):
            pf = pf.item()
            if pf > 0:
                loss_mask[i, :pf] = False

    return cfm_loss(pred_v, x0, x1, loss_mask, sigma=sigma)


# ═══════════════════════════════════════════════════════════════════════════════
# Build DataLoader  (wraps finetune_dataset.build_dataloader)
# ═══════════════════════════════════════════════════════════════════════════════

def make_loader(cfg: FinetuneConfig, model, split: str = "train"):
    meta = cfg.train_metadata if split == "train" else cfg.val_metadata

    # Choose text encoding mode
    text_encoder = tokenizer = text_cache_dir = None
    te_device = "cpu"

    if cfg.text_encoding_mode == "cached":
        text_cache_dir = cfg.text_cache_dir
        logger.info(f"[{split}] Text: loading from disk cache '{text_cache_dir}'")
    else:
        tokenizer    = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        from transformers import UMT5EncoderModel
        text_encoder = model.text_encoder    # already constructed inside model
        if cfg.text_encoding_mode == "cpu":
            te_device = "cpu"
            logger.info(f"[{split}] Text: UMT5 on CPU")
        else:
            te_device = "cuda"
            logger.info(f"[{split}] Text: UMT5 on GPU")

    return build_dataloader(
        metadata_path      = meta,
        model              = model,
        tokenizer          = tokenizer,
        text_encoder       = text_encoder,
        text_cache_dir     = text_cache_dir,
        latent_dir         = cfg.latent_dir,
        text_encoder_device= te_device,
        batch_size         = cfg.batch_size,
        num_workers        = cfg.num_workers,
        shuffle            = (split == "train"),
        max_duration       = cfg.max_duration,
        min_duration       = cfg.min_duration,
        normalize_text_fn  = normalize_text,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def nullcontext():
    yield


def train(cfg: FinetuneConfig):
    torch.manual_seed(cfg.seed)

    # ── Accelerator ───────────────────────────────────────────────────────
    if HAS_ACCELERATE:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with="tensorboard",
            project_dir=cfg.output_dir,
        )
        device   = accelerator.device
        is_main  = accelerator.is_main_process
    else:
        accelerator = None
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main  = True

    if is_main:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Device : {device}")
        logger.info(f"Output : {cfg.output_dir}")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        cfg.mixed_precision, torch.float32
    )

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading model from {cfg.model_dir}")
    model = AudioDiTModel.from_pretrained(cfg.model_dir)
    model.vae.to_half()

    # ── Freeze ───────────────────────────────────────────────────────────
    model.vae.requires_grad_(False)

    if cfg.text_encoding_mode in ("cached", "cpu"):
        # Text encoder not needed on GPU at all
        model.text_encoder.requires_grad_(False)
        if cfg.text_encoding_mode == "cached":
            # Move text encoder to meta device to reclaim VRAM entirely
            # (it is never called during training)
            model.text_encoder = model.text_encoder.to("cpu")
            logger.info("UMT5 text encoder: moved to CPU (not used during training)")
        else:
            model.text_encoder = model.text_encoder.to("cpu")
            logger.info("UMT5 text encoder: CPU offload mode")
    elif cfg.train_text_encoder:
        model.text_encoder.requires_grad_(True)
        logger.info("UMT5 text encoder: TRAINABLE on GPU")
    else:
        model.text_encoder.requires_grad_(False)
        logger.info("UMT5 text encoder: frozen on GPU")

    # Freeze first N DiT blocks
    if cfg.freeze_first_n_blocks > 0:
        for i, blk in enumerate(model.transformer.blocks):
            if i < cfg.freeze_first_n_blocks:
                blk.requires_grad_(False)
        logger.info(f"DiT blocks 0–{cfg.freeze_first_n_blocks-1}: frozen")

    # Gradient checkpointing on the transformer
    if cfg.gradient_checkpointing:
        model.transformer.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: ON")

    # Only the transformer (and optionally text encoder) go to GPU
    model.transformer = model.transformer.to(device)
    model.vae         = model.vae.to(device)          # VAE needed for live latent encode in dataset
    if cfg.text_encoding_mode == "gpu":
        model.text_encoder = model.text_encoder.to(device)

    model.train()
    model.vae.eval()
    model.text_encoder.eval()

    # ── VRAM report ───────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(f"Trainable : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        if torch.cuda.is_available():
            reserved_gb = torch.cuda.memory_reserved(device) / 1e9
            logger.info(f"GPU mem reserved after model load: {reserved_gb:.2f} GB")

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = make_loader(cfg, model, "train")
    val_loader   = make_loader(cfg, model, "val") if cfg.val_metadata else None

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate,
                      weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    # ── LR schedule: linear warmup → cosine ───────────────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = cfg.max_steps or cfg.num_epochs * steps_per_epoch

    warmup_sched = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0,
                             total_iters=cfg.warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=max(1, total_steps - cfg.warmup_steps),
                                     eta_min=cfg.learning_rate * 0.05)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                              milestones=[cfg.warmup_steps])

    # ── Accelerate wrapping ───────────────────────────────────────────────
    if accelerator:
        model.transformer, optimizer, train_loader, scheduler = accelerator.prepare(
            model.transformer, optimizer, train_loader, scheduler
        )
        if val_loader:
            val_loader = accelerator.prepare(val_loader)

    # ── Checkpoint helpers ────────────────────────────────────────────────
    saved_ckpts: list[Path] = []

    def _unwrap(m):
        return accelerator.unwrap_model(m) if accelerator else m

    def save_checkpoint(step: int):
        if not is_main:
            return
        ckpt = Path(cfg.output_dir) / f"checkpoint-{step}"
        ckpt.mkdir(parents=True, exist_ok=True)

        torch.save({
            "step":                    step,
            "transformer_state_dict":  _unwrap(model.transformer).state_dict(),
            "optimizer_state_dict":    optimizer.state_dict(),
            "scheduler_state_dict":    scheduler.state_dict(),
            "config":                  asdict(cfg),
        }, ckpt / "training_state.pt")

        # Full HF model — re-load text encoder for saving
        unwrapped_model = _unwrap(model)
        unwrapped_model.save_pretrained(ckpt / "hf_model")
        # Save tokenizer (need tokenizer obj; re-fetch from config)
        tok = AutoTokenizer.from_pretrained(unwrapped_model.config.text_encoder_model)
        tok.save_pretrained(ckpt / "hf_model")

        saved_ckpts.append(ckpt)
        logger.info(f"Saved checkpoint: {ckpt}")

        # Prune old checkpoints
        while len(saved_ckpts) > cfg.keep_last_n:
            old = saved_ckpts.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            logger.info(f"Removed old checkpoint: {old}")

    # ── Validation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def run_val() -> float:
        if val_loader is None:
            return float("nan")
        model.eval()
        total, count = 0.0, 0
        for batch in val_loader:
            if accelerator is None:
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
            loss = training_step(
                _unwrap(model).transformer, batch,
                cfg.sigma, device, cfg.mask_prompt_loss, amp_dtype,
            )
            total += loss.item()
            count += 1
        model.train()
        return total / max(count, 1)

    # ── Training loop ─────────────────────────────────────────────────────
    global_step  = 0
    running_loss = 0.0
    logger.info(f"Start training — {cfg.num_epochs} epochs / ~{total_steps} opt steps")

    for epoch in range(cfg.num_epochs):
        for batch in train_loader:
            if accelerator is None:
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}

            ctx = accelerator.accumulate(model.transformer) if accelerator else nullcontext()
            with ctx:
                loss = training_step(
                    _unwrap(model).transformer, batch,
                    cfg.sigma, device, cfg.mask_prompt_loss, amp_dtype,
                )

                if accelerator:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                if accelerator:
                    accelerator.clip_grad_norm_(trainable_params, cfg.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            sync = (accelerator is None) or accelerator.sync_gradients
            if sync:
                global_step += 1

                if is_main and global_step % cfg.log_every == 0:
                    avg  = running_loss / cfg.log_every
                    lr   = scheduler.get_last_lr()[0]
                    mem  = (f" | GPU {torch.cuda.memory_reserved(device)/1e9:.1f}GB"
                            if torch.cuda.is_available() else "")
                    logger.info(
                        f"Epoch {epoch+1:3d} | Step {global_step:6d} | "
                        f"Loss {avg:.4f} | LR {lr:.2e}{mem}"
                    )
                    if accelerator:
                        accelerator.log({"train/loss": avg, "train/lr": lr},
                                        step=global_step)
                    running_loss = 0.0

                if val_loader and global_step % cfg.val_every == 0:
                    vl = run_val()
                    if is_main:
                        logger.info(f"  Val loss: {vl:.4f}")
                        if accelerator:
                            accelerator.log({"val/loss": vl}, step=global_step)

                if global_step % cfg.save_every == 0:
                    save_checkpoint(global_step)

                if cfg.max_steps and global_step >= cfg.max_steps:
                    break

        if cfg.max_steps and global_step >= cfg.max_steps:
            break

    save_checkpoint(global_step)
    logger.info("Training complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LongCat-AudioDiT")
    parser.add_argument("--config", type=str, default=None)
    for f in FinetuneConfig.__dataclass_fields__:
        fld = FinetuneConfig.__dataclass_fields__[f]
        t   = fld.type if isinstance(fld.type, type) else str
        parser.add_argument(f"--{f}", type=t, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else FinetuneConfig()
    for k, v in vars(args).items():
        if k != "config" and v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    train(cfg)


if __name__ == "__main__":
    main()
