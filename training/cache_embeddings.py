"""
cache_embeddings.py
--------------------
One-time offline pre-encoding script.
Encodes BOTH VAE latents and UMT5 text embeddings to disk.
After running this, training requires ZERO GPU memory for VAE or text encoder.

Usage:
    python cache_embeddings.py \
        --metadata    data/train.csv \
        --model_dir   meituan-longcat/LongCat-AudioDiT-1B \
        --latent_dir  latent_cache \
        --text_dir    text_cache \
        --device      cuda   # use 'cpu' if GPU is not available

    # Process both train and val in one go:
    python cache_embeddings.py \
        --metadata data/train.csv data/val.csv \
        --model_dir meituan-longcat/LongCat-AudioDiT-1B

Saved files
-----------
latent_cache/<stem>.pt   → (latent_dim, T) float32 tensor
text_cache/<hash>.pt     → {"last_hidden_state": (S, D), "attention_mask": (S,)}
                            key is sha256 of the full_text string (deduplication)
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import logging
from pathlib import Path

# Add parent directory to sys.path to find audiodit and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

import audiodit
from audiodit import AudioDiTModel
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Reuse helpers from finetune_dataset ─────────────────────────────────────
import librosa
import numpy as np


def load_audio_mono(path: str, sr: int) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return torch.from_numpy(audio).unsqueeze(0)  # (1, T)


def pad_to_hop(wav: torch.Tensor, hop: int) -> torch.Tensor:
    remainder = wav.shape[-1] % hop
    if remainder != 0:
        wav = F.pad(wav, (0, hop - remainder))
    return wav


def text_cache_key(text: str) -> str:
    """SHA-256 of the text string — identical texts share one cached file."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ── Load metadata ────────────────────────────────────────────────────────────

def load_metadata(path: str) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path) as f:
            return json.load(f)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


# ── Encode VAE latent ────────────────────────────────────────────────────────

@torch.no_grad()
def encode_latent(wav_path: str, vae, sr: int, hop: int, off: int,
                  device: torch.device) -> torch.Tensor:
    """Returns (latent_dim, T) float32."""
    wav = load_audio_mono(wav_path, sr)       # (1, T)
    wav = pad_to_hop(wav, hop)
    if off > 0:
        wav = F.pad(wav, (0, hop * off))
    wav = wav.unsqueeze(0).to(device)         # (1, 1, T)
    latent = vae.encode(wav)                  # (1, D, T')
    if off > 0:
        latent = latent[..., :-off]
    return latent.squeeze(0).cpu().float()    # (D, T)


# ── Encode text embedding ────────────────────────────────────────────────────

@torch.no_grad()
def encode_text(full_text: str, text_encoder, tokenizer,
                text_add_embed: bool, text_norm_feat: bool,
                device: torch.device) -> dict:
    """
    Returns dict with keys:
        last_hidden_state  (S, d_model) float32
        attention_mask     (S,)         bool
    """
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
    )
    input_ids    = enc.input_ids.to(device)
    attn_mask    = enc.attention_mask.to(device)

    out = text_encoder(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
    )

    emb = out.last_hidden_state   # (1, S, D)
    d_model = text_encoder.config.d_model

    if text_norm_feat:
        emb = F.layer_norm(emb, (d_model,), eps=1e-6)
    if text_add_embed:
        first = out.hidden_states[0]
        if text_norm_feat:
            first = F.layer_norm(first, (d_model,), eps=1e-6)
        emb = emb + first

    return {
        "last_hidden_state": emb.squeeze(0).cpu().float(),   # (S, D)
        "attention_mask":    attn_mask.squeeze(0).cpu(),     # (S,)
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache VAE latents and UMT5 text embeddings to disk."
    )
    parser.add_argument("--metadata",   nargs="+", required=True,
                        help="One or more metadata CSV/JSON paths.")
    parser.add_argument("--model_dir",  type=str,
                        default="meituan-longcat/LongCat-AudioDiT-1B")
    parser.add_argument("--latent_dir", type=str, default="latent_cache",
                        help="Directory for cached VAE latents.")
    parser.add_argument("--text_dir",   type=str, default="text_cache",
                        help="Directory for cached text embeddings.")
    parser.add_argument("--device",     type=str, default="cuda",
                        help="Device for encoding: 'cuda' or 'cpu'.")
    parser.add_argument("--off",        type=int, default=3,
                        help="Extra latent frames appended during VAE encode (default 3).")
    parser.add_argument("--skip_latent", action="store_true",
                        help="Skip VAE encoding (only encode text).")
    parser.add_argument("--skip_text",   action="store_true",
                        help="Skip text encoding (only encode latents).")
    parser.add_argument("--normalize_text", action="store_true",
                        help="Apply normalize_text() before encoding.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    Path(args.latent_dir).mkdir(parents=True, exist_ok=True)
    Path(args.text_dir).mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────
    logger.info(f"Loading model from {args.model_dir} ...")
    model = AudioDiTModel.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

    vae          = model.vae.to(device)
    vae.to_half()                           # VAE uses fp16 (matches pretrain)
    vae.eval()

    text_encoder = model.text_encoder.to(device)
    text_encoder.eval()

    sr           = model.config.sampling_rate   # 24000
    hop          = model.config.latent_hop      # 2048
    text_add_emb = model.config.text_add_embed
    text_norm    = model.config.text_norm_feat

    # ── Collect all rows ────────────────────────────────────────────────
    all_rows = []
    for meta_path in args.metadata:
        rows = load_metadata(meta_path)
        all_rows.extend(rows)
    logger.info(f"Total rows: {len(all_rows)}")

    from utils import normalize_text as _normalize

    # ── Encode ──────────────────────────────────────────────────────────
    n_lat_done = n_lat_skip = 0
    n_txt_done = n_txt_skip = 0

    for i, row in enumerate(all_rows):
        audio_path   = row.get("audio_path") or row.get("path") or row.get("wav", "")
        text         = row.get("text") or row.get("transcript", "")
        prompt_text  = row.get("prompt_text", "") or ""
        prompt_audio = row.get("prompt_audio_path", "") or ""

        if args.normalize_text:
            text = _normalize(text)
            if prompt_text:
                prompt_text = _normalize(prompt_text)

        full_text = f"{prompt_text} {text}".strip() if prompt_text else text

        # ── VAE latent ───────────────────────────────────────────────────
        if not args.skip_latent and audio_path:
            stem      = Path(audio_path).stem
            lat_file  = Path(args.latent_dir) / f"{stem}.pt"
            if lat_file.exists():
                n_lat_skip += 1
            else:
                try:
                    latent = encode_latent(audio_path, vae, sr, hop,
                                           args.off, device)
                    torch.save(latent, lat_file)
                    n_lat_done += 1
                except Exception as e:
                    logger.warning(f"[latent] {audio_path}: {e}")

            # Also encode prompt audio if different from main audio
            if prompt_audio and prompt_audio != audio_path:
                p_stem    = Path(prompt_audio).stem
                p_lat_file = Path(args.latent_dir) / f"{p_stem}.pt"
                if not p_lat_file.exists():
                    try:
                        plat = encode_latent(prompt_audio, vae, sr, hop,
                                              args.off, device)
                        torch.save(plat, p_lat_file)
                        n_lat_done += 1
                    except Exception as e:
                        logger.warning(f"[latent/prompt] {prompt_audio}: {e}")

        # ── Text embedding ───────────────────────────────────────────────
        if not args.skip_text and full_text:
            key      = text_cache_key(full_text)
            txt_file = Path(args.text_dir) / f"{key}.pt"
            if txt_file.exists():
                n_txt_skip += 1
            else:
                try:
                    emb_dict = encode_text(full_text, text_encoder, tokenizer,
                                           text_add_emb, text_norm, device)
                    torch.save(emb_dict, txt_file)
                    n_txt_done += 1
                except Exception as e:
                    logger.warning(f"[text] '{full_text[:40]}': {e}")

        if (i + 1) % 100 == 0:
            logger.info(
                f"[{i+1}/{len(all_rows)}] "
                f"latent: done={n_lat_done} skip={n_lat_skip} | "
                f"text:   done={n_txt_done} skip={n_txt_skip}"
            )

    logger.info(
        f"\nDone!\n"
        f"  Latents : {n_lat_done} encoded, {n_lat_skip} already cached\n"
        f"  Text    : {n_txt_done} encoded, {n_txt_skip} already cached\n"
        f"  Latent dir : {args.latent_dir}\n"
        f"  Text dir   : {args.text_dir}"
    )


if __name__ == "__main__":
    main()
