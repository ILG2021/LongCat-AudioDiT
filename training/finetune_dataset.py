"""
finetune_dataset.py  (v2)
--------------------------
Dataset for fine-tuning LongCat-AudioDiT.

Three text-encoding modes (selected automatically by which args are given):

  Mode 1 — PRE-CACHED (recommended, zero GPU VRAM for text/VAE)
      Pass text_cache_dir=".../text_cache" AND latent_dir=".../latent_cache".
      Both were written by cache_embeddings.py.
      Training loop needs NO text encoder or VAE on the GPU at all.

  Mode 2 — CPU OFFLOAD
      Pass text_encoder + tokenizer; they stay on CPU.
      ~0 GPU VRAM for text encoder; small CPU latency per batch.

  Mode 3 — LIVE GPU (original)
      Pass text_encoder + tokenizer; they move to the same device as the model.
      Requires ~1.1 GB extra GPU VRAM for UMT5-base.

VAE latents are always cached to disk after the first encode.

Metadata CSV / JSON columns
---------------------------
    audio_path          required
    text                required
    duration            optional (used for fast filtering)
    prompt_audio_path   optional (voice-cloning: reference speaker WAV)
    prompt_text         optional (transcript of prompt_audio)
"""

import csv
import json
import hashlib
import logging
import re
import os
import sys
from pathlib import Path

# Add parent directory to sys.path to find audiodit and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio_mono(path: str, sr: int) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return torch.from_numpy(audio).unsqueeze(0)      # (1, T)


def _pad_to_hop(wav: torch.Tensor, hop: int) -> torch.Tensor:
    r = wav.shape[-1] % hop
    return F.pad(wav, (0, hop - r)) if r != 0 else wav


def _text_cache_key(text: str) -> str:
    """SHA-256 of the text string (matches cache_embeddings.py)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AudioDiTFinetuneDataset(Dataset):
    """
    Args
    ----
    metadata_path   CSV / JSON file listing samples.
    model           AudioDiTModel — used for VAE encode and config only.
    tokenizer       HF tokenizer (required for modes 2/3; ignored in mode 1).
    text_encoder    UMT5EncoderModel (required for modes 2/3; ignored in mode 1).
    text_cache_dir  Directory written by cache_embeddings.py (mode 1).
    latent_dir      Directory for cached VAE latents (all modes).
    text_encoder_device  'cpu' (mode 2) or same device as model (mode 3).
    max_duration    Upper clip length in seconds.
    min_duration    Lower clip length in seconds.
    off             Extra latent frames used during VAE encode (default 3).
    normalize_text_fn  Optional callable applied to transcript strings.
    """

    def __init__(
        self,
        metadata_path: str,
        model,                                  # AudioDiTModel
        tokenizer=None,
        text_encoder=None,
        text_cache_dir: Optional[str] = None,
        latent_dir: str = "latent_cache",
        text_encoder_device: str = "cpu",
        max_duration: float = 15.0,
        min_duration: float = 0.5,
        off: int = 3,
        normalize_text_fn: Optional[Callable] = None,
    ):
        super().__init__()

        # ── model config ──────────────────────────────────────────────────
        self.sr         = model.config.sampling_rate    # 24000
        self.hop        = model.config.latent_hop       # 2048
        self.latent_dim = model.config.latent_dim       # 64
        self.vae        = model.vae
        self.add_embed  = model.config.text_add_embed
        self.norm_feat  = model.config.text_norm_feat

        self.max_frames = int(max_duration * self.sr // self.hop)
        self.min_frames = max(1, int(min_duration * self.sr // self.hop))
        self.off        = off
        self.normalize  = normalize_text_fn

        # ── caching dirs ──────────────────────────────────────────────────
        self.latent_dir    = Path(latent_dir)
        self.text_cache_dir = Path(text_cache_dir) if text_cache_dir else None
        self.latent_dir.mkdir(parents=True, exist_ok=True)

        # ── encoding mode ─────────────────────────────────────────────────
        if self.text_cache_dir is not None and self.text_cache_dir.exists():
            self.mode = 1   # pre-cached embeddings from disk
            logger.info("Text mode: PRE-CACHED (zero GPU VRAM for text encoder)")
        elif text_encoder is not None and tokenizer is not None:
            self.mode        = 2   # CPU or GPU text encoder
            self.tokenizer   = tokenizer
            self.text_encoder = text_encoder.to(torch.device(text_encoder_device))
            self.te_device   = torch.device(text_encoder_device)
            self.text_encoder.eval()
            logger.info(f"Text mode: LIVE on {text_encoder_device}")
        else:
            raise ValueError(
                "Provide either text_cache_dir (mode 1) "
                "or text_encoder + tokenizer (mode 2/3)."
            )

        # ── metadata ─────────────────────────────────────────────────────
        self.samples = self._load_metadata(metadata_path, max_duration, min_duration)
        logger.info(f"Dataset ready: {len(self.samples)} samples from {metadata_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Metadata loading
    # ──────────────────────────────────────────────────────────────────────

    def _load_metadata(self, path: str, max_dur: float, min_dur: float):
        path = Path(path)
        if path.suffix.lower() == ".json":
            with open(path) as f:
                rows = json.load(f)
        else:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        kept = []
        for row in rows:
            ap   = row.get("audio_path") or row.get("path") or row.get("wav", "")
            text = row.get("text") or row.get("transcript", "")
            if not ap or not text:
                continue
            dur = row.get("duration")
            if dur is not None:
                dur = float(dur)
                if dur > max_dur or dur < min_dur:
                    continue
            kept.append({
                "audio_path":        ap.strip(),
                "text":              text.strip(),
                "duration":          dur,
                "prompt_audio_path": (row.get("prompt_audio_path") or "").strip() or None,
                "prompt_text":       (row.get("prompt_text") or "").strip() or None,
            })
        return kept

    # ──────────────────────────────────────────────────────────────────────
    # VAE encode (always disk-cached after first call)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_latent(self, wav_path: str) -> torch.Tensor:
        """Returns (latent_dim, T) float32, loaded from disk if available."""
        stem     = Path(wav_path).stem
        lat_file = self.latent_dir / f"{stem}.pt"

        if lat_file.exists():
            return torch.load(lat_file, map_location="cpu")

        # Encode on whichever device the VAE currently lives on
        vae_dev = next(self.vae.parameters()).device
        wav     = _load_audio_mono(wav_path, self.sr)   # (1, T)
        wav     = _pad_to_hop(wav, self.hop)
        if self.off > 0:
            wav = F.pad(wav, (0, self.hop * self.off))
        wav = wav.unsqueeze(0).to(vae_dev)              # (1, 1, T)

        latent = self.vae.encode(wav)                   # (1, D, T')
        if self.off > 0:
            latent = latent[..., :-self.off]
        latent = latent.squeeze(0).cpu().float()        # (D, T)

        torch.save(latent, lat_file)
        return latent

    # ──────────────────────────────────────────────────────────────────────
    # Text embed (mode 1: disk cache; mode 2/3: live encoder)
    # ──────────────────────────────────────────────────────────────────────

    def _get_text_embedding(self, full_text: str) -> dict:
        """
        Returns dict:
            last_hidden_state  (S, D) float32
            attention_mask     (S,)   bool
        """
        if self.mode == 1:
            key      = _text_cache_key(full_text)
            txt_file = self.text_cache_dir / f"{key}.pt"
            if not txt_file.exists():
                raise FileNotFoundError(
                    f"Text cache miss for '{full_text[:60]}'\n"
                    f"  Expected: {txt_file}\n"
                    f"  Run cache_embeddings.py first."
                )
            return torch.load(txt_file, map_location="cpu")

        # Mode 2/3 — live encode
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        input_ids = enc.input_ids.to(self.te_device)
        attn_mask = enc.attention_mask.to(self.te_device)

        with torch.no_grad():
            out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        d  = self.text_encoder.config.d_model
        emb = out.last_hidden_state                         # (1, S, D)
        if self.norm_feat:
            emb = F.layer_norm(emb, (d,), eps=1e-6)
        if self.add_embed:
            first = out.hidden_states[0]
            if self.norm_feat:
                first = F.layer_norm(first, (d,), eps=1e-6)
            emb = emb + first

        return {
            "last_hidden_state": emb.squeeze(0).cpu().float(),  # (S, D)
            "attention_mask":    attn_mask.squeeze(0).cpu(),    # (S,)
        }

    # ──────────────────────────────────────────────────────────────────────
    # __len__ / __getitem__
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # ── text ──────────────────────────────────────────────────────────
        text = s["text"]
        if self.normalize:
            text = self.normalize(text)

        prompt_text = s["prompt_text"] or ""
        if prompt_text and self.normalize:
            prompt_text = self.normalize(prompt_text)

        full_text = f"{prompt_text} {text}".strip() if prompt_text else text

        emb_dict = self._get_text_embedding(full_text)
        text_emb  = emb_dict["last_hidden_state"]   # (S, D)
        text_amask = emb_dict["attention_mask"]     # (S,)

        # ── target latent ─────────────────────────────────────────────────
        target = self._encode_latent(s["audio_path"])   # (D, T)
        T      = target.shape[-1]

        # Duration guard
        if T > self.max_frames:
            target = target[:, :self.max_frames]
            T = self.max_frames
        if T < self.min_frames:
            target = F.pad(target, (0, self.min_frames - T))
            T = self.min_frames

        # ── prompt latent (voice cloning) ─────────────────────────────────
        prompt_frames = 0
        if s["prompt_audio_path"]:
            p_lat = self._encode_latent(s["prompt_audio_path"])  # (D, P)
            P = p_lat.shape[-1]
            target = torch.cat([p_lat, target], dim=-1)           # (D, P+T)
            T = target.shape[-1]
            if T > self.max_frames:
                target = target[:, :self.max_frames]
                T = self.max_frames
            prompt_frames = min(P, T)

        return {
            "text_emb":     text_emb,        # (S, D)  pre-computed embedding
            "text_amask":   text_amask,      # (S,)
            "target_latent": target,         # (D, T)
            "latent_len":    T,
            "prompt_frames": prompt_frames,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    Pad variable-length sequences.

    Returns
    -------
    text_emb      (B, S_max, D)
    text_amask    (B, S_max)    bool (True = real token)
    target_latent (B, T_max, D)  transposed for DiT input
    latent_mask   (B, T_max)    bool (True = valid frame)
    prompt_frames (B,)
    """
    max_S   = max(x["text_emb"].shape[0]   for x in batch)
    max_T   = max(x["latent_len"]          for x in batch)
    D_lat   = batch[0]["target_latent"].shape[0]
    D_text  = batch[0]["text_emb"].shape[-1]

    text_embs, text_amasks = [], []
    latents, lat_masks, pf_list = [], [], []

    for x in batch:
        # ── text ──────────────────────────────────────────────────────────
        S    = x["text_emb"].shape[0]
        pad  = max_S - S
        text_embs.append(F.pad(x["text_emb"],   (0, 0, 0, pad)))    # (S_max, D)
        text_amasks.append(F.pad(x["text_amask"], (0, pad)))         # (S_max,)

        # ── latent ────────────────────────────────────────────────────────
        T    = x["latent_len"]
        pad_T = max_T - T
        lat  = F.pad(x["target_latent"], (0, pad_T))                 # (D, T_max)
        latents.append(lat.T)                                        # (T_max, D)

        mask = torch.zeros(max_T, dtype=torch.bool)
        mask[:T] = True
        lat_masks.append(mask)
        pf_list.append(x["prompt_frames"])

    return {
        "text_emb":      torch.stack(text_embs),    # (B, S_max, D)
        "text_amask":    torch.stack(text_amasks).bool(),  # (B, S_max)
        "target_latent": torch.stack(latents),      # (B, T_max, D)
        "latent_mask":   torch.stack(lat_masks),    # (B, T_max)
        "prompt_frames": torch.tensor(pf_list, dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    metadata_path: str,
    model,
    tokenizer=None,
    text_encoder=None,
    text_cache_dir: Optional[str] = None,
    latent_dir: str = "latent_cache",
    text_encoder_device: str = "cpu",
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    dataset = AudioDiTFinetuneDataset(
        model=model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_cache_dir=text_cache_dir,
        latent_dir=latent_dir,
        text_encoder_device=text_encoder_device,
        metadata_path=metadata_path,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
