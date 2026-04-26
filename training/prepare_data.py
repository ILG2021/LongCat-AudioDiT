"""
prepare_data.py
---------------
Prepare a fine-tuning metadata CSV from a directory of WAV + transcript files.

Supported input layouts
-----------------------
Layout A — flat directory with a text file per WAV:
    data/
        speaker1_001.wav
        speaker1_001.txt     # one line: the transcript
        speaker1_002.wav
        speaker1_002.txt
        ...

Layout B — flat directory with a single metadata file:
    data/
        metadata.csv         # columns: filename,text  (no header or with header)
        *.wav

Layout C — LJSpeech-style:
    data/
        wavs/
            LJ001-0001.wav
            ...
        metadata.csv         # pipe-separated: id|text|text_normalized

Usage
-----
    # Layout A (auto-detect .txt sidecar)
    python prepare_data.py --wav_dir data/wavs --output data/train.csv

    # Layout B (custom CSV)
    python prepare_data.py --wav_dir data/wavs --metadata data/metadata.csv --output data/train.csv

    # Layout C (LJSpeech)
    python prepare_data.py --wav_dir data/wavs --metadata data/metadata.csv \\
        --sep "|" --id_col 0 --text_col 2 --output data/train.csv

    # Add a fixed speaker prompt for voice-cloning fine-tuning
    python prepare_data.py --wav_dir data/wavs --output data/train.csv \\
        --prompt_audio data/prompt.wav --prompt_text "参考音频对应的文本"

    # Split into train / val
    python prepare_data.py --wav_dir data/wavs --output data/train.csv \\
        --val_output data/val.csv --val_ratio 0.05
"""

import os
import sys
import re
import csv
import argparse
import random
from pathlib import Path

# Add parent directory to sys.path to find audiodit and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import librosa


# ============================================================================
# Helpers
# ============================================================================

def get_audio_duration(path: str, target_sr: int = 24000) -> float:
    """Return duration in seconds without fully decoding the file."""
    try:
        duration = librosa.get_duration(path=path)
        return duration
    except Exception:
        return 0.0


def collect_layout_a(wav_dir: Path) -> list[dict]:
    """WAV + sidecar .txt files in the same directory."""
    samples = []
    for wav_path in sorted(wav_dir.glob("**/*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        samples.append({"audio_path": str(wav_path), "text": text})
    return samples


def collect_layout_b(wav_dir: Path, metadata_path: Path,
                     sep: str = ",", id_col: int = 0, text_col: int = 1,
                     has_header: bool = True) -> list[dict]:
    """WAV directory + separate metadata CSV."""
    samples = []
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=sep)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) <= max(id_col, text_col):
                continue
            stem = row[id_col].strip()
            text = row[text_col].strip()
            if not text:
                continue
            # Try stem as-is, then with .wav extension
            wav_path = wav_dir / stem
            if not wav_path.exists():
                wav_path = wav_dir / (stem + ".wav")
            if not wav_path.exists():
                # Search recursively
                matches = list(wav_dir.glob(f"**/{stem}.wav"))
                if matches:
                    wav_path = matches[0]
                else:
                    continue
            samples.append({"audio_path": str(wav_path), "text": text})
    return samples


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning metadata for LongCat-AudioDiT")

    # Input
    parser.add_argument("--wav_dir",  type=str, required=True,
                        help="Directory containing WAV files (searched recursively).")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Optional metadata CSV/TSV (Layout B / C). "
                             "If omitted, sidecar .txt files are used (Layout A).")
    parser.add_argument("--sep",      type=str, default=",",
                        help="Delimiter for --metadata file (default ',').")
    parser.add_argument("--id_col",   type=int, default=0,
                        help="Column index for audio ID/filename (default 0).")
    parser.add_argument("--text_col", type=int, default=1,
                        help="Column index for transcript text (default 1).")
    parser.add_argument("--no_header", action="store_true",
                        help="Metadata CSV has no header row.")

    # Output
    parser.add_argument("--output",     type=str, required=True,
                        help="Output CSV path (train set).")
    parser.add_argument("--val_output", type=str, default=None,
                        help="Output CSV path for validation set (optional).")
    parser.add_argument("--val_ratio",  type=float, default=0.05,
                        help="Fraction of data to use for validation (default 0.05).")

    # Filtering
    parser.add_argument("--min_duration", type=float, default=0.5,
                        help="Minimum audio duration in seconds (default 0.5).")
    parser.add_argument("--max_duration", type=float, default=15.0,
                        help="Maximum audio duration in seconds (default 15.0).")
    parser.add_argument("--min_text_len", type=int, default=2,
                        help="Minimum transcript character count (default 2).")

    # Voice-cloning prompt (fixed reference for the entire dataset)
    parser.add_argument("--prompt_audio", type=str, default=None,
                        help="Fixed reference WAV for voice-cloning fine-tuning.")
    parser.add_argument("--prompt_text",  type=str, default=None,
                        help="Transcript of --prompt_audio.")

    # Misc
    parser.add_argument("--target_sr", type=int, default=24000,
                        help="Target sample rate (must match model, default 24000).")
    parser.add_argument("--seed",      type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    wav_dir = Path(args.wav_dir)
    assert wav_dir.is_dir(), f"WAV directory not found: {wav_dir}"

    # ── Collect samples ──────────────────────────────────────────────────
    if args.metadata:
        meta_path = Path(args.metadata)
        assert meta_path.exists(), f"Metadata file not found: {meta_path}"
        samples = collect_layout_b(
            wav_dir, meta_path,
            sep=args.sep,
            id_col=args.id_col,
            text_col=args.text_col,
            has_header=not args.no_header,
        )
    else:
        samples = collect_layout_a(wav_dir)

    print(f"Found {len(samples)} raw samples.")

    # ── Filter by duration and text length ───────────────────────────────
    kept = []
    skipped_dur = skipped_txt = 0
    for s in samples:
        text = s["text"].strip()
        if len(text) < args.min_text_len:
            skipped_txt += 1
            continue
        dur = get_audio_duration(s["audio_path"], args.target_sr)
        if dur < args.min_duration or dur > args.max_duration:
            skipped_dur += 1
            continue
        s["duration"] = round(dur, 3)
        s["text"]     = text
        kept.append(s)

    print(f"After filtering: {len(kept)} samples "
          f"(skipped {skipped_dur} by duration, {skipped_txt} by text length).")

    if not kept:
        print("No samples remain after filtering. Check your paths and filters.")
        return

    # ── Add prompt audio (optional, for voice-cloning style training) ────
    if args.prompt_audio:
        assert Path(args.prompt_audio).exists(), \
            f"Prompt audio not found: {args.prompt_audio}"
        for s in kept:
            s["prompt_audio_path"] = args.prompt_audio
            s["prompt_text"]       = args.prompt_text or ""

    # ── Shuffle and split ────────────────────────────────────────────────
    random.shuffle(kept)
    if args.val_output and args.val_ratio > 0:
        n_val  = max(1, int(len(kept) * args.val_ratio))
        n_val  = min(n_val, len(kept) - 1)
        val_samples   = kept[:n_val]
        train_samples = kept[n_val:]
    else:
        train_samples = kept
        val_samples   = []

    # ── Write CSV ────────────────────────────────────────────────────────
    FIELDNAMES = ["audio_path", "text", "duration",
                  "prompt_audio_path", "prompt_text"]

    def write_csv(path: str, rows: list[dict]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                # Fill missing optional columns
                for col in FIELDNAMES:
                    row.setdefault(col, "")
                writer.writerow(row)
        print(f"Written: {path} ({len(rows)} samples)")

    write_csv(args.output, train_samples)
    if val_samples and args.val_output:
        write_csv(args.val_output, val_samples)

    # ── Summary stats ────────────────────────────────────────────────────
    durations = [s["duration"] for s in train_samples if s.get("duration")]
    if durations:
        total_h  = sum(durations) / 3600
        avg_s    = sum(durations) / len(durations)
        print(f"\nTrain set: {len(train_samples)} samples | "
              f"{total_h:.2f} h total | {avg_s:.2f} s avg duration")
    if val_samples:
        print(f"Val   set: {len(val_samples)} samples")


if __name__ == "__main__":
    main()
