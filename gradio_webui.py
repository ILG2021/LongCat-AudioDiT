import argparse
import os
import threading
import uuid
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audiodit import AudioDiTModel
from utils import approx_duration_from_text, load_audio, normalize_text


DEFAULT_MODEL = "meituan-longcat/LongCat-AudioDiT-3.5B"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VoiceSynthesisEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None

    def load_model(self, model_path: str):
        model_path = model_path.strip() or DEFAULT_MODEL
        if self.model is not None and self.model_path == model_path:
            return

        self.model = AudioDiTModel.from_pretrained(model_path).to(DEVICE)
        if DEVICE.type == "cuda":
            self.model.vae.to_half()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder_model)
        self.model_path = model_path

    @torch.no_grad()
    def generate(
        self,
        text: str,
        use_reference: bool,
        reference_text: str | None,
        reference_audio_path: str | None,
        model_path: str,
        quality_steps: int,
        guidance_strength: float,
        guidance_method: str,
        seed: int,
    ):
        if not text or not text.strip():
            raise gr.Error("请输入要合成的文本。")

        if use_reference and not reference_audio_path:
            raise gr.Error("使用参考声音时需要上传一段参考音频。")

        self.load_model(model_path)

        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        sr = self.model.config.sampling_rate
        full_hop = self.model.config.latent_hop
        max_duration = self.model.config.max_wav_duration

        text = normalize_text(text.strip())
        reference_text = normalize_text(reference_text.strip()) if reference_text else ""
        full_text = f"{reference_text} {text}".strip() if use_reference else text
        inputs = self.tokenizer([full_text], padding="longest", return_tensors="pt")

        prompt_wav = None
        prompt_dur = 0
        if use_reference:
            prompt_wav = load_audio(reference_audio_path, sr).unsqueeze(0)

            padded_prompt = load_audio(reference_audio_path, sr)
            if padded_prompt.shape[-1] % full_hop != 0:
                padded_prompt = F.pad(
                    padded_prompt,
                    (0, full_hop - padded_prompt.shape[-1] % full_hop),
                )
            off = 3
            padded_prompt = F.pad(padded_prompt, (0, full_hop * off))
            prompt_latent = self.model.vae.encode(padded_prompt.unsqueeze(0).to(DEVICE))
            if off:
                prompt_latent = prompt_latent[..., :-off]
            prompt_dur = prompt_latent.shape[-1]

        prompt_time = prompt_dur * full_hop / sr
        dur_sec = approx_duration_from_text(text, max_duration=max_duration - prompt_time)
        if use_reference and reference_text:
            approx_prompt_dur = approx_duration_from_text(reference_text, max_duration=max_duration)
            ratio = np.clip(prompt_time / max(approx_prompt_dur, 1e-6), 1.0, 1.5)
            dur_sec *= ratio

        duration = int(dur_sec * sr // full_hop)
        duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))
        output = self.model(
            input_ids=inputs.input_ids.to(DEVICE),
            attention_mask=inputs.attention_mask.to(DEVICE),
            prompt_audio=prompt_wav.to(DEVICE) if prompt_wav is not None else None,
            duration=duration,
            steps=int(quality_steps),
            cfg_strength=float(guidance_strength),
            guidance_method=guidance_method,
        )

        wav = output.waveform.squeeze().detach().cpu().numpy()
        return sr, wav


engine = VoiceSynthesisEngine()
transcription_model = None
generation_lock = threading.Lock()
transcription_lock = threading.Lock()


def get_transcription_model():
    global transcription_model
    if transcription_model is not None:
        return transcription_model

    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    transcription_model = WhisperModel("turbo", device=device, compute_type=compute_type)
    return transcription_model


def transcribe_reference_audio(audio_path):
    if not audio_path:
        return ""

    try:
        with transcription_lock:
            model = get_transcription_model()
            segments, _ = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = "".join(segment.text for segment in segments).strip()
        if not text:
            raise gr.Error("没有识别到参考音频中的语音内容。")
        return text
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"自动识别参考音频失败：{exc}") from exc


def toggle_reference_mode(mode: str):
    enabled = mode == "参考声音"
    return gr.update(visible=enabled), gr.update(visible=enabled)


def run_inference(
    text,
    mode,
    reference_audio,
    reference_text,
    model_path,
    quality_steps,
    guidance_strength,
    guidance_method,
    seed,
):
    with generation_lock:
        sr, wav = engine.generate(
            text=text,
            use_reference=mode == "参考声音",
            reference_text=reference_text,
            reference_audio_path=reference_audio,
            model_path=model_path,
            quality_steps=quality_steps,
            guidance_strength=guidance_strength,
            guidance_method=guidance_method,
            seed=seed,
        )

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_OUTPUT_DIR / f"voice_{uuid.uuid4().hex}.wav"
    sf.write(output_path, wav, sr)
    return (sr, wav), str(output_path)


def build_demo(default_model: str):
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="emerald", neutral_hue="zinc")
    css = """
    .app-title { text-align: center; margin: 0 0 18px; }
    .app-title h1 { font-size: 34px; line-height: 1.15; margin: 0 0 8px; }
    .app-title p { color: var(--body-text-color-subdued); margin: 0; }
    """

    with gr.Blocks(
        title="语音合成工作台",
        theme=theme,
        css=css,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(
            """
            <div class="app-title">
              <h1>TT工作台</h1>
              <p>输入文本，生成自然清晰的语音</p>
            </div>
            """
        )

        model_path = gr.State(default_model)

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="合成文本",
                    placeholder="输入要合成的文本",
                    lines=5,
                )
                mode = gr.Radio(
                    label="声音来源",
                    choices=["默认声音", "参考声音"],
                    value="默认声音",
                )
                reference_audio = gr.Audio(
                    label="参考音频",
                    type="filepath",
                    visible=False,
                )
                reference_text = gr.Textbox(
                    label="参考音频文本",
                    placeholder="输入参考音频中说的话，可提升生成稳定性",
                    lines=2,
                    visible=False,
                )
                generate = gr.Button("生成音频", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Accordion("高级参数", open=False):
                    quality_steps = gr.Slider(label="推理步数 NFE", minimum=1, maximum=64, value=16, step=1)
                    guidance_strength = gr.Slider(
                        label="引导强度",
                        minimum=0.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.1,
                    )
                    guidance_method = gr.Dropdown(
                        label="引导方式",
                        choices=["cfg", "apg"],
                        value="apg",
                    )
                    seed = gr.Number(label="Seed", value=1024, precision=0)

                audio_output = gr.Audio(label="生成结果", type="numpy")
                file_output = gr.File(label="音频文件")

        mode.change(toggle_reference_mode, inputs=mode, outputs=[reference_audio, reference_text])
        reference_audio.change(
            transcribe_reference_audio,
            inputs=reference_audio,
            outputs=reference_text,
        )
        generate.click(
            run_inference,
            inputs=[
                text,
                mode,
                reference_audio,
                reference_text,
                model_path,
                quality_steps,
                guidance_strength,
                guidance_method,
                seed,
            ],
            outputs=[audio_output, file_output],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch voice synthesis web interface")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL, help="Backend model path")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_demo(args.model_dir)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
