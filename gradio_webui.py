import argparse
import os
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


DEFAULT_MODEL = "meituan-longcat/LongCat-AudioDiT-1B"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LongCatAudioDiTWebUI:
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
        use_prompt: bool,
        prompt_text: str | None,
        prompt_audio_path: str | None,
        model_path: str,
        nfe: int,
        guidance_strength: float,
        guidance_method: str,
        seed: int,
    ):
        if not text or not text.strip():
            raise gr.Error("请输入要合成的文本。")

        if use_prompt and not prompt_audio_path:
            raise gr.Error("音色克隆模式需要上传参考音频。")

        self.load_model(model_path)

        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        sr = self.model.config.sampling_rate
        full_hop = self.model.config.latent_hop
        max_duration = self.model.config.max_wav_duration

        text = normalize_text(text.strip())
        prompt_text = normalize_text(prompt_text.strip()) if prompt_text else ""
        full_text = f"{prompt_text} {text}".strip() if use_prompt else text
        inputs = self.tokenizer([full_text], padding="longest", return_tensors="pt")

        prompt_wav = None
        prompt_dur = 0
        if use_prompt:
            prompt_wav = load_audio(prompt_audio_path, sr).unsqueeze(0)

            padded_prompt = load_audio(prompt_audio_path, sr)
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
        if use_prompt and prompt_text:
            approx_prompt_dur = approx_duration_from_text(prompt_text, max_duration=max_duration)
            ratio = np.clip(prompt_time / max(approx_prompt_dur, 1e-6), 1.0, 1.5)
            dur_sec *= ratio

        duration = int(dur_sec * sr // full_hop)
        duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

        output = self.model(
            input_ids=inputs.input_ids.to(DEVICE),
            attention_mask=inputs.attention_mask.to(DEVICE),
            prompt_audio=prompt_wav.to(DEVICE) if prompt_wav is not None else None,
            duration=duration,
            steps=int(nfe),
            cfg_strength=float(guidance_strength),
            guidance_method=guidance_method,
        )

        wav = output.waveform.squeeze().detach().cpu().numpy()
        return sr, wav


webui = LongCatAudioDiTWebUI()


def toggle_prompt_mode(mode: str):
    enabled = mode == "音色克隆"
    return (
        gr.update(visible=enabled),
        gr.update(visible=enabled),
    )


def run_inference(
    text,
    mode,
    prompt_audio,
    prompt_text,
    model_path,
    nfe,
    guidance_strength,
    guidance_method,
    seed,
):
    sr, wav = webui.generate(
        text=text,
        use_prompt=mode == "音色克隆",
        prompt_text=prompt_text,
        prompt_audio_path=prompt_audio,
        model_path=model_path,
        nfe=nfe,
        guidance_strength=guidance_strength,
        guidance_method=guidance_method,
        seed=seed,
    )

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_OUTPUT_DIR / "gradio_output.wav"
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
        title="LongCat-AudioDiT WebUI",
        theme=theme,
        css=css,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(
            """
            <div class="app-title">
              <h1>LongCat-AudioDiT 推理界面</h1>
              <p>文本转语音与零样本音色克隆</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="合成文本",
                    placeholder="输入要合成的文本",
                    lines=5,
                )
                mode = gr.Radio(
                    label="推理模式",
                    choices=["基础 TTS", "音色克隆"],
                    value="基础 TTS",
                )
                prompt_audio = gr.Audio(
                    label="参考音频",
                    type="filepath",
                    visible=False,
                )
                prompt_text = gr.Textbox(
                    label="参考音频文本",
                    placeholder="输入参考音频对应文本，可提升克隆稳定性",
                    lines=2,
                    visible=False,
                )
                generate = gr.Button("生成音频", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Accordion("高级参数", open=True):
                    model_path = gr.Textbox(label="模型路径或 Hugging Face ID", value=default_model)
                    nfe = gr.Slider(label="推理步数 NFE", minimum=1, maximum=64, value=16, step=1)
                    guidance_strength = gr.Slider(
                        label="Guidance Strength",
                        minimum=0.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.1,
                    )
                    guidance_method = gr.Dropdown(
                        label="Guidance Method",
                        choices=["cfg", "apg"],
                        value="cfg",
                    )
                    seed = gr.Number(label="Seed", value=1024, precision=0)

                audio_output = gr.Audio(label="生成结果", type="numpy")
                file_output = gr.File(label="WAV 文件")

        mode.change(toggle_prompt_mode, inputs=mode, outputs=[prompt_audio, prompt_text])
        generate.click(
            run_inference,
            inputs=[
                text,
                mode,
                prompt_audio,
                prompt_text,
                model_path,
                nfe,
                guidance_strength,
                guidance_method,
                seed,
            ],
            outputs=[audio_output, file_output],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch LongCat-AudioDiT Gradio WebUI")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL, help="Model path or Hugging Face ID")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_demo(args.model_dir)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
