"""Shared helpers for MOSS-TTS experiment scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer


USER_INST_TEMPLATE = """\
<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""


FIXED_TEXT = "The weather is so nice today and the birds are singing in the trees."
SAMPLE_RATE = 24_000


def build_tts_prompt(text: str, model_path: str) -> str:
    """Build the direct-TTS prompt used in the reported experiments."""
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.abspath(model_path),
        trust_remote_code=True,
    )
    content = USER_INST_TEMPLATE.format(
        reference="None",
        instruction="None",
        tokens="None",
        quality="None",
        sound_event="None",
        ambient_sound="None",
        language="None",
        text=str(text),
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "<|audio_start|>"


def get_stop_ids(model_path: str) -> list[int]:
    """Return audio-end/eos stop IDs for Local AR generation."""
    cfg = AutoConfig.from_pretrained(
        os.path.abspath(model_path),
        trust_remote_code=True,
    )
    audio_end_id = getattr(cfg, "audio_end_token_id", None)
    eos_id = getattr(cfg, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_ids = eos_id
    elif eos_id is not None:
        eos_ids = [eos_id]
    else:
        eos_ids = []
    return list(dict.fromkeys(([audio_end_id] if audio_end_id is not None else []) + eos_ids))


def stage_config_name(model_type: str, mode: str) -> str:
    if model_type == "local":
        return "moss_tts_local_async.yaml" if mode == "async" else "moss_tts_local.yaml"
    if model_type == "delay":
        return "moss_tts_delay_async.yaml" if mode == "async" else "moss_tts_delay.yaml"
    raise ValueError(f"unknown model_type: {model_type}")


def find_stage_config(repo: str, model_type: str, mode: str) -> str:
    cfg = Path(repo) / "vllm_omni" / "model_executor" / "stage_configs" / stage_config_name(model_type, mode)
    if not cfg.exists():
        raise FileNotFoundError(f"missing stage config: {cfg}")
    return str(cfg)


def concat_audio_tensor(audio: Any) -> torch.Tensor | None:
    """Normalize a Stage 1 audio payload to one tensor."""
    if audio is None:
        return None
    if isinstance(audio, list):
        chunks = [x for x in audio if isinstance(x, torch.Tensor) and x.numel() > 0]
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)
    if isinstance(audio, torch.Tensor):
        return audio
    return None


def audio_duration_seconds(audio: torch.Tensor) -> float:
    return float(audio.float().detach().cpu().numpy().reshape(-1).shape[0]) / SAMPLE_RATE


def maybe_init_wandb(project: str, name: str, config: dict[str, Any]):
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None
    try:
        import wandb

        return wandb.init(project=project, name=name, config=config)
    except Exception as exc:  # pragma: no cover - logging fallback
        print(f"[warn] W&B disabled because init failed: {exc}")
        return None


def wandb_log(run: Any, payload: dict[str, Any]) -> None:
    if run is not None:
        run.log(payload)


def wandb_finish(run: Any) -> None:
    if run is not None:
        run.finish()
