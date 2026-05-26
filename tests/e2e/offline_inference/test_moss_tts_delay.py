# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E offline inference test for MOSS-TTS-Delay.

This test intentionally targets the direct-TTS Delay integration only:
MOSS-TTS-Delay Stage 0 -> de-delay stage processor -> MOSS-Audio-Tokenizer
Stage 1.
"""

from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

from pathlib import Path

import pytest
import torch
from vllm import SamplingParams

from tests.utils import hardware_test
from vllm_omni.entrypoints.omni import Omni

MODEL = "OpenMOSS-Team/MOSS-TTS"
STAGE_CONFIG = str(
    Path(__file__).parent.parent.parent.parent
    / "vllm_omni"
    / "model_executor"
    / "stage_configs"
    / "moss_tts_delay.yaml"
)
TEST_TEXT = "The weather is nice today."
MIN_AUDIO_SAMPLES = 1000
SEED = 42
_USER_INST_TEMPLATE = """\
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


def _build_tts_prompt(text: str, model: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
    )
    content = (
        _USER_INST_TEMPLATE.replace("{reference}", "None")
        .replace("{instruction}", "None")
        .replace("{tokens}", "None")
        .replace("{quality}", "None")
        .replace("{sound_event}", "None")
        .replace("{ambient_sound}", "None")
        .replace("{language}", "None")
        .replace("{text}", text)
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "<|audio_start|>"


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "A100"}, num_cards=1)
def test_moss_tts_delay_offline_direct_tts():
    prompt = _build_tts_prompt(TEST_TEXT, MODEL)
    ar_params = SamplingParams(
        temperature=1.5,
        top_p=1.0,
        top_k=50,
        max_tokens=180,
        seed=SEED,
        repetition_penalty=1.0,
    )
    decoder_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=32768,
        seed=SEED,
        detokenize=False,
    )

    omni = Omni(
        model=MODEL,
        stage_configs_path=STAGE_CONFIG,
        init_sleep_seconds=20,
        batch_timeout=5,
        init_timeout=5000,
        shm_threshold_bytes=65536,
    )
    try:
        outputs = list(omni.generate([{"prompt": prompt}], [ar_params, decoder_params]))
    finally:
        omni.close()

    audio_tensor = None
    for stage_output in outputs:
        if stage_output.final_output_type != "audio":
            continue
        output = stage_output.request_output.outputs[0]
        mm = getattr(output, "multimodal_output", None) or {}
        audio_tensor = mm.get("audio")
        if audio_tensor is not None:
            break

    assert audio_tensor is not None, "MOSS-TTS-Delay final stage did not return audio"
    if isinstance(audio_tensor, list):
        chunks = [chunk for chunk in audio_tensor if isinstance(chunk, torch.Tensor) and chunk.numel() > 0]
        assert chunks, "MOSS-TTS-Delay returned only empty audio chunks"
        audio_tensor = torch.cat(chunks, dim=0)
    elif not isinstance(audio_tensor, torch.Tensor):
        audio_tensor = torch.tensor(audio_tensor)

    audio = audio_tensor.float().reshape(-1).cpu()
    assert audio.numel() > MIN_AUDIO_SAMPLES
    assert torch.max(torch.abs(audio)).item() > 0.001
