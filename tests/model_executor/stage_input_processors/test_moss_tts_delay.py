# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    _apply_de_delay_pattern,
    llm2decoder_delay,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stage(engine_outputs):
    return SimpleNamespace(engine_outputs=engine_outputs)


def _make_request_output(codes, *, audio_pad_code=1024, request_id="rid"):
    return SimpleNamespace(
        request_id=request_id,
        outputs=[
            SimpleNamespace(
                multimodal_output={
                    "code_predictor_codes": codes,
                    "audio_pad_code": audio_pad_code,
                }
            )
        ],
    )


def test_apply_de_delay_pattern_restores_frame_major_rows():
    pad = 99
    delayed = torch.tensor(
        [
            [10, pad, pad],
            [20, 11, pad],
            [30, 21, 12],
            [pad, 31, 22],
            [pad, pad, 32],
            [pad, pad, pad],
        ],
        dtype=torch.long,
    )

    restored = _apply_de_delay_pattern(delayed, audio_pad_code=pad)

    assert restored.tolist() == [
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
    ]


def test_llm2decoder_delay_flattens_restored_codes():
    pad = 99
    delayed = torch.tensor(
        [
            [[10], [pad], [pad]],
            [[20], [11], [pad]],
            [[30], [21], [12]],
            [[pad], [31], [22]],
            [[pad], [pad], [32]],
            [[pad], [pad], [pad]],
        ],
        dtype=torch.long,
    )
    # Shape expected from stage accumulation: [T_steps, n_vq, 1]
    request_output = _make_request_output(delayed, audio_pad_code=pad)
    stage = _make_stage([request_output])

    prompts = llm2decoder_delay([stage], engine_input_source=[0])

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
    ]


def test_llm2decoder_delay_rejects_empty_engine_input_source():
    stage = _make_stage([])
    with pytest.raises(ValueError, match="cannot be empty"):
        llm2decoder_delay([stage], engine_input_source=[])
