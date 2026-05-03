# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.moss_tts_decoder import (
    _parse_flat_codes,
)
from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    _apply_de_delay_pattern,
    llm2decoder_delay,
    llm2decoder_delay_async_chunk,
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


def _make_transfer_manager(*, chunk_frames=2):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": chunk_frames}}),
    )


def _make_request(request_id="rid"):
    return SimpleNamespace(external_req_id=request_id)


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
        10,
        11,
        12,
        20,
        21,
        22,
        30,
        31,
        32,
    ]


def test_llm2decoder_delay_output_matches_stage1_flat_code_contract():
    pad = 99
    delayed = torch.tensor(
        [
            [[10], [pad], [pad]],
            [[20], [11], [pad]],
            [[30], [21], [12]],
            [[pad], [31], [22]],
            [[pad], [pad], [32]],
        ],
        dtype=torch.long,
    )
    request_output = _make_request_output(delayed, audio_pad_code=pad)
    prompts = llm2decoder_delay([_make_stage([request_output])], engine_input_source=[0])

    flat = torch.tensor(prompts[0]["prompt_token_ids"], dtype=torch.long)
    stage1_codes = _parse_flat_codes(flat, n_vq=3)

    assert stage1_codes is not None
    assert stage1_codes.tolist() == [
        [10, 20, 30],
        [11, 21, 31],
        [12, 22, 32],
    ]


def test_llm2decoder_delay_async_chunk_flushes_only_new_dedelayed_frames():
    pad = 99
    tm = _make_transfer_manager(chunk_frames=2)
    req = _make_request("rid")
    delay_rows = [
        [10, pad, pad],
        [20, 11, pad],
        [30, 21, 12],
        [pad, 31, 22],
        [pad, pad, 32],
    ]

    payloads = []
    for row in delay_rows:
        payloads.append(
            llm2decoder_delay_async_chunk(
                tm,
                {"code_predictor_codes": torch.tensor(row), "audio_pad_code": pad},
                req,
                is_finished=False,
            )
        )

    assert payloads[:3] == [None, None, None]
    assert payloads[3] is not None
    assert payloads[3]["code_predictor_codes"] == [
        10,
        11,
        12,
        20,
        21,
        22,
    ]
    assert payloads[3]["code_flat_numel"] == 6
    assert payloads[3]["finished"] == torch.tensor(False, dtype=torch.bool)
    assert payloads[4] is None

    final_payload = llm2decoder_delay_async_chunk(
        tm,
        None,
        req,
        is_finished=True,
    )

    assert final_payload is not None
    assert final_payload["code_predictor_codes"] == [30, 31, 32]
    assert final_payload["request_id"] == "rid"
    assert final_payload["finished"] == torch.tensor(True, dtype=torch.bool)
    assert getattr(tm, "_delay_frames_sent", {}) == {}
    assert getattr(tm, "_moss_tts_delay_audio_pad_code", {}) == {}


def test_llm2decoder_delay_async_final_sentinel_includes_request_id_with_no_remaining_frames():
    pad = 99
    tm = _make_transfer_manager(chunk_frames=1)
    req = _make_request("rid")

    payloads = []
    for row in [
        [10, pad, pad],
        [20, 11, pad],
        [30, 21, 12],
    ]:
        payloads.append(
            llm2decoder_delay_async_chunk(
                tm,
                {"code_predictor_codes": torch.tensor(row), "audio_pad_code": pad},
                req,
                is_finished=False,
            )
        )

    assert payloads[-1] is not None
    assert payloads[-1]["request_id"] == "rid"
    assert payloads[-1]["code_predictor_codes"] == [10, 11, 12]

    final_payload = llm2decoder_delay_async_chunk(
        tm,
        None,
        req,
        is_finished=True,
    )

    assert final_payload is not None
    assert final_payload["code_predictor_codes"] == []
    assert final_payload["request_id"] == "rid"
    assert final_payload["finished"] == torch.tensor(True, dtype=torch.bool)


def test_llm2decoder_delay_rejects_empty_engine_input_source():
    stage = _make_stage([])
    with pytest.raises(ValueError, match="cannot be empty"):
        llm2decoder_delay([stage], engine_input_source=[])
