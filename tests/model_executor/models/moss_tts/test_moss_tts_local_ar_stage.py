# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.moss_tts_local_ar_stage import (
    MossTTSARStageModel,
    _apply_top_p_filter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_top_p_filter_masks_tokens_outside_nucleus():
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0]])

    filtered = _apply_top_p_filter(logits.clone(), 0.8)

    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_compute_logits_forces_audio_start_when_fsm_state_missing():
    model = object.__new__(MossTTSARStageModel)
    model.audio_start_token_id = 4
    model._last_request_ids = []
    model.lm_heads = [torch.nn.Linear(2, 8, bias=False)]
    with torch.no_grad():
        model.lm_heads[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                    [5.0, 0.0],
                    [6.0, 0.0],
                    [7.0, 0.0],
                    [8.0, 0.0],
                ]
            )
        )

    logits = MossTTSARStageModel.compute_logits(model, torch.ones((1, 2)))

    assert torch.isfinite(logits[0, 4])
    masked = torch.cat([logits[0, :4], logits[0, 5:]])
    assert torch.isneginf(masked).all()
