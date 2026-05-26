# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.moss_tts_local_decoder import (
    MossTTSDecoderModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_decode_one_request_decodes_all_zero_rvq_codes():
    model = object.__new__(MossTTSDecoderModel)
    model.n_vq = 3
    model.device = torch.device("cpu")
    model._streaming_states = {}
    model._batched_streaming_stack = None
    model._batched_streaming_request_ids = None
    model._first_chunk_dir = None
    model._first_chunk_seen = set()

    wav = torch.ones(7, dtype=torch.float32)
    model._codec = SimpleNamespace(decode=Mock(return_value=wav))

    out = MossTTSDecoderModel._decode_one_request(
        model,
        torch.zeros(6, dtype=torch.long),
    )

    model._codec.decode.assert_called_once()
    assert torch.equal(out, wav)
