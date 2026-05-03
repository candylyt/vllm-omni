# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.moss_tts_decoder import (
    MossTTSDecoderModel,
    _split_streaming_state,
    _stack_streaming_state,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_N_VQ = 3
_SAMPLES_PER_FRAME = 5


def _flat_codes(num_frames: int, *, offset: int = 1) -> torch.Tensor:
    codes = torch.arange(
        offset,
        offset + num_frames * _N_VQ,
        dtype=torch.long,
    )
    return codes.reshape(num_frames, _N_VQ).reshape(-1)


def _minimal_decoder_model(decode_frame):
    model = object.__new__(MossTTSDecoderModel)
    model.device = torch.device("cpu")
    model.n_vq = _N_VQ
    model._streaming_states = {}
    model._active_request_id = None
    model._streaming_modules_cache = None
    model._batch_stateless_decode_failed = False
    model._batch_streaming_decode_failed = False
    model._logged_stateless_batch_decode = False
    model._logged_streaming_batch_decode = False

    codec = SimpleNamespace(
        decoder=[],
        _decode_frame=Mock(side_effect=decode_frame),
    )
    model._codec = SimpleNamespace(
        codec=codec,
        decode=Mock(),
    )
    return model, codec


def _fake_decode_frame(codes: torch.Tensor, lengths: torch.Tensor):
    batch = codes.shape[1]
    max_samples = int(lengths.max().item()) * _SAMPLES_PER_FRAME
    audio = torch.zeros(batch, 1, max_samples, dtype=torch.float32)
    audio_lengths = []
    for i, length in enumerate(lengths.tolist()):
        valid_samples = int(length) * _SAMPLES_PER_FRAME
        audio[i, 0, :valid_samples] = float(i + 1)
        audio_lengths.append(valid_samples)
    return SimpleNamespace(
        audio=audio,
        audio_lengths=torch.tensor(audio_lengths, dtype=torch.long),
    )


def test_stateless_batch_decode_uses_one_codec_forward_for_multiple_requests():
    model, codec = _minimal_decoder_model(_fake_decode_frame)
    inputs = [_flat_codes(2), _flat_codes(4, offset=100)]

    out = MossTTSDecoderModel._batch_decode(model, inputs)

    codec._decode_frame.assert_called_once()
    packed_codes, lengths = codec._decode_frame.call_args[0]
    assert packed_codes.shape == (_N_VQ, 2, 4)
    assert lengths.tolist() == [2, 4]
    assert len(out) == 2
    assert out[0].shape == (2 * _SAMPLES_PER_FRAME,)
    assert out[1].shape == (4 * _SAMPLES_PER_FRAME,)
    model._codec.decode.assert_not_called()


def test_streaming_batch_decode_stacks_and_splits_request_states():
    module = SimpleNamespace(_streaming_state=None)

    def decode_frame(codes: torch.Tensor, lengths: torch.Tensor):
        assert module._streaming_state["cache"].shape == (2, 2)
        module._streaming_state["cache"] = module._streaming_state["cache"] + 10
        return _fake_decode_frame(codes, lengths)

    model, codec = _minimal_decoder_model(decode_frame)
    model._streaming_modules_cache = [module]
    model._streaming_states = {
        "r0": {0: {"cache": torch.zeros(1, 2)}},
        "r1": {0: {"cache": torch.ones(1, 2)}},
    }

    out = MossTTSDecoderModel._batch_decode(
        model,
        [_flat_codes(2), _flat_codes(2, offset=100)],
        request_ids=["r0", "r1"],
        finished_flags=[False, True],
    )

    codec._decode_frame.assert_called_once()
    assert len(out) == 2
    assert out[0].shape == (2 * _SAMPLES_PER_FRAME,)
    assert out[1].shape == (2 * _SAMPLES_PER_FRAME,)
    assert torch.equal(model._streaming_states["r0"][0]["cache"], torch.full((1, 2), 10.0))
    assert "r1" not in model._streaming_states
    assert module._streaming_state is None


def test_streaming_batch_decode_preserves_divergent_scalar_counters():
    module = SimpleNamespace(_streaming_state=None)

    def decode_frame(codes: torch.Tensor, lengths: torch.Tensor):
        assert module._streaming_state.offset_cpu.values == [3, 7]
        module._streaming_state.offset_cpu += int(lengths.max().item())
        module._streaming_state.cache = module._streaming_state.cache + 10
        return _fake_decode_frame(codes, lengths)

    model, codec = _minimal_decoder_model(decode_frame)
    model._streaming_modules_cache = [module]
    model._streaming_states = {
        "r0": {0: SimpleNamespace(cache=torch.zeros(1, 2), offset_cpu=3)},
        "r1": {0: SimpleNamespace(cache=torch.ones(1, 2), offset_cpu=7)},
    }

    out = MossTTSDecoderModel._batch_decode(
        model,
        [_flat_codes(2), _flat_codes(2, offset=100)],
        request_ids=["r0", "r1"],
        finished_flags=[False, False],
    )

    codec._decode_frame.assert_called_once()
    assert len(out) == 2
    assert model._streaming_states["r0"][0].offset_cpu == 5
    assert model._streaming_states["r1"][0].offset_cpu == 9
    assert torch.equal(model._streaming_states["r0"][0].cache, torch.full((1, 2), 10.0))
    assert torch.equal(model._streaming_states["r1"][0].cache, torch.full((1, 2), 11.0))
    assert module._streaming_state is None


def test_streaming_state_stack_split_round_trip_nested_payload():
    states = [
        {"cache": torch.zeros(1, 2), "meta": [torch.ones(1, 1), True, torch.device("cpu")]},
        {"cache": torch.ones(1, 2), "meta": [torch.full((1, 1), 2.0), True, torch.device("cpu")]},
    ]

    stacked = _stack_streaming_state(states)
    assert stacked["cache"].shape == (2, 2)
    assert stacked["meta"][0].shape == (2, 1)

    split = _split_streaming_state(stacked, 2)
    assert torch.equal(split[0]["cache"], states[0]["cache"])
    assert torch.equal(split[1]["cache"], states[1]["cache"])
    assert split[0]["meta"][1] is True
    assert split[1]["meta"][1] is True
    assert split[0]["meta"][2] == torch.device("cpu")
    assert split[1]["meta"][2] == torch.device("cpu")


def test_streaming_state_stack_split_round_trip_divergent_scalar_counter():
    states = [
        SimpleNamespace(cache=torch.zeros(1, 2), offset_cpu=3),
        SimpleNamespace(cache=torch.ones(1, 2), offset_cpu=7),
    ]

    stacked = _stack_streaming_state(states)
    stacked.offset_cpu += 2

    split = _split_streaming_state(stacked, 2)
    assert split[0].offset_cpu == 5
    assert split[1].offset_cpu == 9
    assert torch.equal(split[0].cache, states[0].cache)
    assert torch.equal(split[1].cache, states[1].cache)


def test_divergent_batched_scalar_raises_when_used_as_single_scalar():
    states = [
        SimpleNamespace(offset_cpu=3),
        SimpleNamespace(offset_cpu=7),
    ]

    stacked = _stack_streaming_state(states)

    with pytest.raises(TypeError, match="divergent batched scalar"):
        _ = 1 + stacked.offset_cpu


def test_streaming_state_stack_split_kv_cache_batch_dim_one():
    states = [
        torch.zeros(2, 1, 3, 4, 5),
        torch.ones(2, 1, 3, 4, 5),
    ]

    stacked = _stack_streaming_state(states)

    assert stacked.shape == (2, 2, 3, 4, 5)

    split = _split_streaming_state(stacked, 2)
    assert torch.equal(split[0], states[0])
    assert torch.equal(split[1], states[1])
