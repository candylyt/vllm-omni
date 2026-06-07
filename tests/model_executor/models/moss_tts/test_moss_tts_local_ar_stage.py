# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.moss_tts_local_ar_stage import (
    MossTTSARStageModel,
    MossTTSLocalKVCache,
    MossTTSLocalRequestState,
    MossTTSNativeLocalTransformer,
    _apply_top_p_filter,
    _select_local_cudagraph_bucket,
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


def test_native_local_transformer_recompute_forward_shape():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
    )
    model = MossTTSNativeLocalTransformer(config)

    hidden, past = model(torch.randn(2, 3, 16))

    assert hidden.shape == (2, 3, 16)
    assert past is None


def test_native_local_transformer_rejects_non_incremental_cache_path():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
    )
    model = MossTTSNativeLocalTransformer(config)

    with pytest.raises(RuntimeError, match="single-token incremental"):
        model(torch.randn(1, 2, 16), use_cache=True)


def test_native_local_transformer_rejects_cache_without_use_cache():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
    )
    model = MossTTSNativeLocalTransformer(config)

    with pytest.raises(RuntimeError, match="use_cache=False"):
        model(
            torch.randn(1, 1, 16),
            past_key_values=MossTTSLocalKVCache(config.num_hidden_layers),
            use_cache=False,
        )


def test_native_local_transformer_incremental_cache_matches_last_token_recompute():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
    )
    model = MossTTSNativeLocalTransformer(config)
    inputs = torch.randn(2, 4, 16)

    full_hidden, _ = model(inputs)

    cache = None
    step_hidden = None
    for pos in range(inputs.shape[1]):
        step_hidden, cache = model(
            inputs[:, pos : pos + 1, :],
            past_key_values=cache,
            use_cache=True,
        )

    assert cache is not None
    assert cache.get_seq_length() == inputs.shape[1]
    assert step_hidden is not None
    torch.testing.assert_close(step_hidden[:, -1, :], full_hidden[:, -1, :])


def test_request_state_keeps_pending_audio_row_on_input_device():
    state = MossTTSLocalRequestState(n_vq=4, audio_pad_code=1024)
    row = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    state.store_next_audio_row(row)

    assert state.pending_audio_row.device == row.device
    assert state.pending_audio_row.tolist() == [1, 2, 3, 4]


def test_local_cudagraph_bucket_selection_covers_32():
    buckets = {1, 2, 4, 8, 16, 32}

    assert _select_local_cudagraph_bucket(1, buckets) == 1
    assert _select_local_cudagraph_bucket(3, buckets) == 4
    assert _select_local_cudagraph_bucket(17, buckets) == 32
    assert _select_local_cudagraph_bucket(32, buckets) == 32
    assert _select_local_cudagraph_bucket(33, buckets) is None


def test_local_forward_cpu_uses_eager_fallback(monkeypatch):
    model = object.__new__(MossTTSARStageModel)
    model._local_forward_cg_by_bs = {1: object()}
    model._local_forward_cg_params = None
    model._local_forward_cg_init_attempted = False

    eager_calls = []

    def fake_eager(global_hidden, forced_text_token_id, temperature, top_k, top_p):
        eager_calls.append((forced_text_token_id, temperature, top_k, top_p))
        return (
            torch.ones((global_hidden.shape[0], 4), dtype=torch.long),
            torch.zeros((global_hidden.shape[0], 8)),
        )

    def fail_capture(**kwargs):
        raise AssertionError("CPU fallback must not try to capture CUDA graphs")

    monkeypatch.setattr(model, "_local_forward_eager", fake_eager)
    monkeypatch.setattr(model, "_capture_local_forward_cudagraphs", fail_capture)

    codes, text_logits = MossTTSARStageModel._local_forward(
        model,
        torch.randn(2, 6),
        forced_text_token_id=5,
        temperature=0.7,
        top_k=10,
        top_p=0.9,
    )

    assert eager_calls == [(5, 0.7, 10, 0.9)]
    assert codes.shape == (2, 4)
    assert text_logits.shape == (2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_forward_cudagraph_replay_path_with_stub(monkeypatch):
    model = object.__new__(MossTTSARStageModel)
    torch.nn.Module.__init__(model)
    model.hidden_size = 4
    model.speech_embedding_to_local_mlp = torch.nn.Linear(4, 4, bias=False).cuda()
    model._local_forward_cg_by_bs = {}
    model._local_forward_cg_params = None
    model._local_forward_cg_init_attempted = False

    def fake_eager(global_hidden, forced_text_token_id, temperature, top_k, top_p):
        del forced_text_token_id, temperature, top_k, top_p
        codes = global_hidden[:, :2].round().to(torch.long)
        text_logits = global_hidden * 2
        return codes, text_logits

    monkeypatch.setattr(model, "_local_forward_eager", fake_eager)

    hidden = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        device="cuda",
    )

    codes, text_logits = MossTTSARStageModel._local_forward(
        model,
        hidden,
        forced_text_token_id=5,
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
    )

    assert set(model._local_forward_cg_by_bs) == {1, 2, 4, 8, 16, 32}
    torch.testing.assert_close(codes.cpu(), torch.tensor([[1, 2], [5, 6]]))
    torch.testing.assert_close(text_logits.cpu(), (hidden * 2).cpu())
