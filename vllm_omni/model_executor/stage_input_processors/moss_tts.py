# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
#
# Stage 0 → Stage 1 transition processor for MOSS-TTS-Local.
#
# Role
# ----
# At the end of Stage 0 (AR stage), each decode step produces a
# code_predictor_codes tensor of shape [B, 1, n_vq, 1] = [B, 1, 32, 1].
# This processor accumulates those per-step codes across the full generation,
# then packages them as a flat token sequence for the Stage 1 (CAT codec).
#
# Flat format (row-major, matching _parse_flat_codes in moss_tts_decoder.py):
#
#   For a sequence of T audio frames, each with 32 RVQ codes:
#   [code_t0_vq0, code_t0_vq1, …, code_t0_vq31,   ← frame 0
#    code_t1_vq0, code_t1_vq1, …, code_t1_vq31,   ← frame 1
#    …
#    code_tT_vq0, …, code_tT_vq31]                 ← frame T
#
#   i.e. flat shape [T * n_vq]
#
# Two entry points (mirroring mimo_audio.py):
#   llm2decoder()            — batch mode  (called once at end of generation)
#   llm2decoder_async_chunk() — streaming  (called every N frames; future use)

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)

# Default chunk / context sizes for streaming (post-MVP).
_DEFAULT_CHUNK_FRAMES   = 3
_DEFAULT_CONTEXT_FRAMES = 3


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_codes_empty(codes: Any) -> bool:
    """Return True if the codes payload should be treated as empty / invalid."""
    if codes is None:
        return True
    if isinstance(codes, torch.Tensor):
        return codes.numel() == 0 or not codes.any()
    if hasattr(codes, "__len__") and len(codes) == 0:
        return True
    t = (
        codes
        if isinstance(codes, torch.Tensor)
        else torch.tensor(codes, dtype=torch.long)
    )
    return not t.any()


def _codes_to_flat_list(codes: Any, n_vq: int) -> list[int] | None:
    """
    Convert code_predictor_codes from the AR stage output to a flat list.

    Input can be:
      - Tensor  [B, 1, n_vq, 1]  (standard batch output, B=1 per request)
      - Tensor  [n_vq]           (already squeezed)
      - list / other             (converted first)

    Returns a flat list of length n_vq, or None if invalid.
    """
    if not isinstance(codes, torch.Tensor):
        codes = torch.tensor(codes, dtype=torch.long)

    codes = codes.to(torch.long).reshape(-1)  # flatten everything
    if codes.numel() != n_vq:
        logger.debug(
            "[MossTTS processor] Unexpected codes shape after reshape: %d "
            "(expected n_vq=%d)",
            codes.numel(),
            n_vq,
        )
        # Accept if it's a multiple of n_vq (take the first n_vq values)
        if codes.numel() >= n_vq:
            codes = codes[:n_vq]
        else:
            return None

    return codes.tolist()


def _make_finished_sentinel() -> dict[str, Any]:
    """Minimal payload signalling Stage 1 to end the request."""
    return {"code_predictor_codes": [], "finished": torch.tensor(True, dtype=torch.bool)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch mode  (mandatory for MVP)
# ═══════════════════════════════════════════════════════════════════════════════

def llm2decoder(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Convert Stage-0 AR outputs to Stage-1 CAT-codec inputs.

    Called once at the end of a full generation (batch / non-streaming mode).

    Workflow
    --------
    1. Collect all code_predictor_codes tensors from Stage 0's engine_outputs.
       Each output carries codes for one generation step, shape [B, 1, n_vq, 1].
    2. Concatenate across steps → [T, n_vq] per request.
    3. Flatten in row-major order → [T * n_vq] per request.
    4. Wrap each in an OmniTokensPrompt for Stage 1.

    Parameters
    ----------
    stage_list          : list of stage objects (injected by vllm-omni runtime)
    engine_input_source : [stage_id] of the AR stage (typically [0])
    prompt              : original prompt (unused here)
    requires_multimodal_data : unused for TTS

    Returns
    -------
    list of OmniTokensPrompt, one per active request
    """
    if not engine_input_source:
        raise ValueError("[MossTTS processor] engine_input_source cannot be empty.")

    src_stage_id = engine_input_source[0]
    if src_stage_id >= len(stage_list):
        raise IndexError(
            f"[MossTTS processor] Invalid stage_id={src_stage_id} "
            f"(only {len(stage_list)} stages present)."
        )

    stage = stage_list[src_stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(
            f"[MossTTS processor] Stage {src_stage_id} has no outputs yet."
        )

    ar_outputs = stage.engine_outputs   # list[RequestOutput]
    decoder_inputs: list[OmniTokensPrompt] = []

    for req_idx, req_output in enumerate(ar_outputs):
        output = req_output.outputs[0]
        mm_out = output.multimodal_output or {}

        codes_raw = mm_out.get("code_predictor_codes")

        if _is_codes_empty(codes_raw):
            logger.warning(
                "[MossTTS processor] Request %s: empty code_predictor_codes — "
                "forwarding empty decoder input.",
                getattr(req_output, "request_id", req_idx),
            )
            decoder_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            )
            continue

        # codes_raw: Tensor [T_steps, 1, n_vq, 1]  or  [T_steps, n_vq]
        # (shape depends on how the AR stage stacked steps)
        codes = codes_raw.to(torch.long)

        # Normalise to [T, n_vq]
        if codes.ndim == 4:
            # [T, 1, n_vq, 1] → [T, n_vq]
            codes = codes.squeeze(1).squeeze(-1)
        elif codes.ndim == 3:
            # [T, n_vq, 1] → [T, n_vq]
            codes = codes.squeeze(-1)
        elif codes.ndim == 2:
            pass   # already [T, n_vq]
        else:
            logger.warning(
                "[MossTTS processor] Unexpected codes ndim=%d for request %s — skipping.",
                codes.ndim,
                req_idx,
            )
            continue

        # Skip all-zero sequences (dummy output)
        if not codes.any():
            logger.warning(
                "[MossTTS processor] Request %s: all-zero codes — "
                "forwarding empty decoder input.",
                req_idx,
            )
            decoder_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            )
            continue

        n_vq = codes.shape[1]

        # Flatten: [T, n_vq] → [T * n_vq]  (row-major = frame-first)
        flat = codes.reshape(-1).tolist()

        decoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return decoder_inputs


# ═══════════════════════════════════════════════════════════════════════════════
#  Streaming / async-chunk mode  (post-MVP, wired but not fully active)
# ═══════════════════════════════════════════════════════════════════════════════

def llm2decoder_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """
    Async-chunk version: accumulate per-step codes and flush every chunk_frames steps.

    Returns a payload dict when a full chunk is ready (or when is_finished=True),
    otherwise returns None to signal "still accumulating".

    Payload keys consumed by MossTTSDecoderModel.forward():
        code_predictor_codes : list[int]    flat codes for this chunk
        code_flat_numel      : int          length of the list
        finished             : Tensor(bool)
        codec_chunk_frames   : int          (informational)
        left_context_size    : int          (informational, 0 for MOSS — no delay)
    """
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg   = getattr(connector, "config", {}) or {}
    cfg       = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size  = int(cfg.get("codec_chunk_frames",   _DEFAULT_CHUNK_FRAMES))

    request_id = getattr(request, "external_req_id", None)

    codes_raw = pooling_output.get("code_predictor_codes")

    # ── Nothing to flush ──────────────────────────────────────────────
    if _is_codes_empty(codes_raw):
        if is_finished:
            return _flush_remaining(transfer_manager, request_id, chunk_size)
        return None

    # ── Convert to per-step flat list ─────────────────────────────────
    codes = (
        codes_raw
        if isinstance(codes_raw, torch.Tensor)
        else torch.tensor(codes_raw, dtype=torch.long)
    )
    codes = codes.to(torch.long).reshape(-1)  # [n_vq]
    n_vq  = codes.numel()

    if n_vq == 0:
        if is_finished:
            return _flush_remaining(transfer_manager, request_id, chunk_size)
        return None

    if request_id is None:
        return None

    # Accumulate this frame's codes
    transfer_manager.code_prompt_token_ids[request_id].append(codes.tolist())
    accumulated = transfer_manager.code_prompt_token_ids[request_id]
    n_frames    = len(accumulated)

    # Flush when chunk is full or generation is done
    if n_frames % chunk_size != 0 and not is_finished:
        return None   # still waiting

    # Build flush payload
    flat  = [code for frame in accumulated for code in frame]
    numel = len(flat)
    return {
        "code_predictor_codes": flat,
        "code_flat_numel":      numel,
        "codec_chunk_frames":   chunk_size,
        "left_context_size":    0,   # MOSS has no delay pattern → no left context
        "finished":             torch.tensor(is_finished, dtype=torch.bool),
    }


def _flush_remaining(
    transfer_manager: Any,
    request_id: str | None,
    chunk_size: int,
) -> dict[str, Any]:
    """Flush any leftover codes when the request finishes mid-chunk."""
    if request_id is None:
        return _make_finished_sentinel()

    accumulated = transfer_manager.code_prompt_token_ids.get(request_id, [])
    if not accumulated:
        return _make_finished_sentinel()

    flat  = [code for frame in accumulated for code in frame]
    numel = len(flat)
    return {
        "code_predictor_codes": flat,
        "code_flat_numel":      numel,
        "codec_chunk_frames":   chunk_size,
        "left_context_size":    0,
        "finished":             torch.tensor(True, dtype=torch.bool),
    }
