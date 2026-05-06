# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
"""
Stage 0 → Stage 1 bridge processor for MOSS-TTS-Local.

Role
----
At the end of every Stage 0 decode step the AR stage produces a
``code_predictor_codes`` tensor of shape [B, 1, n_vq, 1] = [B, 1, 32, 1].
This module collects those tensors and reshapes them into the flat token
sequence that Stage 1 (CAT codec) expects as its input prompt.

Flat format (row-major / frame-first, matching ``_parse_flat_codes`` in
``moss_tts_decoder.py``).  For T audio frames with 32 RVQ codes each:

    [code_t0_vq0, code_t0_vq1, …, code_t0_vq31,   ← frame 0
     code_t1_vq0, code_t1_vq1, …, code_t1_vq31,   ← frame 1
     …
     code_tT_vq0, …, code_tT_vq31]                ← frame T

i.e. flat shape [T * n_vq].

Two entry points (one per pipeline mode), each wired in via a different
YAML key on the AR stage's engine_args:

    llm2decoder              — batch / sync mode
        Wired via ``custom_process_input_func`` on Stage 1.
        Called once at the end of a full generation.  Reads
        ``stage.engine_outputs`` from Stage 0 and emits one
        ``OmniTokensPrompt`` per request.

    llm2decoder_async_chunk  — streaming / async-chunk mode
        Wired via ``custom_process_next_stage_input_func`` on Stage 0.
        Called after every Stage-0 decode step.  Accumulates frames in the
        SharedMemoryConnector's transfer manager and flushes
        ``codec_chunk_frames`` worth of new frames whenever the buffer fills,
        so Stage 1 can decode incrementally.

Both functions return ``OmniTokensPrompt`` objects whose ``prompt_token_ids``
are the flat code list — they do NOT consolidate codes across requests.
Stage 1 receives one input per request.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)

# Default chunk / context sizes used by ``llm2decoder_async_chunk`` if the
# SharedMemoryConnector config doesn't override them.  In practice
# moss_tts_async.yaml does set ``codec_chunk_frames: 3`` explicitly, so these
# fall-back values match production.  ``_DEFAULT_CONTEXT_FRAMES`` is unused
# by the no-delay variant (left context is always 0) and is kept here only
# for parity with future delay-pattern variants.
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


def _make_finished_sentinel(
    request_id: str,
    chunk_size: int,
) -> dict[str, Any]:
    """Minimal payload signalling Stage 1 to end the request."""
    return {
        "code_predictor_codes": [],
        "code_flat_numel": 0,
        "codec_chunk_frames": chunk_size,
        "left_context_size": 0,
        "request_id": request_id,
        "finished": torch.tensor(True, dtype=torch.bool),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch mode
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
                "skipping (model may not have generated audio).",
                getattr(req_output, "request_id", req_idx),
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
                "[MossTTS processor] Request %s: all-zero codes — skipping.",
                req_idx,
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
#  Streaming / async-chunk mode
# ═══════════════════════════════════════════════════════════════════════════════

def llm2decoder_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """
    Async-chunk processor for MOSS-TTS-Local (no-delay variant).

    Called by Stage 0 (AR stage) after every decode step via
    ``custom_process_next_stage_input_func``.  Accumulates the frame-major
    RVQ rows emitted by the AR stage and flushes ``codec_chunk_frames``
    newly generated frames to Stage 1 (CAT codec) whenever a chunk fills
    up, or when ``is_finished=True``.

    Buffer strategy
    ---------------
    All generated rows are kept in
    ``transfer_manager.code_prompt_token_ids[request_id]`` for the sender's
    internal bookkeeping (the framework's ``cleanup_sender()`` clears it
    after the final chunk).  ``transfer_manager._moss_frames_sent[request_id]``
    tracks how many frames have already been shipped to Stage 1 so that
    only the *new* frames are included in each chunk payload — the codec
    uses a streaming KV-cache across calls, so resending earlier frames
    would produce overlapping / duplicate audio.

    left_context_size = 0
    ---------------------
    ``MossTTSDecoderModel.forward()`` does not trim a left-context region
    from its decoded audio, so we send only the new frames with no overlap.

    Returns a payload dict when enough new frames are ready (or when
    ``is_finished=True`` with any remaining frames), otherwise ``None``.

    Payload keys consumed by MossTTSDecoderModel.forward():
        code_predictor_codes : list[int]    flat codes for this chunk
        code_flat_numel      : int          length of the list
        request_id           : str          used to key streaming KV-cache
        finished             : Tensor(bool)
        codec_chunk_frames   : int          (informational)
        left_context_size    : int          (informational, 0 for MOSS-Local)
    """
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg   = getattr(connector, "config", {}) or {}
    cfg       = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", _DEFAULT_CHUNK_FRAMES))

    request_id = getattr(request, "external_req_id", None)
    if request_id is None:
        return None

    # ── Accumulate this step's frame-major row ────────────────────────
    if isinstance(pooling_output, dict):
        codes_raw = pooling_output.get("code_predictor_codes")
        if not _is_codes_empty(codes_raw):
            row = (
                codes_raw
                if isinstance(codes_raw, torch.Tensor)
                else torch.tensor(codes_raw, dtype=torch.long)
            )
            row = row.to(torch.long).reshape(-1)   # [n_vq]
            if row.numel() > 0:
                transfer_manager.code_prompt_token_ids[request_id].append(row.tolist())

    accumulated = transfer_manager.code_prompt_token_ids[request_id]
    n_frames = len(accumulated)

    # ── Track how many frames have already been sent ──────────────────
    sent_map: dict[str, int] = getattr(transfer_manager, "_moss_frames_sent", None)  # type: ignore[assignment]
    if sent_map is None:
        sent_map = {}
        transfer_manager._moss_frames_sent = sent_map
    frames_sent = sent_map.get(request_id, 0)
    n_new = n_frames - frames_sent

    if n_new <= 0:
        if is_finished:
            sent_map.pop(request_id, None)
            return _make_finished_sentinel(request_id, chunk_size)
        return None

    if not is_finished and n_new < chunk_size:
        return None   # still accumulating

    # ── Decide how many frames to flush this call ─────────────────────
    if is_finished:
        flush_count = n_new
    else:
        flush_count = (n_new // chunk_size) * chunk_size
        if flush_count == 0:
            return None

    flush_frames = accumulated[frames_sent : frames_sent + flush_count]
    flat = [code for frame in flush_frames for code in frame]

    # ── Update sent counter; clean up on final flush ──────────────────
    sent_map[request_id] = frames_sent + flush_count
    if is_finished:
        sent_map.pop(request_id, None)

    return {
        "code_predictor_codes": flat,
        "code_flat_numel":      len(flat),
        "codec_chunk_frames":   chunk_size,
        "left_context_size":    0,   # MOSS-Local has no delay pattern → no left context
        "request_id":           request_id,
        "finished":             torch.tensor(is_finished, dtype=torch.bool),
    }
