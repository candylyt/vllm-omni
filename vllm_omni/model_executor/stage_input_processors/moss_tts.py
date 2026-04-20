# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
#
# Stage 0 → Stage 1 transition processors for MOSS-TTS.
#
# Role
# ----
# At the end of Stage 0 (AR stage), each decode step produces a
# code_predictor_codes tensor of shape [B, 1, n_vq, 1] = [B, 1, 32, 1].
# This processor accumulates those per-step codes across the full generation,
# then packages them as a flat token sequence for the Stage 1 (CAT codec).
#
# Variants:
#   - Local: Stage 0 already emits frame-major [T, n_vq] rows.
#   - Delay: Stage 0 emits delay-pattern rows that must be de-delayed before
#            flattening for the shared CAT decoder.
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
_DEFAULT_AUDIO_PAD_CODE = 1024


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


def _has_no_codes(codes: Any) -> bool:
    """Treat only None / zero-length payloads as missing.

    Audio code value 0 is valid for MOSS, so unlike `_is_codes_empty` we must
    not treat all-zero rows as empty data.
    """
    if codes is None:
        return True
    if isinstance(codes, torch.Tensor):
        return codes.numel() == 0
    if hasattr(codes, "__len__"):
        return len(codes) == 0
    return False


def _normalize_codes_tensor(codes_raw: Any) -> torch.Tensor | None:
    """Normalise stacked stage outputs to [T, n_vq]."""
    codes = codes_raw if isinstance(codes_raw, torch.Tensor) else torch.tensor(codes_raw, dtype=torch.long)
    codes = codes.to(torch.long)

    if codes.ndim == 4:
        return codes.squeeze(1).squeeze(-1)
    if codes.ndim == 3:
        return codes.squeeze(-1)
    if codes.ndim == 2:
        return codes
    return None


def _apply_de_delay_pattern(
    codes: torch.Tensor,
    *,
    audio_pad_code: int = _DEFAULT_AUDIO_PAD_CODE,
) -> torch.Tensor:
    """Invert the diagonal delay schedule back to frame-major [T, n_vq].

    Input rows follow the upstream Delay pattern:
      delayed[t, q] = frame[t - q, q]  (pad when out of range)

    So the original frame-major codes are recovered by:
      frame[f, q] = delayed[f + q, q]

    A frame is *valid* iff ALL n_vq positions contain actual codes (no pad).
    Pad appears in two cases:
      (1) Ramp-up rows (first ~n_vq delay-slot steps) where the AR model
          emits pad because no quantizer is active yet.
      (2) Trailing rows where the buffer lacks enough future delay steps to
          fill all quantizer positions (f + n_vq > total_steps).

    We use the stricter .any() check (invalid if ANY code is pad) rather
    than .all() (invalid only if ALL codes are pad).  The .all() check misses
    partially-filled frames which would send pad=audio_pad_code to the codec
    and trigger an out-of-bounds gather assertion (codebook index 1024 ≥
    codebook_size 1024).

    audio_pad_code (1024) is explicitly masked to -inf during AR sampling, so
    it never appears as a real audio token — the .any() filter is lossless.
    """
    if codes.ndim != 2:
        raise ValueError(f"Expected [T, n_vq] codes, got shape {tuple(codes.shape)}")

    total_steps, n_vq = codes.shape
    restored = torch.full_like(codes, fill_value=audio_pad_code)

    for ch_idx in range(n_vq):
        n_elems = total_steps - ch_idx
        if n_elems > 0:
            restored[:n_elems, ch_idx] = codes[ch_idx : ch_idx + n_elems, ch_idx]

    # Keep only rows where every quantizer position holds an actual code.
    valid_rows = ~(restored == audio_pad_code).any(dim=1)
    if not valid_rows.any():
        return restored[:0]
    return restored[valid_rows]


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


def llm2decoder_delay(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Convert Delay-stage outputs to decoder-ready flat CAT codes.

    The Delay AR stage emits diagonally shifted RVQ rows. Before Stage 1 can
    decode them, we must invert that schedule back to ordinary frame-major
    [T, n_vq] codes, then flatten row-major.
    """
    if not engine_input_source:
        raise ValueError("[MossTTS Delay processor] engine_input_source cannot be empty.")

    src_stage_id = engine_input_source[0]
    if src_stage_id >= len(stage_list):
        raise IndexError(
            f"[MossTTS Delay processor] Invalid stage_id={src_stage_id} "
            f"(only {len(stage_list)} stages present)."
        )

    stage = stage_list[src_stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(
            f"[MossTTS Delay processor] Stage {src_stage_id} has no outputs yet."
        )

    decoder_inputs: list[OmniTokensPrompt] = []

    for req_idx, req_output in enumerate(stage.engine_outputs):
        output = req_output.outputs[0]
        mm_out = output.multimodal_output or {}
        codes_raw = mm_out.get("code_predictor_codes")

        if _has_no_codes(codes_raw):
            logger.warning(
                "[MossTTS Delay processor] Request %s: no code_predictor_codes — skipping.",
                getattr(req_output, "request_id", req_idx),
            )
            continue

        codes = _normalize_codes_tensor(codes_raw)
        if codes is None:
            logger.warning(
                "[MossTTS Delay processor] Unexpected codes shape for request %s: %s",
                getattr(req_output, "request_id", req_idx),
                tuple(codes_raw.shape) if isinstance(codes_raw, torch.Tensor) else type(codes_raw),
            )
            continue

        restored = _apply_de_delay_pattern(
            codes,
            audio_pad_code=int(mm_out.get("audio_pad_code", _DEFAULT_AUDIO_PAD_CODE)),
        )
        if restored.numel() == 0:
            logger.warning(
                "[MossTTS Delay processor] Request %s: de-delayed codes are empty — skipping.",
                getattr(req_output, "request_id", req_idx),
            )
            continue

        decoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=restored.reshape(-1).tolist(),
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
        "request_id":           request_id,
        "finished":             torch.tensor(True, dtype=torch.bool),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Delay-model streaming / async-chunk mode
# ═══════════════════════════════════════════════════════════════════════════════

def llm2decoder_delay_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """
    Async-chunk processor for MOSS-TTS-Delay.

    Called by Stage 0 (delay AR) after every decode step via
    ``custom_process_next_stage_input_func``.  Accumulates the raw
    delay-pattern rows emitted by the AR stage, recovers valid frame-major
    codes by applying the inverse delay transform on the growing buffer,
    and flushes a payload to Stage 1 (CAT codec) whenever ``chunk_frames``
    new valid frames are ready.

    De-delay math
    -------------
    The delay AR stage emits rows where ``delayed[t, q] = frame[t-q, q]``.
    Inverse: ``frame[t, q] = delayed[t+q, q]``.
    Accumulating N delay steps recovers at most ``max(0, N - n_vq + 1)``
    valid frame-major rows.  With n_vq=32 (MOSS), the first flush therefore
    requires ``chunk_frames + 31`` delay steps.

    Buffer strategy
    ---------------
    All delay-pattern rows are kept in
    ``transfer_manager.code_prompt_token_ids[request_id]`` (never cleared
    per-chunk).  ``transfer_manager._delay_frames_sent[request_id]`` tracks
    how many valid frames have already been delivered to Stage 1.  On each
    call the full buffer is de-delayed, and only the *newly recovered* frames
    are shipped.  The framework's ``cleanup_sender()`` clears
    ``code_prompt_token_ids`` after the final chunk; we clear
    ``_delay_frames_sent`` in-function when ``is_finished=True``.

    left_context_size = 0
    ---------------------
    ``MossTTSDecoderModel.forward()`` does not trim a left-context region
    from its decoded audio, so we send only the new frames with no overlap
    and set ``left_context_size=0``.

    Returns a payload dict when enough new frames are ready (or when
    ``is_finished=True`` with any remaining frames), otherwise ``None``.
    """
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg   = getattr(connector, "config", {}) or {}
    cfg       = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", _DEFAULT_CHUNK_FRAMES))

    request_id = getattr(request, "external_req_id", None)
    if request_id is None:
        return None

    # ── Accumulate this step's raw delay-pattern row ──────────────────────
    if isinstance(pooling_output, dict):
        codes_raw = pooling_output.get("code_predictor_codes")
        if not _has_no_codes(codes_raw):
            row = (
                codes_raw
                if isinstance(codes_raw, torch.Tensor)
                else torch.tensor(codes_raw, dtype=torch.long)
            )
            row = row.to(torch.long).reshape(-1)   # [n_vq]
            if row.numel() > 0:
                transfer_manager.code_prompt_token_ids[request_id].append(row.tolist())
    elif not is_finished:
        return None

    accumulated = transfer_manager.code_prompt_token_ids[request_id]
    n_steps = len(accumulated)

    if n_steps == 0:
        if is_finished:
            return _make_finished_sentinel()
        return None

    # ── De-delay all accumulated rows → valid frame-major codes ──────────
    # Build tensor robustly: filter out any inconsistent entries (e.g. from
    # unexpected calls with wrong shapes) and use torch.stack instead of
    # torch.tensor(list_of_lists) which fails on inhomogeneous content in
    # newer PyTorch.
    row_tensors = []
    for entry in accumulated:
        if isinstance(entry, (list, tuple)) and len(entry) > 0:
            row_tensors.append(torch.tensor(entry, dtype=torch.long))
        elif isinstance(entry, torch.Tensor) and entry.numel() > 0:
            row_tensors.append(entry.to(torch.long).reshape(-1))
    if not row_tensors:
        if is_finished:
            return _make_finished_sentinel()
        return None
    n_vq_ref = row_tensors[0].numel()
    row_tensors = [r for r in row_tensors if r.numel() == row_tensors[0].numel()]
    if not row_tensors:
        if is_finished:
            return _make_finished_sentinel()
        return None
    raw_tensor = torch.stack(row_tensors, dim=0)  # [n_steps, n_vq]
    valid_frames = _apply_de_delay_pattern(
        raw_tensor, audio_pad_code=_DEFAULT_AUDIO_PAD_CODE
    )   # [n_valid, n_vq]
    n_valid = valid_frames.shape[0]

    # ── Track how many valid frames have already been sent ────────────────
    _delay_sent: dict[str, int] = getattr(transfer_manager, "_delay_frames_sent", None)  # type: ignore[assignment]
    if _delay_sent is None:
        _delay_sent = {}
        transfer_manager._delay_frames_sent = _delay_sent
    frames_sent = _delay_sent.get(request_id, 0)
    n_new = n_valid - frames_sent

    if n_new <= 0:
        # Not enough delay steps yet to recover a new valid frame.
        if is_finished:
            _delay_sent.pop(request_id, None)
            return _make_finished_sentinel()
        return None

    if not is_finished and n_new < chunk_size:
        return None   # still accumulating

    # ── Decide how many frames to flush this call ─────────────────────────
    if is_finished:
        flush_count = n_new   # everything remaining
    else:
        flush_count = (n_new // chunk_size) * chunk_size
        if flush_count == 0:
            return None

    flush_frames = valid_frames[frames_sent : frames_sent + flush_count]   # [flush_count, n_vq]
    flat  = flush_frames.reshape(-1).tolist()

    # ── Update sent counter; clean up on final flush ──────────────────────
    _delay_sent[request_id] = frames_sent + flush_count
    if is_finished:
        _delay_sent.pop(request_id, None)

    return {
        "code_predictor_codes": flat,
        "code_flat_numel":      len(flat),
        "codec_chunk_frames":   chunk_size,
        "left_context_size":    0,
        "request_id":           request_id,
        "finished":             torch.tensor(is_finished, dtype=torch.bool),
    }
