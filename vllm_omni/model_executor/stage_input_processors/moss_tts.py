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
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)

# Default chunk / context sizes for streaming (post-MVP).
_DEFAULT_CHUNK_FRAMES = 3
_DEFAULT_CONTEXT_FRAMES = 3
_DEFAULT_AUDIO_PAD_CODE = 1024
_DEFAULT_INITIAL_CHUNK_FRAMES = 0


@dataclass
class _DelayAsyncRequestState:
    """Incremental inverse-delay state for one streaming request.

    The delay model emits delayed[t, q].  We immediately place each code into
    restored[t - q, q], then keep only incomplete restored rows plus completed
    rows that are waiting for the next Stage-1 chunk.
    """

    n_vq: int
    audio_pad_code: int
    step: int = 0
    next_emit_frame: int = 0
    max_frame_seen: int = -1
    frames_flushed: int = 0
    pending_chunk: list[torch.Tensor] = field(default_factory=list)
    _capacity: int = field(init=False)
    _rows: torch.Tensor = field(init=False)
    _filled_counts: torch.Tensor = field(init=False)
    _frame_ids: torch.Tensor = field(init=False)
    _vq_positions: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self._capacity = max(2 * self.n_vq + 2, 4)
        self._rows = torch.full(
            (self._capacity, self.n_vq),
            self.audio_pad_code,
            dtype=torch.long,
        )
        self._filled_counts = torch.zeros(self._capacity, dtype=torch.long)
        self._frame_ids = torch.full((self._capacity,), -1, dtype=torch.long)
        self._vq_positions = torch.arange(self.n_vq, dtype=torch.long)

    @property
    def restored_rows(self) -> dict[int, list[int]]:
        active = torch.nonzero(self._frame_ids >= 0, as_tuple=False).flatten()
        return {
            int(self._frame_ids[slot].item()): self._rows[slot].tolist()
            for slot in active.tolist()
        }

    def _clear_slot(self, slot: int) -> None:
        self._rows[slot].fill_(self.audio_pad_code)
        self._filled_counts[slot] = 0
        self._frame_ids[slot] = -1

    def append_delay_row(self, row: torch.Tensor, audio_pad_code: int) -> bool:
        row = row.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if row.numel() != self.n_vq:
            logger.debug(
                "[MossTTS Delay async] Ignoring row with %d codes; expected n_vq=%d",
                row.numel(),
                self.n_vq,
            )
            self.step += 1
            return False

        self.audio_pad_code = int(audio_pad_code)
        delayed_step = self.step
        self.step += 1

        frame_idxs = delayed_step - self._vq_positions
        valid = (frame_idxs >= 0) & (row != self.audio_pad_code)
        if valid.any():
            frames = frame_idxs[valid]
            channels = self._vq_positions[valid]
            codes = row[valid]
            slots = torch.remainder(frames, self._capacity).to(torch.long)

            new_slots = self._frame_ids[slots] != frames
            if new_slots.any():
                init_slots = slots[new_slots]
                self._rows[init_slots] = self.audio_pad_code
                self._filled_counts[init_slots] = 0
                self._frame_ids[init_slots] = frames[new_slots]

            was_empty = self._rows[slots, channels] == self.audio_pad_code
            self._rows[slots, channels] = codes
            self._filled_counts[slots] += was_empty.to(dtype=torch.long)
            self.max_frame_seen = max(self.max_frame_seen, int(frames.max().item()))

        self.drain_available(final=False)
        return True

    def drain_available(self, *, final: bool) -> None:
        """Move complete restored rows into pending_chunk in frame order."""
        while True:
            slot = self.next_emit_frame % self._capacity
            slot_frame = int(self._frame_ids[slot].item())
            has_slot = slot_frame == self.next_emit_frame
            filled = int(self._filled_counts[slot].item()) if has_slot else 0

            if has_slot and filled == self.n_vq:
                self.pending_chunk.append(self._rows[slot].clone())
                self._clear_slot(slot)
                self.next_emit_frame += 1
                continue

            # Once all q positions for a frame have had their chance to arrive,
            # an incomplete row can never become valid.  This matches the sync
            # path's "drop any restored row containing pad" rule.
            stale = self.step >= self.next_emit_frame + self.n_vq
            if final:
                stale = self.next_emit_frame <= self.max_frame_seen
            if stale:
                if has_slot:
                    self._clear_slot(slot)
                self.next_emit_frame += 1
                continue

            break


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
    t = codes if isinstance(codes, torch.Tensor) else torch.tensor(codes, dtype=torch.long)
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
            "[MossTTS processor] Unexpected codes shape after reshape: %d (expected n_vq=%d)",
            codes.numel(),
            n_vq,
        )
        # Accept if it's a multiple of n_vq (take the first n_vq values)
        if codes.numel() >= n_vq:
            codes = codes[:n_vq]
        else:
            return None

    return codes.tolist()


def _make_finished_sentinel(request_id: str | None = None) -> dict[str, Any]:
    """Minimal payload signalling Stage 1 to end the request."""
    payload: dict[str, Any] = {
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


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


def _async_delay_chunk_size(cfg: dict[str, Any], state: _DelayAsyncRequestState) -> int:
    steady_chunk = int(cfg.get("codec_chunk_frames", _DEFAULT_CHUNK_FRAMES))
    initial_chunk = cfg.get(
        "initial_codec_chunk_frames",
        cfg.get("codec_chunk_frames_at_begin", _DEFAULT_INITIAL_CHUNK_FRAMES),
    )
    initial_chunk = int(initial_chunk or 0)
    if state.frames_flushed == 0 and initial_chunk > 0:
        return initial_chunk
    return steady_chunk


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].item())
    return int(value)


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

    frame_idx = torch.arange(total_steps, device=codes.device).unsqueeze(1)
    ch_idx = torch.arange(n_vq, device=codes.device).unsqueeze(0)
    src_steps = frame_idx + ch_idx
    valid = src_steps < total_steps
    restored[valid] = codes[
        src_steps[valid],
        ch_idx.expand(total_steps, n_vq)[valid],
    ]

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
            f"[MossTTS processor] Invalid stage_id={src_stage_id} (only {len(stage_list)} stages present)."
        )

    stage = stage_list[src_stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"[MossTTS processor] Stage {src_stage_id} has no outputs yet.")

    ar_outputs = stage.engine_outputs  # list[RequestOutput]
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
            pass  # already [T, n_vq]
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
            f"[MossTTS Delay processor] Invalid stage_id={src_stage_id} (only {len(stage_list)} stages present)."
        )

    stage = stage_list[src_stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"[MossTTS Delay processor] Stage {src_stage_id} has no outputs yet.")

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
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", _DEFAULT_CHUNK_FRAMES))

    request_id = getattr(request, "external_req_id", None)

    codes_raw = pooling_output.get("code_predictor_codes")

    # ── Nothing to flush ──────────────────────────────────────────────
    if _is_codes_empty(codes_raw):
        if is_finished:
            return _flush_remaining(transfer_manager, request_id, chunk_size)
        return None

    # ── Convert to per-step flat list ─────────────────────────────────
    codes = codes_raw if isinstance(codes_raw, torch.Tensor) else torch.tensor(codes_raw, dtype=torch.long)
    codes = codes.to(torch.long).reshape(-1)  # [n_vq]
    n_vq = codes.numel()

    if n_vq == 0:
        if is_finished:
            return _flush_remaining(transfer_manager, request_id, chunk_size)
        return None

    if request_id is None:
        return None

    # Accumulate this frame's codes
    transfer_manager.code_prompt_token_ids[request_id].append(codes.tolist())
    accumulated = transfer_manager.code_prompt_token_ids[request_id]
    n_frames = len(accumulated)

    # Flush when chunk is full or generation is done
    if n_frames % chunk_size != 0 and not is_finished:
        return None  # still waiting

    # Build flush payload
    flat = [code for frame in accumulated for code in frame]
    numel = len(flat)
    return {
        "code_predictor_codes": flat,
        "code_flat_numel": numel,
        "codec_chunk_frames": chunk_size,
        "left_context_size": 0,  # MOSS has no delay pattern → no left context
        "request_id": request_id,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
    }


def _flush_remaining(
    transfer_manager: Any,
    request_id: str | None,
    chunk_size: int,
) -> dict[str, Any]:
    """Flush any leftover codes when the request finishes mid-chunk."""
    if request_id is None:
        return _make_finished_sentinel(request_id)

    accumulated = transfer_manager.code_prompt_token_ids.get(request_id, [])
    if not accumulated:
        return _make_finished_sentinel(request_id)

    flat = [code for frame in accumulated for code in frame]
    numel = len(flat)
    return {
        "code_predictor_codes": flat,
        "code_flat_numel": numel,
        "codec_chunk_frames": chunk_size,
        "left_context_size": 0,
        "request_id": request_id,
        "finished": torch.tensor(True, dtype=torch.bool),
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
    ``custom_process_next_stage_input_func``.  Each raw delay-pattern row is
    placed directly into an incremental restored-frame ring buffer, and
    completed frame-major rows are flushed to Stage 1 (CAT codec) whenever
    enough new valid frames are ready.  ``initial_codec_chunk_frames`` may be
    used for a small first chunk, while ``codec_chunk_frames`` controls the
    larger steady-state chunks.

    De-delay math
    -------------
    The delay AR stage emits rows where ``delayed[t, q] = frame[t-q, q]``.
    Inverse: ``frame[t, q] = delayed[t+q, q]``.
    Accumulating N delay steps recovers at most ``max(0, N - n_vq + 1)``
    valid frame-major rows.  With n_vq=32 (MOSS), the first flush therefore
    requires ``initial_codec_chunk_frames + 31`` delay steps when an initial
    chunk size is configured, otherwise ``codec_chunk_frames + 31``.

    Buffer strategy
    ---------------
    ``transfer_manager._moss_tts_delay_async_states[request_id]`` stores only:
    incomplete restored rows, completed rows waiting for chunk flush, and the
    next frame index to emit.  Once a frame is emitted it is removed, so memory
    stays bounded by roughly ``n_vq + codec_chunk_frames`` rows plus any
    completed rows waiting for the next flush.

    left_context_size = 0
    ---------------------
    ``MossTTSDecoderModel.forward()`` does not trim a left-context region
    from its decoded audio, so we send only the new frames with no overlap
    and set ``left_context_size=0``.

    Returns a payload dict when enough new frames are ready (or when
    ``is_finished=True`` with any remaining frames), otherwise ``None``.
    """
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}

    request_id = getattr(request, "external_req_id", None)
    if request_id is None:
        return None

    states: dict[str, _DelayAsyncRequestState] = getattr(
        transfer_manager,
        "_moss_tts_delay_async_states",
        None,
    )
    if states is None:
        states = {}
        transfer_manager._moss_tts_delay_async_states = states

    state = states.get(request_id)

    # ── Incrementally place this raw delay row into restored-frame slots ──
    if isinstance(pooling_output, dict):
        audio_pad_code = _as_int(
            pooling_output.get("audio_pad_code"),
            _DEFAULT_AUDIO_PAD_CODE,
        )
        _pad_codes: dict[str, int] = getattr(  # type: ignore[assignment]
            transfer_manager,
            "_moss_tts_delay_audio_pad_code",
            None,
        )
        if _pad_codes is None:
            _pad_codes = {}
            transfer_manager._moss_tts_delay_audio_pad_code = _pad_codes
        _pad_codes[request_id] = audio_pad_code

        codes_raw = pooling_output.get("code_predictor_codes")
        if not _has_no_codes(codes_raw):
            row = codes_raw if isinstance(codes_raw, torch.Tensor) else torch.tensor(codes_raw, dtype=torch.long)
            row = row.to(torch.long).reshape(-1)  # [n_vq]
            if row.numel() > 0:
                if state is None:
                    state = _DelayAsyncRequestState(
                        n_vq=int(row.numel()),
                        audio_pad_code=audio_pad_code,
                    )
                    states[request_id] = state
                state.append_delay_row(row, audio_pad_code)
    elif not is_finished:
        return None

    if state is None:
        if is_finished:
            return _make_finished_sentinel(request_id)
        return None

    if is_finished:
        state.drain_available(final=True)

    n_pending = len(state.pending_chunk)
    if n_pending <= 0:
        if is_finished:
            states.pop(request_id, None)
            if hasattr(transfer_manager, "_moss_tts_delay_audio_pad_code"):
                transfer_manager._moss_tts_delay_audio_pad_code.pop(request_id, None)
            return _make_finished_sentinel(request_id)
        return None

    chunk_size = _async_delay_chunk_size(cfg, state)
    if not is_finished and chunk_size <= 0:
        return None

    if not is_finished and n_pending < chunk_size:
        return None  # still accumulating

    # ── Decide how many frames to flush this call ─────────────────────────
    if is_finished:
        flush_count = n_pending  # everything remaining
    else:
        flush_count = (n_pending // chunk_size) * chunk_size
        if flush_count == 0:
            return None

    flush_frames = state.pending_chunk[:flush_count]
    del state.pending_chunk[:flush_count]
    flat = torch.stack(flush_frames, dim=0).reshape(-1).tolist()
    state.frames_flushed += flush_count
    payload_chunk_size = chunk_size if chunk_size > 0 else flush_count

    # ── Update sent counter; clean up on final flush ──────────────────────
    if is_finished:
        states.pop(request_id, None)
        if hasattr(transfer_manager, "_moss_tts_delay_audio_pad_code"):
            transfer_manager._moss_tts_delay_audio_pad_code.pop(request_id, None)

    return {
        "code_predictor_codes": flat,
        "code_flat_numel": len(flat),
        "codec_chunk_frames": payload_chunk_size,
        "left_context_size": 0,
        "request_id": request_id,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
    }
