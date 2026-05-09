from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_FRAMES = 3

def _is_codes_empty(codes: Any) -> bool:
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

# payload for final chunk
def _make_finished_sentinel(
    request_id: str,
    chunk_size: int,
) -> dict[str, Any]:
    return {
        "code_predictor_codes": [],
        "code_flat_numel": 0,
        "codec_chunk_frames": chunk_size,
        "left_context_size": 0,
        "request_id": request_id,
        "finished": torch.tensor(True, dtype=torch.bool),
    }

# batch mode
def llm2decoder(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
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

    ar_outputs = stage.engine_outputs 
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


        codes = codes_raw.to(torch.long)

        # Normalize to [T, n_vq]
        if codes.ndim == 4:
            codes = codes.squeeze(1).squeeze(-1)
        elif codes.ndim == 3:
            codes = codes.squeeze(-1)
        elif codes.ndim == 2:
            pass
        else:
            logger.warning(
                "[MossTTS processor] Unexpected codes ndim=%d for request %s — skipping.",
                codes.ndim,
                req_idx,
            )
            continue

        if not codes.any():
            logger.warning(
                "[MossTTS processor] Request %s: all-zero codes — skipping.",
                req_idx,
            )
            continue

        n_vq = codes.shape[1]

        # flatten in row-major order (frame-first) since the decoder expects codes grouped by frame
        flat = codes.reshape(-1).tolist()

        decoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return decoder_inputs

# async-chunk mode
def llm2decoder_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", _DEFAULT_CHUNK_FRAMES))

    request_id = getattr(request, "external_req_id", None)
    if request_id is None:
        return None

    # accumulate current step's frame-major row into transfer manager's buffer
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

    # track how many frames have been sent out for this request so far
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

    # decide how many frames to flush out this call
    if is_finished:
        flush_count = n_new
    else:
        flush_count = (n_new // chunk_size) * chunk_size
        if flush_count == 0:
            return None

    flush_frames = accumulated[frames_sent : frames_sent + flush_count]
    flat = [code for frame in flush_frames for code in frame]

    # update sent count, and clean up if finished
    sent_map[request_id] = frames_sent + flush_count
    if is_finished:
        sent_map.pop(request_id, None)

    return {
        "code_predictor_codes": flat,
        "code_flat_numel": len(flat),
        "codec_chunk_frames": chunk_size,
        "left_context_size": 0,
        "request_id": request_id,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
    }
