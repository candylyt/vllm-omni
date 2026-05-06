# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
#
# Stage 1: CAT codec decoder for MOSS-TTS-Local.
#
# The CAT codec (Causal Audio Tokenizer) is MOSS's 1.6B Transformer-based
# audio tokenizer from https://github.com/OpenMOSS/MOSS-Audio-Tokenizer.
#
# It converts a sequence of 32-layer RVQ codes back to a 24 kHz waveform:
#
#   flat_codes  [N]    (N = T * n_vq, flattened by stage processor)
#        ↓  reshape
#   codes       [n_vq, T]  = [32, T]
#        ↓  cat_codec.decode()
#   waveform    [samples]  (float32, 24 kHz)
#
# The CAT codec is a SEPARATE model from the main MOSS-TTS checkpoint.
# It must be downloaded independently:
#
#   Main model : OpenMOSS-Team/MOSS-TTS-Local-Transformer  (~6.1 GB on disk)
#   CAT codec  : OpenMOSS-Team/MOSS-Audio-Tokenizer        (separate download)
#
# Verified from processing_moss_tts.py — default codec path is:
#   "OpenMOSS-Team/MOSS-Audio-Tokenizer"
#
# Environment variable:
#   MOSS_AUDIO_TOKENIZER_PATH  — local path or HF repo ID for the CAT codec.
#                                Defaults to "OpenMOSS-Team/MOSS-Audio-Tokenizer"
#                                if unset (will auto-download from HuggingFace).

import logging
import os
import time
from collections.abc import Iterable
from contextlib import ExitStack
from typing import Any, Optional

import torch
from torch import nn
from torch.profiler import record_function
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.moss_tts._stage1_timing import get_timer
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)
_TIMER = get_timer()


# ═══════════════════════════════════════════════════════════════════════════════
#  CAT Codec worker (singleton per process / device)
# ═══════════════════════════════════════════════════════════════════════════════

class CATCodecWorker:
    """
    Loads the MOSS CAT codec and exposes encode / decode helpers.

    The CAT codec API (from MOSS-Audio-Tokenizer) is:
        tokenizer.encode(waveform)         → codes [n_vq, T]
        tokenizer.decode(codes [n_vq, T]) → waveform [samples]

    This wrapper caches one instance per (device, checkpoint_path) pair
    so multiple requests share the same loaded model.
    """

    def __init__(self, device_str: str, codec_path: str):
        self.device = torch.device(device_str)
        # Resolve to absolute path so transformers doesn't reject relative paths
        # (transformers ≥ 4.40 validates that local paths are absolute)
        if os.path.exists(codec_path):
            codec_path = os.path.realpath(codec_path)
        logger.info("[MossTTS Decoder] Loading CAT codec from %s on %s", codec_path, device_str)

        # Patch stale cached HF module: the cached configuration_moss_audio_tokenizer.py
        # does `from transformers.configuration_utils import PreTrainedConfig`.
        # Newer transformers renamed the class to PretrainedConfig (lowercase 't').
        # Inject the alias so the old import path resolves.
        import transformers.configuration_utils as _cfg_utils
        if not hasattr(_cfg_utils, "PreTrainedConfig"):
            from transformers import PretrainedConfig as _ptc
            _cfg_utils.PreTrainedConfig = _ptc

        from transformers import AutoModel
        # Do NOT pass torch_dtype here — the CAT codec has mixed-precision layers
        # that break when forced to bfloat16.  It outputs float32 natively.
        self.codec = AutoModel.from_pretrained(
            codec_path,
            trust_remote_code=True,
        )
        self.codec = self.codec.to(self.device).eval().float()

        self.sample_rate: int = getattr(self.codec.config, "sampling_rate", 24_000)
        # Config exposes num_quantizers as a @property backed by quantizer_kwargs
        self.n_vq: int        = getattr(self.codec.config, "num_quantizers", 32)

        logger.info(
            "[MossTTS Decoder] CAT codec loaded: sample_rate=%d, n_vq=%d",
            self.sample_rate,
            self.n_vq,
        )

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes : [n_vq, T]  long  (on any device)
        Returns: [samples]  float32  CPU

        Verified API (from MOSS-Audio-Tokenizer):
            out = codec.decode(codes)          # codes: [n_vq, T] long
            out.audio          shape [1, 1, T_audio]  float32
            out.audio_lengths  shape [1]               int64
            sampling_rate = 24000, downsample_rate = 1920
        """
        codes = codes.to(self.device)
        out = self.codec.decode(codes, chunk_duration=8)  # MossAudioTokenizerDecoderOutput
        wav = out.audio[0, 0]                   # [T_audio]  float32
        return wav.float().cpu()


# Module-level cache: (device_type, codec_path) → CATCodecWorker
_CODEC_WORKER_CACHE: dict[tuple[str, str], CATCodecWorker] = {}


def _get_codec_worker(device: torch.device, codec_path: str) -> CATCodecWorker:
    key = (device.type, os.path.realpath(codec_path))
    if key not in _CODEC_WORKER_CACHE:
        _CODEC_WORKER_CACHE[key] = CATCodecWorker(device.type, codec_path)
    return _CODEC_WORKER_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════════════
#  Flat-code parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_flat_codes(
    flat_codes: torch.Tensor,
    n_vq: int,
) -> Optional[torch.Tensor]:
    """
    Reshape a 1-D flat code tensor into [n_vq, T].

    The stage processor (stage_input_processors/moss_tts.py) produces codes
    flattened in row-major order: [code_t0_vq0, code_t0_vq1, ..., code_t0_vq31,
                                   code_t1_vq0, ..., code_tN_vq31]
    i.e. shape [T * n_vq] → reshape to [T, n_vq] → transpose to [n_vq, T].

    Returns None if the flat tensor is empty or not a multiple of n_vq.
    """
    flat_codes = flat_codes.reshape(-1).to(torch.long)
    total = flat_codes.numel()
    if total == 0 or total % n_vq != 0:
        return None
    T = total // n_vq
    return flat_codes.reshape(T, n_vq).transpose(0, 1).contiguous()  # [n_vq, T]


def _split_per_request(
    ids: torch.Tensor,
    runtime_info: Optional[list[dict[str, Any]]],
    seq_token_counts: Optional[list[int]],
) -> list[torch.Tensor]:
    """
    Split a flat code tensor into per-request slices.

    Prefers `code_flat_numel` from the runtime_additional_information dict
    (set by the async-chunk processor) over the raw seq_token_counts.
    Mirrors MiMo's _split_flat_codes_for_requests().
    """
    n = ids.numel()
    if n == 0:
        return [ids]

    if runtime_info and all(
        isinstance(info.get("code_flat_numel"), int) and info["code_flat_numel"] > 0
        for info in runtime_info
    ):
        sizes = [int(info["code_flat_numel"]) for info in runtime_info]
        if sum(sizes) == n:
            parts, offset = [], 0
            for sz in sizes:
                parts.append(ids[offset: offset + sz])
                offset += sz
            return parts

    if seq_token_counts and len(seq_token_counts) > 1:
        boundaries = [0]
        for c in seq_token_counts:
            boundaries.append(boundaries[-1] + c)
        return [ids[boundaries[i]: min(boundaries[i + 1], n)]
                for i in range(len(seq_token_counts))]

    return [ids]


# ═══════════════════════════════════════════════════════════════════════════════
#  Decoder Stage Model
# ═══════════════════════════════════════════════════════════════════════════════

class MossTTSDecoderModel(nn.Module, SupportsPP):
    """
    Stage 1: CAT codec decoder.

    Receives flat RVQ code sequences from the stage processor and decodes them
    to 24 kHz waveform tensors using the MOSS CAT codec.

    The decoder has no trainable parameters that are updated during inference;
    load_weights() is a no-op except for logging.
    """

    have_multimodal_outputs    = True
    enable_update_additional_information = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        self.config = cfg

        self.n_vq: int        = cfg.n_vq           # 32
        self.sample_rate: int = getattr(cfg, "sampling_rate", 24_000)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Resolve CAT codec checkpoint path
        # Resolution order (mirrors processing_moss_tts.py logic):
        #   1. model_config.audio_tokenizer_path  (set in YAML engine_args)
        #   2. MOSS_AUDIO_TOKENIZER_PATH env var   (local path or HF repo ID)
        #   3. Hard-coded HF default               (auto-downloads on first use)
        codec_path = (
            getattr(vllm_config.model_config, "audio_tokenizer_path", None)
            or os.environ.get("MOSS_AUDIO_TOKENIZER_PATH")
            or "OpenMOSS-Team/MOSS-Audio-Tokenizer"  # verified default from processing_moss_tts.py
        )

        self._codec: CATCodecWorker = _get_codec_worker(self.device, codec_path)

        # Dummy logits processor / sampler required by vllm's model protocol
        self.logits_processor = LogitsProcessor(cfg.language_config.vocab_size)
        self.sampler = Sampler()

        # Per-request streaming state: request_id → ExitStack holding codec
        # KV-cache context.  Entries are created on the first chunk for a
        # request and closed when ``is_finished=True`` is observed.
        self._streaming_states: dict[str, ExitStack] = {}
        # Batched streaming state for multi-request async decode.  The CAT
        # codec exposes a single global streaming mode per codec instance, so
        # concurrent requests must share one batch-sized streaming context.
        self._batched_streaming_stack: ExitStack | None = None
        self._batched_streaming_request_ids: list[str] | None = None

        # Synthetic request-id tracking (workaround: the SharedMemoryConnector
        # strips the processor's ``request_id`` / ``finished`` keys, so Stage 1
        # only sees ``generated_len`` + ``left_context_size``).  With
        # max_num_seqs=1 there is at most one active stream at any time, so we
        # mint a stable key and detect a new request by watching
        # ``generated_len`` reset to a smaller value.
        self._active_key: Optional[str] = None
        self._last_gen_len: Optional[int] = None
        self._run_counter: int = 0

        # First-chunk latency instrumentation.
        # When MOSS_FIRST_CHUNK_DIR is set, we write a wall-clock timestamp
        # file the very first time a non-empty waveform is produced for each
        # request_id.  The benchmark reads these files to compute the
        # time-to-first-audio metric that highlights the async_chunk win.
        self._first_chunk_dir: Optional[str] = os.environ.get("MOSS_FIRST_CHUNK_DIR")
        self._first_chunk_seen: set[str] = set()
        if self._first_chunk_dir:
            try:
                os.makedirs(self._first_chunk_dir, exist_ok=True)
            except OSError:
                self._first_chunk_dir = None

    def _record_first_chunk(self, request_id: Optional[str]) -> None:
        """Write a wall-clock timestamp file the first time a non-empty
        waveform is produced.  For single-request benchmarks we also write
        a ``first.first_chunk.ts`` fallback so the metric still works even
        when ``request_id`` isn't propagated through the connector payload."""
        now = time.time()
        logger.info(
            "[MossTTS Decoder][TIMING] non-empty wav produced at wall=%.3f "
            "request_id=%s",
            now, request_id,
        )
        if not self._first_chunk_dir:
            return
        # Candidate keys: the real request_id (if any) plus a generic
        # "first" fallback covering single-request benchmarks.
        keys = [k for k in (request_id, "first") if k]
        for key in keys:
            if key in self._first_chunk_seen:
                continue
            self._first_chunk_seen.add(key)
            path = os.path.join(self._first_chunk_dir, f"{key}.first_chunk.ts")
            try:
                with open(path, "w") as f:
                    f.write(f"{now:.6f}\n")
            except OSError as exc:
                logger.debug("[MossTTS Decoder] Could not write %s: %s", path, exc)

    # ══════════════════════════════════════════════════════════════════
    #  Core decode logic
    # ══════════════════════════════════════════════════════════════════

    def _enter_streaming(self, request_id: str) -> None:
        """Enter codec streaming mode for a new request, storing KV-cache state."""
        if request_id in self._streaming_states:
            return
        stack = ExitStack()
        codec = self._codec.codec
        for decoder_module in codec.decoder:
            if hasattr(decoder_module, "streaming") and callable(decoder_module.streaming):
                stack.enter_context(decoder_module.streaming(batch_size=1))
        self._streaming_states[request_id] = stack

    def _exit_streaming(self, request_id: str) -> None:
        """Exit and discard the streaming state for a finished request."""
        stack = self._streaming_states.pop(request_id, None)
        if stack is not None:
            stack.close()

    def _reset_streaming_topology(self) -> None:
        """Close any active codec streaming sessions before switching modes."""
        for request_id in list(self._streaming_states.keys()):
            self._exit_streaming(request_id)
        self._exit_batched_streaming()

    def _enter_batched_streaming(self, request_ids: list[str]) -> None:
        """Enter one shared codec streaming context for a multi-request batch."""
        if self._batched_streaming_stack is not None:
            if self._batched_streaming_request_ids != request_ids:
                logger.warning(
                    "[MossTTS Decoder] Batched streaming request set changed "
                    "from %s to %s; resetting codec streaming state.",
                    self._batched_streaming_request_ids,
                    request_ids,
                )
                self._exit_batched_streaming()
            else:
                return

        if self._streaming_states:
            logger.warning(
                "[MossTTS Decoder] Switching from per-request streaming to "
                "batched streaming; resetting %d active single-request states.",
                len(self._streaming_states),
            )
            self._reset_streaming_topology()

        stack = ExitStack()
        codec = self._codec.codec
        batch_size = len(request_ids)
        for decoder_module in codec.decoder:
            if hasattr(decoder_module, "streaming") and callable(decoder_module.streaming):
                stack.enter_context(decoder_module.streaming(batch_size=batch_size))
        self._batched_streaming_stack = stack
        self._batched_streaming_request_ids = list(request_ids)

    def _exit_batched_streaming(self) -> None:
        """Exit the shared codec streaming context for a multi-request batch."""
        if self._batched_streaming_stack is not None:
            self._batched_streaming_stack.close()
        self._batched_streaming_stack = None
        self._batched_streaming_request_ids = None

    def _decode_one_request(
        self,
        flat_codes: torch.Tensor,
        request_id: Optional[str] = None,
        is_finished: bool = False,
    ) -> torch.Tensor:
        """
        Decode one request's flat code tensor to a waveform.

        When ``request_id`` is provided, uses the codec's streaming KV-cache
        so that causal state is maintained across successive chunk calls
        for the same request (async_chunk mode).  When ``request_id`` is
        None, falls back to the stateless single-call decode (sync/batch).

        flat_codes  : [T * n_vq]  long
        request_id  : str or None — if set, enables per-request streaming state
        is_finished : bool — if True, streaming state is released after this call
        Returns     : [samples]   float32  (empty tensor if codes are invalid)
        """
        empty = torch.zeros(0, dtype=torch.float32)

        with record_function("stage1/decode_one_request"), _TIMER.gpu("stage1/decode_one_request"):
            if flat_codes is None or flat_codes.numel() == 0:
                if request_id and is_finished:
                    self._exit_streaming(request_id)
                return empty

            codes = _parse_flat_codes(flat_codes, self.n_vq)  # [n_vq, T] or None
            if codes is None:
                if request_id and is_finished:
                    self._exit_streaming(request_id)
                return empty

            # Skip all-zero code tensors (dummy / padding frames)
            if not codes.any():
                if request_id and is_finished:
                    self._exit_streaming(request_id)
                return empty

            try:
                if request_id is not None:
                    # Streaming path: maintain causal KV-cache across chunks.
                    if self._batched_streaming_stack is not None:
                        logger.warning(
                            "[MossTTS Decoder] Switching from batched "
                            "streaming to per-request streaming for %s; "
                            "resetting codec streaming state.",
                            request_id,
                        )
                        self._reset_streaming_topology()
                    self._enter_streaming(request_id)
                    codec = self._codec.codec
                    codes_3d = codes.unsqueeze(1).to(self.device)   # [n_vq, 1, T]
                    lengths = torch.tensor(
                        [codes_3d.shape[-1]], device=self.device, dtype=torch.long
                    )
                    result = codec._decode_frame(codes_3d, lengths)
                    wav = result.audio[0, 0, : result.audio_lengths[0]].float().cpu()
                    if is_finished:
                        self._exit_streaming(request_id)
                else:
                    # Fallback: stateless single-call decode (sync / non-streaming mode).
                    wav = self._codec.decode(codes)

                if wav is not None and wav.numel() > 0:
                    self._record_first_chunk(request_id)
                return wav
            except Exception as exc:
                logger.error("[MossTTS Decoder] Codec decode failed: %s", exc, exc_info=True)
                if request_id and is_finished:
                    self._exit_streaming(request_id)
                return empty

    @torch.inference_mode()
    def _batch_decode(
        self,
        request_codes_list: list[torch.Tensor],
        request_ids: Optional[list[Optional[str]]] = None,
        finished_flags: Optional[list[bool]] = None,
    ) -> list[torch.Tensor]:
        """Decode multiple requests, using per-request streaming state when available."""
        empty = torch.zeros(0, dtype=torch.float32)
        logger.info(
            "[MossTTS Decoder][DEBUG] _batch_decode num_req=%d request_ids=%s "
            "finished=%s batched_streaming_active=%s single_streams=%d",
            len(request_codes_list),
            request_ids,
            finished_flags,
            self._batched_streaming_stack is not None,
            len(self._streaming_states),
        )

        if (
            request_ids
            and len(request_codes_list) > 1
            and all(isinstance(rid, str) and rid for rid in request_ids)
        ):
            logger.info(
                "[MossTTS Decoder][DEBUG] taking batched streaming path for "
                "request_ids=%s",
                request_ids,
            )
            return self._batch_decode_streaming(
                request_codes_list,
                [rid for rid in request_ids if isinstance(rid, str)],
                finished_flags or [False] * len(request_codes_list),
            )

        if self._batched_streaming_stack is not None:
            logger.warning(
                "[MossTTS Decoder] Falling back to non-batched decode for %d "
                "request(s); resetting active batched streaming state.",
                len(request_codes_list),
            )
            self._reset_streaming_topology()

        results: list[torch.Tensor] = []
        for i, req_codes in enumerate(request_codes_list):
            req_id = request_ids[i] if request_ids else None
            finished = finished_flags[i] if finished_flags else False
            logger.info(
                "[MossTTS Decoder][DEBUG] taking per-request path idx=%d "
                "request_id=%s finished=%s numel=%d",
                i,
                req_id,
                finished,
                int(req_codes.numel()) if req_codes is not None else -1,
            )
            wav = self._decode_one_request(req_codes, request_id=req_id, is_finished=finished)
            results.append(wav if wav.numel() > 0 else empty)
        return results

    def _batch_decode_streaming(
        self,
        request_codes_list: list[torch.Tensor],
        request_ids: list[str],
        finished_flags: list[bool],
    ) -> list[torch.Tensor]:
        """Decode multiple active requests through one batched codec stream."""
        empty = torch.zeros(0, dtype=torch.float32)

        if len(request_codes_list) != len(request_ids):
            raise ValueError(
                "request_codes_list and request_ids must have the same length "
                f"(got {len(request_codes_list)} vs {len(request_ids)})."
            )

        parsed_codes: list[torch.Tensor] = []
        lengths: list[int] = []
        for flat_codes in request_codes_list:
            if flat_codes is None or flat_codes.numel() == 0:
                parsed = torch.zeros(self.n_vq, 0, dtype=torch.long)
            else:
                parsed = _parse_flat_codes(flat_codes, self.n_vq)
                if parsed is None:
                    parsed = torch.zeros(self.n_vq, 0, dtype=torch.long)
            parsed_codes.append(parsed)
            lengths.append(int(parsed.shape[-1]))

        logger.info(
            "[MossTTS Decoder][DEBUG] batched stream request_ids=%s lengths=%s "
            "finished=%s",
            request_ids,
            lengths,
            finished_flags,
        )

        if max(lengths, default=0) == 0:
            if all(finished_flags):
                self._exit_batched_streaming()
            return [empty for _ in request_codes_list]

        self._enter_batched_streaming(request_ids)

        max_len = max(lengths)
        batch_size = len(request_ids)
        padded = torch.zeros(
            self.n_vq,
            batch_size,
            max_len,
            dtype=torch.long,
            device=self.device,
        )
        for idx, (codes, code_len) in enumerate(zip(parsed_codes, lengths)):
            if code_len > 0:
                padded[:, idx, :code_len] = codes.to(self.device)

        lengths_t = torch.tensor(lengths, device=self.device, dtype=torch.long)

        try:
            result = self._codec.codec._decode_frame(padded, lengths_t)
            wav_batch = result.audio[:, 0].float().cpu()
            audio_lengths = result.audio_lengths
            results: list[torch.Tensor] = []
            for idx, req_id in enumerate(request_ids):
                wav_len = int(audio_lengths[idx].item()) if audio_lengths.numel() > idx else int(wav_batch.shape[-1])
                wav = wav_batch[idx, :wav_len]
                if wav.numel() > 0:
                    self._record_first_chunk(req_id)
                results.append(wav if wav.numel() > 0 else empty)
        except Exception:
            if all(finished_flags):
                self._exit_batched_streaming()
            raise

        if all(finished_flags):
            self._exit_batched_streaming()

        return results

    # ══════════════════════════════════════════════════════════════════
    #  Forward
    # ══════════════════════════════════════════════════════════════════

    def forward(
        self,
        input_ids: Optional[torch.Tensor]              = None,
        codes: Optional[torch.Tensor]                  = None,
        runtime_additional_information: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> OmniOutput:
        with record_function("stage1/forward"), _TIMER.cpu("stage1/forward_total"):
            # Resolve runtime_additional_information from alternate kwargs keys
            if runtime_additional_information is None:
                runtime_additional_information = (
                    kwargs.get("model_intermediate_buffer")
                    or kwargs.get("runtime_additional_information")
                )

            code_tensor = codes if codes is not None else input_ids
            empty       = torch.zeros(0, dtype=torch.float32)

            with record_function("stage1/extract_codes"), _TIMER.cpu("stage1/extract_codes"):
                # Async-chunk path: when input_ids is empty/None but codes were
                # shipped through runtime_additional_information (SharedMemory
                # connector payload), extract them directly from the per-request
                # ``code_predictor_codes`` field.
                if (code_tensor is None or code_tensor.numel() == 0) and runtime_additional_information:
                    code_parts: list[torch.Tensor] = []
                    for info in runtime_additional_information:
                        if not isinstance(info, dict):
                            continue
                        cp = info.get("code_predictor_codes")
                        if cp is None:
                            continue
                        if not isinstance(cp, torch.Tensor):
                            cp = torch.tensor(cp, dtype=torch.long)
                        code_parts.append(cp.reshape(-1).to(torch.long))
                    if code_parts:
                        code_tensor = torch.cat(code_parts, dim=0)

            if code_tensor is None or code_tensor.numel() == 0:
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={"model_outputs": [empty]},
                )

            # Skip decode during CUDA graph capture
            if torch.cuda.is_current_stream_capturing():
                n = len(runtime_additional_information) if runtime_additional_information else 1
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={"model_outputs": [empty] * n},
                )

            with record_function("stage1/split_requests"), _TIMER.cpu("stage1/split_requests"):
                ids = code_tensor.reshape(-1).to(torch.long)

                # Split flat tensor into per-request slices
                request_codes_list = _split_per_request(
                    ids,
                    runtime_additional_information,
                    kwargs.get("seq_token_counts"),
                )

            # Extract per-request streaming metadata when available.
            request_ids: Optional[list[Optional[str]]] = None
            finished_flags: Optional[list[bool]] = None
            with record_function("stage1/metadata_prep"), _TIMER.cpu("stage1/metadata_prep"):
                if runtime_additional_information:
                    request_ids = []
                    finished_flags = []
                    for info in runtime_additional_information:
                        if not isinstance(info, dict):
                            request_ids.append(None)
                            finished_flags.append(False)
                            continue

                        rid = info.get("request_id")
                        if rid is None:
                            gen_len = info.get("generated_len")
                            if (
                                self._active_key is None
                                or (
                                    isinstance(gen_len, int)
                                    and isinstance(self._last_gen_len, int)
                                    and gen_len < self._last_gen_len
                                )
                            ):
                                if self._active_key is not None:
                                    self._exit_streaming(self._active_key)
                                self._run_counter += 1
                                self._active_key = f"run_{self._run_counter}"
                            rid = self._active_key
                            if isinstance(gen_len, int):
                                self._last_gen_len = gen_len

                        request_ids.append(rid)

                        fin = info.get("finished")
                        if isinstance(fin, torch.Tensor):
                            fin = bool(fin.item())
                        finished_flags.append(bool(fin) if fin is not None else False)
                    if not getattr(self, "_dumped_info_keys", False):
                        self._dumped_info_keys = True
                        for idx, info in enumerate(runtime_additional_information):
                            if isinstance(info, dict):
                                logger.info(
                                    "[MossTTS Decoder][DEBUG] info[%d] keys=%s types=%s",
                                    idx, list(info.keys()),
                                    {k: type(v).__name__ for k, v in info.items()},
                                )
                            else:
                                logger.info(
                                    "[MossTTS Decoder][DEBUG] info[%d] type=%s (not a dict)",
                                    idx, type(info).__name__,
                                )

            with record_function("stage1/decode"), _TIMER.gpu("stage1/decode_total"):
                audios = self._batch_decode(request_codes_list, request_ids, finished_flags)

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios},
            )

    # ══════════════════════════════════════════════════════════════════
    #  vllm model protocol stubs
    # ══════════════════════════════════════════════════════════════════

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: bool = False,
    ) -> torch.Tensor:
        # Stage 1 has no meaningful embeddings; return a zero tensor to satisfy
        # vllm's pipeline interface.
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return torch.zeros(
            input_ids.shape[0], hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if hidden_states is None or hidden_states.numel() == 0:
            return None
        vocab_size = self.config.language_config.vocab_size
        return torch.zeros(
            hidden_states.shape[0], vocab_size,
            dtype=hidden_states.dtype,
            device=self.device,
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        if logits is None or logits.numel() == 0:
            return None
        return self.sampler(logits, sampling_metadata)

    def make_omni_output(self, model_output: Any, **kwargs) -> OmniOutput:
        if isinstance(model_output, OmniOutput):
            return model_output
        empty = torch.zeros(0, dtype=torch.float32)
        if model_output is None:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty]},
            )
        if isinstance(model_output, torch.Tensor):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [model_output.float().reshape(-1)]
                },
            )
        raise TypeError(f"Unexpected model output type: {type(model_output)}")

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        **kwargs,
    ) -> set[str]:
        # The CAT codec is loaded separately via from_pretrained().
        # No weights to load through vllm's standard mechanism.
        logger.info(
            "[MossTTS Decoder] load_weights() called — CAT codec already "
            "loaded in __init__; nothing to do here."
        )
        return set()

    def _clear_warmup_state(self) -> None:
        _TIMER.reset()
