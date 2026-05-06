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

import copy
import atexit
import logging
import os
import time
from collections.abc import Iterable
from contextlib import ExitStack
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models import SupportsPP
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)

_STREAMING_STATE_SCALAR_TYPES = (
    bool,
    int,
    float,
    str,
    bytes,
    torch.device,
    torch.dtype,
    torch.layout,
)


class _BatchedScalarStreamingState:
    """Preserve divergent per-request scalar fields inside batched codec state."""

    __slots__ = ("values",)

    def __init__(self, values: Iterable[Any]):
        self.values = list(values)

    def _single_value(self) -> Any:
        first = self.values[0]
        if all(value == first for value in self.values):
            return first
        raise TypeError("Cannot use divergent batched scalar streaming-state value as a single scalar")

    def __iadd__(self, other: Any):
        self.values = [value + other for value in self.values]
        return self

    def __add__(self, other: Any):
        return self._single_value() + other

    def __radd__(self, other: Any):
        return other + self._single_value()

    def __int__(self) -> int:
        return int(self._single_value())

    def __float__(self) -> float:
        return float(self._single_value())

    def __index__(self) -> int:
        return int(self._single_value())

    def __bool__(self) -> bool:
        return bool(self._single_value())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.values!r})"


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
            from transformers import PretrainedConfig as _PretrainedConfig

            _cfg_utils.PreTrainedConfig = _PretrainedConfig

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
        self.n_vq: int = getattr(self.codec.config, "num_quantizers", 32)

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
        wav = out.audio[0, 0]  # [T_audio]  float32
        return wav.float()


# Module-level cache: (device_type, codec_path) → CATCodecWorker
_CODEC_WORKER_CACHE: dict[tuple[str, str], CATCodecWorker] = {}


def _normalize_codec_path_for_cache(codec_path: str) -> str:
    if os.path.exists(codec_path):
        return os.path.realpath(codec_path)
    return codec_path


def _resolve_codec_path(vllm_config: VllmConfig) -> str:
    codec_path = (
        getattr(vllm_config.model_config, "audio_tokenizer_path", None)
        or os.environ.get("MOSS_AUDIO_TOKENIZER_PATH")
        or "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    )
    codec_path = os.path.expanduser(str(codec_path))
    if os.path.isabs(codec_path) and not os.path.exists(codec_path):
        raise FileNotFoundError(
            "[MossTTS Decoder] audio_tokenizer_path points to a missing local path: "
            f"{codec_path}. Set model_config.audio_tokenizer_path or "
            "MOSS_AUDIO_TOKENIZER_PATH to a valid MOSS-Audio-Tokenizer checkpoint, "
            "or use the HuggingFace repo ID OpenMOSS-Team/MOSS-Audio-Tokenizer."
        )
    return codec_path


def _get_codec_worker(device: torch.device, codec_path: str) -> CATCodecWorker:
    key = (device.type, _normalize_codec_path_for_cache(codec_path))
    if key not in _CODEC_WORKER_CACHE:
        _CODEC_WORKER_CACHE[key] = CATCodecWorker(device.type, codec_path)
    return _CODEC_WORKER_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════════════
#  Flat-code parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_flat_codes(
    flat_codes: torch.Tensor,
    n_vq: int,
) -> torch.Tensor | None:
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
    runtime_info: list[dict[str, Any]] | None,
    seq_token_counts: list[int] | None,
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
        isinstance(info.get("code_flat_numel"), int) and info["code_flat_numel"] > 0 for info in runtime_info
    ):
        sizes = [int(info["code_flat_numel"]) for info in runtime_info]
        if sum(sizes) == n:
            parts, offset = [], 0
            for sz in sizes:
                parts.append(ids[offset : offset + sz])
                offset += sz
            return parts

    if seq_token_counts and len(seq_token_counts) > 1:
        boundaries = [0]
        for c in seq_token_counts:
            boundaries.append(boundaries[-1] + c)
        return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]

    return [ids]


def _runtime_request_id(info: dict[str, Any]) -> str | None:
    request_id = info.get("request_id") or info.get("req_id")
    return str(request_id) if request_id is not None else None


def _stack_streaming_state(states: list[Any]) -> Any:
    """Stack per-request codec streaming states along batch dimension."""
    if not states:
        return None

    first = states[0]
    if isinstance(first, _BatchedScalarStreamingState):
        values = []
        for state in states:
            if not isinstance(state, _BatchedScalarStreamingState):
                raise TypeError("Cannot stack mixed batched-scalar/non-batched-scalar streaming states")
            values.extend(state.values)
        return _BatchedScalarStreamingState(values)

    if first is None:
        if all(state is None for state in states):
            return None
        raise TypeError("Cannot stack mixed None/non-None streaming states")

    if isinstance(first, torch.Tensor):
        tensors = []
        batch_dim = 0
        if first.ndim >= 2 and first.shape[0] != 1 and first.shape[1] == 1:
            # Some codec KV caches are shaped [kv, batch, ...].
            batch_dim = 1
        for state in states:
            if not isinstance(state, torch.Tensor):
                raise TypeError("Cannot stack mixed tensor/non-tensor streaming states")
            if state.ndim == 0:
                if not torch.equal(state, first):
                    raise TypeError("Cannot stack divergent scalar streaming states")
                tensors.append(state.reshape(1))
            elif state.ndim > batch_dim and state.shape[batch_dim] == 1:
                tensors.append(state)
            else:
                raise TypeError(
                    "Expected per-request streaming tensor with batch dim 1, "
                    f"got {tuple(state.shape)} at dim {batch_dim}"
                )
        return torch.cat(tensors, dim=batch_dim)

    if isinstance(first, dict):
        keys = set(first.keys())
        if any(not isinstance(state, dict) or set(state.keys()) != keys for state in states):
            raise TypeError("Cannot stack streaming-state dicts with different keys")
        return {key: _stack_streaming_state([state[key] for state in states]) for key in first.keys()}

    if isinstance(first, list):
        size = len(first)
        if any(not isinstance(state, list) or len(state) != size for state in states):
            raise TypeError("Cannot stack streaming-state lists with different lengths")
        return [_stack_streaming_state([state[i] for state in states]) for i in range(size)]

    if isinstance(first, tuple):
        size = len(first)
        if any(not isinstance(state, tuple) or len(state) != size for state in states):
            raise TypeError("Cannot stack streaming-state tuples with different lengths")
        values = [_stack_streaming_state([state[i] for state in states]) for i in range(size)]
        if hasattr(first, "_fields"):
            return type(first)(*values)
        return type(first)(values)

    if isinstance(first, _STREAMING_STATE_SCALAR_TYPES):
        if all(state == first for state in states):
            return first
        if any(not isinstance(state, type(first)) for state in states):
            raise TypeError("Cannot stack scalar streaming-state values with different types")
        return _BatchedScalarStreamingState(states)

    if hasattr(first, "__dict__"):
        keys = set(vars(first).keys())
        if any(not hasattr(state, "__dict__") or set(vars(state).keys()) != keys for state in states):
            raise TypeError("Cannot stack streaming-state objects with different attributes")
        result = copy.copy(first)
        for key in vars(first).keys():
            setattr(result, key, _stack_streaming_state([getattr(state, key) for state in states]))
        return result

    raise TypeError(f"Unsupported streaming-state type: {type(first)!r}")


def _split_streaming_state(state: Any, batch_size: int) -> list[Any]:
    """Split a batched codec streaming state into per-request states."""
    if state is None:
        return [None] * batch_size

    if isinstance(state, _BatchedScalarStreamingState):
        if len(state.values) != batch_size:
            raise TypeError(
                "Cannot split batched scalar streaming state: "
                f"{len(state.values)} values does not match batch size {batch_size}"
            )
        return list(state.values)

    if isinstance(state, torch.Tensor):
        if state.ndim == 0:
            return [state.clone() for _ in range(batch_size)]
        batch_dim = 0
        if (
            state.ndim >= 2
            and state.shape[1] == batch_size
            and (state.shape[0] != batch_size or (state.ndim >= 4 and state.shape[0] == 2))
        ):
            batch_dim = 1
        if state.shape[batch_dim] != batch_size:
            raise TypeError(
                "Cannot split streaming tensor: "
                f"batch dim {state.shape[batch_dim]} at dim {batch_dim} does not match batch size {batch_size}"
            )
        return [state.narrow(batch_dim, i, 1).clone() for i in range(batch_size)]

    if isinstance(state, dict):
        split_values = {key: _split_streaming_state(value, batch_size) for key, value in state.items()}
        return [{key: values[i] for key, values in split_values.items()} for i in range(batch_size)]

    if isinstance(state, list):
        split_items = [_split_streaming_state(value, batch_size) for value in state]
        return [[values[i] for values in split_items] for i in range(batch_size)]

    if isinstance(state, tuple):
        split_items = [_split_streaming_state(value, batch_size) for value in state]
        if hasattr(state, "_fields"):
            return [type(state)(*[values[i] for values in split_items]) for i in range(batch_size)]
        return [type(state)(values[i] for values in split_items) for i in range(batch_size)]

    if isinstance(state, _STREAMING_STATE_SCALAR_TYPES):
        return [state for _ in range(batch_size)]

    if hasattr(state, "__dict__"):
        split_attrs = {key: _split_streaming_state(value, batch_size) for key, value in vars(state).items()}
        result = []
        for i in range(batch_size):
            item = copy.copy(state)
            for key, values in split_attrs.items():
                setattr(item, key, values[i])
            result.append(item)
        return result

    raise TypeError(f"Unsupported streaming-state type: {type(state)!r}")


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

    have_multimodal_outputs = True
    enable_update_additional_information = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        self.config = cfg

        self.n_vq: int = cfg.n_vq  # 32
        self.sample_rate: int = getattr(cfg, "sampling_rate", 24_000)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Resolve CAT codec checkpoint path.
        # Resolution order:
        #   1. model_config.audio_tokenizer_path  (set in YAML engine_args)
        #   2. MOSS_AUDIO_TOKENIZER_PATH env var   (local path or HF repo ID)
        #   3. Hard-coded HF default               (auto-downloads on first use)
        self.audio_tokenizer_path = _resolve_codec_path(vllm_config)

        self._codec: CATCodecWorker = _get_codec_worker(self.device, self.audio_tokenizer_path)

        # Per-request streaming state: request_id -> {module_index: _streaming_state}.
        #
        # The CAT codec stores streaming KV/cache buffers on shared decoder modules.
        # Holding multiple decoder_module.streaming() contexts at the same time
        # fails with "is already streaming!". To support async multi-request decode,
        # keep each request's streaming states off-module between chunk calls and
        # temporarily re-attach only the active request's state during _decode_frame.
        self._streaming_states: dict[str, dict[int, Any]] = {}
        self._active_request_id: str | None = None
        self._streaming_modules_cache: list | None = None
        self._batch_stateless_decode_failed = False
        self._batch_streaming_decode_failed = False
        self._logged_stateless_batch_decode = False
        self._logged_streaming_batch_decode = False
        self._profile_decoder = os.environ.get("MOSS_TTS_DECODER_PROFILE") == "1"
        self._decoder_profile_stats = {
            "forward_calls": 0,
            "batch_streaming_calls": 0,
            "batch_streaming_wall_s": 0.0,
            "batch_stateless_calls": 0,
            "batch_stateless_wall_s": 0.0,
            "single_decode_calls": 0,
            "single_decode_wall_s": 0.0,
        }
        if self._profile_decoder:
            atexit.register(self._log_decoder_profile)

        # Dummy logits processor / sampler required by vllm's model protocol
        self.logits_processor = LogitsProcessor(cfg.language_config.vocab_size)
        self.sampler = Sampler()

    # ══════════════════════════════════════════════════════════════════
    #  Core decode logic
    # ══════════════════════════════════════════════════════════════════

    def _get_all_streaming_modules(self) -> list:
        """Return a stable ordered list of every StreamingModule in the codec decoder."""
        if self._streaming_modules_cache is not None:
            return self._streaming_modules_cache

        modules: list = []
        seen: set[int] = set()
        for decoder_module in self._codec.codec.decoder:
            if not (hasattr(decoder_module, "streaming") and callable(decoder_module.streaming)):
                continue

            # Build the codec module's cached streaming children list without
            # mutating any streaming state.
            if hasattr(decoder_module, "_apply_named_streaming"):
                decoder_module._apply_named_streaming(lambda _name, _module: None)
            cached_children = getattr(decoder_module, "_cached_children", None)
            if cached_children is None:
                continue

            for _, module in cached_children:
                if id(module) not in seen:
                    seen.add(id(module))
                    modules.append(module)

        self._streaming_modules_cache = modules
        return modules

    def _enter_streaming(self, request_id: str) -> None:
        """Allocate fresh codec streaming state for a new request, then suspend it."""
        if request_id in self._streaming_states:
            return
        if self._active_request_id is not None:
            self._suspend_active_streaming()

        for decoder_module in self._codec.codec.decoder:
            if hasattr(decoder_module, "_start_streaming"):
                decoder_module._start_streaming(batch_size=1, exit_stack=ExitStack())

        modules = self._get_all_streaming_modules()
        states = {i: module._streaming_state for i, module in enumerate(modules)}

        # Detach from shared modules so another request can allocate/activate its
        # own streaming state without tripping the codec's single-active guard.
        for module in modules:
            module._streaming_state = None
        self._streaming_states[request_id] = states

    def _activate_streaming(self, request_id: str) -> None:
        """Attach one request's saved streaming state to the shared codec modules."""
        if self._active_request_id == request_id:
            return
        if self._active_request_id is not None:
            self._suspend_active_streaming()

        states = self._streaming_states[request_id]
        modules = self._get_all_streaming_modules()
        for i, module in enumerate(modules):
            module._streaming_state = states.get(i)
        self._active_request_id = request_id

    def _suspend_active_streaming(self) -> None:
        """Detach streaming states from codec modules without freeing KV/cache buffers."""
        for module in self._get_all_streaming_modules():
            module._streaming_state = None
        self._active_request_id = None

    def _activate_streaming_batch(self, request_ids: list[str]) -> None:
        """Attach a stacked batch of per-request streaming states."""
        if self._active_request_id is not None:
            self._suspend_active_streaming()

        for request_id in request_ids:
            if request_id not in self._streaming_states:
                self._enter_streaming(request_id)

        modules = self._get_all_streaming_modules()
        for module_index, module in enumerate(modules):
            states = [self._streaming_states[request_id].get(module_index) for request_id in request_ids]
            module._streaming_state = _stack_streaming_state(states)
        self._active_request_id = None

    def _save_streaming_batch(self, request_ids: list[str]) -> None:
        """Persist a batched streaming state back into per-request slots."""
        modules = self._get_all_streaming_modules()
        batch_size = len(request_ids)
        for module_index, module in enumerate(modules):
            split_states = _split_streaming_state(module._streaming_state, batch_size)
            for request_id, state in zip(request_ids, split_states):
                self._streaming_states.setdefault(request_id, {})[module_index] = state
            module._streaming_state = None
        self._active_request_id = None

    def _exit_streaming(self, request_id: str) -> None:
        """Discard per-request streaming state for a finished request."""
        if self._active_request_id == request_id:
            self._suspend_active_streaming()
        self._streaming_states.pop(request_id, None)

    def on_requests_finished(self, request_ids) -> None:
        for request_id in request_ids:
            self._exit_streaming(str(request_id))

    def _empty_audio(self) -> torch.Tensor:
        return torch.zeros(0, dtype=torch.float32, device=self.device)

    def _pack_code_batch(
        self,
        request_codes_list: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[int], torch.Tensor | None, torch.Tensor | None]:
        """Parse and pad flat per-request codec codes into [n_vq, B, max_T]."""
        empty = self._empty_audio()
        num_req = len(request_codes_list)
        results: list[torch.Tensor] = [empty] * num_req
        parsed_codes: list[torch.Tensor] = []
        valid_indices: list[int] = []
        lengths: list[int] = []

        for i, req_codes in enumerate(request_codes_list):
            if req_codes is None or req_codes.numel() == 0:
                continue
            codes = _parse_flat_codes(req_codes, self.n_vq)
            if codes is None:
                continue
            parsed_codes.append(codes)
            valid_indices.append(i)
            lengths.append(codes.shape[-1])

        if not parsed_codes:
            return results, valid_indices, None, None

        max_len = max(lengths)
        # Pad with code 0, not audio_pad_code. audio_pad_code is outside the
        # codec codebook and would fail embedding lookup if the backend ignores
        # lengths. The decode result is trimmed by audio_lengths below.
        batch_codes = torch.zeros(
            (self.n_vq, len(parsed_codes), max_len),
            dtype=torch.long,
            device=self.device,
        )
        for batch_idx, codes in enumerate(parsed_codes):
            T = codes.shape[-1]
            batch_codes[:, batch_idx, :T] = codes.to(self.device)

        batch_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        return results, valid_indices, batch_codes, batch_lengths

    def _extract_audio_list(
        self,
        decode_result: Any,
        batch_size: int,
    ) -> list[torch.Tensor]:
        """Normalize codec decode outputs to a list of 1-D float32 tensors."""
        looks_like_audio_lengths_pair = (
            isinstance(decode_result, (tuple, list))
            and len(decode_result) == 2
            and all(isinstance(item, torch.Tensor) for item in decode_result)
            and decode_result[0].ndim >= 2
            and decode_result[1].ndim <= 1
            and decode_result[1].numel() == batch_size
        )
        if (
            isinstance(decode_result, (tuple, list))
            and not looks_like_audio_lengths_pair
            and len(decode_result) == batch_size
            and all(isinstance(item, torch.Tensor) for item in decode_result)
            and all(item.ndim <= 2 for item in decode_result)
        ):
            wavs = []
            for item in decode_result:
                wav = item
                if wav.ndim > 1:
                    wav = wav[0]
                wavs.append(wav.to(self.device, dtype=torch.float32).reshape(-1))
            return wavs

        audio = getattr(decode_result, "audio", None)
        audio_lengths = getattr(decode_result, "audio_lengths", None)

        if audio is None and isinstance(decode_result, (tuple, list)) and decode_result:
            audio = decode_result[0]
            if len(decode_result) > 1:
                audio_lengths = decode_result[1]
        if audio is None:
            raise TypeError(f"Codec decode result has no audio tensor: {type(decode_result)!r}")

        if audio.shape[0] != batch_size and audio.ndim >= 3 and audio.shape[1] == batch_size:
            audio = audio.transpose(0, 1)

        if audio.shape[0] != batch_size:
            raise ValueError(f"Codec returned audio batch={audio.shape[0]}, expected {batch_size}")

        if audio_lengths is None:
            audio_lengths = torch.full(
                (batch_size,),
                audio.shape[-1],
                dtype=torch.long,
                device=audio.device,
            )
        elif not isinstance(audio_lengths, torch.Tensor):
            audio_lengths = torch.tensor(audio_lengths, dtype=torch.long, device=audio.device)
        else:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long).reshape(-1)

        wavs: list[torch.Tensor] = []
        for i in range(batch_size):
            wav = audio[i]
            if wav.ndim > 1:
                wav = wav[0]
            valid_len = int(audio_lengths[i].item()) if i < audio_lengths.numel() else wav.numel()
            wavs.append(wav[:valid_len].to(self.device, dtype=torch.float32).reshape(-1))
        return wavs

    def _decode_one_request(
        self,
        flat_codes: torch.Tensor,
        request_id: str | None = None,
        is_finished: bool = False,
    ) -> torch.Tensor:
        """
        Decode one request's flat code tensor to a waveform.

        When request_id is provided, uses the codec's streaming KV-cache so that
        causal state is maintained across successive chunk calls for the same request.

        flat_codes  : [T * n_vq]  long
        request_id  : str or None — if set, enables per-request streaming state
        is_finished : bool — if True, streaming state is released after this call
        Returns     : [samples]   float32  (empty tensor if codes are invalid)
        """
        empty = torch.zeros(0, dtype=torch.float32, device=self.device)
        profile_t0 = time.perf_counter() if self._profile_decoder else 0.0

        if flat_codes is None or flat_codes.numel() == 0:
            if request_id and is_finished:
                self._exit_streaming(request_id)
            if self._profile_decoder:
                self._decoder_profile_stats["single_decode_calls"] += 1
                self._decoder_profile_stats["single_decode_wall_s"] += time.perf_counter() - profile_t0
            return empty

        codes = _parse_flat_codes(flat_codes, self.n_vq)  # [n_vq, T] or None
        if codes is None:
            if request_id and is_finished:
                self._exit_streaming(request_id)
            if self._profile_decoder:
                self._decoder_profile_stats["single_decode_calls"] += 1
                self._decoder_profile_stats["single_decode_wall_s"] += time.perf_counter() - profile_t0
            return empty

        try:
            if request_id is not None:
                # Streaming path: maintain per-request causal KV/cache across
                # chunks while sharing one codec instance among concurrent requests.
                if request_id not in self._streaming_states:
                    self._enter_streaming(request_id)
                self._activate_streaming(request_id)
                codec = self._codec.codec
                codes_3d = codes.unsqueeze(1).to(self.device)  # [n_vq, 1, T]
                lengths = torch.tensor([codes_3d.shape[-1]], device=self.device, dtype=torch.long)
                result = codec._decode_frame(codes_3d, lengths)
                wav = self._extract_audio_list(result, 1)[0]
                self._suspend_active_streaming()
                if is_finished:
                    self._exit_streaming(request_id)
            else:
                # Fallback: stateless single-call decode (sync / non-streaming mode).
                wav = self._codec.decode(codes)

            return wav.to(self.device)
        except Exception as exc:
            logger.error("[MossTTS Decoder] Codec decode failed: %s", exc, exc_info=True)
            self._suspend_active_streaming()
            if request_id and is_finished:
                self._exit_streaming(request_id)
            return empty
        finally:
            if self._profile_decoder:
                self._decoder_profile_stats["single_decode_calls"] += 1
                self._decoder_profile_stats["single_decode_wall_s"] += time.perf_counter() - profile_t0

    def _batch_decode_stateless(
        self,
        request_codes_list: list[torch.Tensor],
    ) -> list[torch.Tensor] | None:
        """Decode sync-mode Stage-1 requests in one CAT codec forward."""
        if len(request_codes_list) <= 1 or self._batch_stateless_decode_failed:
            return None
        profile_t0 = time.perf_counter() if self._profile_decoder else 0.0
        if self._profile_decoder:
            self._decoder_profile_stats["batch_stateless_calls"] += 1

        results, valid_indices, batch_codes, batch_lengths = self._pack_code_batch(request_codes_list)
        if batch_codes is None or batch_lengths is None:
            return results
        if len(valid_indices) <= 1:
            return None

        codec = self._codec.codec
        batch_decode = getattr(codec, "batch_decode", None)
        if callable(batch_decode):
            try:
                if self._active_request_id is not None:
                    self._suspend_active_streaming()
                codes_btc = batch_codes.permute(1, 2, 0).contiguous()
                padding_mask = (
                    torch.arange(batch_codes.shape[-1], device=self.device)[None, :]
                    < batch_lengths[:, None]
                )
                decode_result = batch_decode(codes_btc, padding_mask=padding_mask)
                wavs = self._extract_audio_list(decode_result, len(valid_indices))
                for slot, wav in zip(valid_indices, wavs):
                    results[slot] = wav
                if not self._logged_stateless_batch_decode:
                    logger.info(
                        "[MossTTS Decoder] Enabled public batched stateless CAT decode: batch=%d",
                        len(valid_indices),
                    )
                    self._logged_stateless_batch_decode = True
                return results
            except Exception as exc:
                logger.debug(
                    "[MossTTS Decoder] Public batch_decode failed; trying _decode_frame. Error: %s",
                    exc,
                    exc_info=True,
                )
                self._suspend_active_streaming()

        decode_frame = getattr(codec, "_decode_frame", None)
        if not callable(decode_frame):
            self._batch_stateless_decode_failed = True
            return None

        try:
            if self._active_request_id is not None:
                self._suspend_active_streaming()
            decode_result = decode_frame(batch_codes, batch_lengths)
            wavs = self._extract_audio_list(decode_result, len(valid_indices))
            for slot, wav in zip(valid_indices, wavs):
                results[slot] = wav
            if not self._logged_stateless_batch_decode:
                logger.info(
                    "[MossTTS Decoder] Enabled batched stateless CAT decode: batch=%d",
                    len(valid_indices),
                )
                self._logged_stateless_batch_decode = True
            return results
        except Exception as exc:
            self._batch_stateless_decode_failed = True
            logger.warning(
                "[MossTTS Decoder] Batched stateless CAT decode failed; "
                "falling back to per-request decode. Error: %s",
                exc,
                exc_info=True,
            )
            self._suspend_active_streaming()
            return None
        finally:
            if self._profile_decoder:
                self._decoder_profile_stats["batch_stateless_wall_s"] += time.perf_counter() - profile_t0

    def _batch_decode_streaming(
        self,
        request_codes_list: list[torch.Tensor],
        request_ids: list[str | None],
        finished_flags: list[bool],
    ) -> list[torch.Tensor] | None:
        """Decode async chunks for multiple requests in one CAT codec forward."""
        if len(request_codes_list) <= 1 or self._batch_streaming_decode_failed:
            return None
        if any(request_id is None for request_id in request_ids):
            return None
        results, valid_indices, batch_codes, batch_lengths = self._pack_code_batch(request_codes_list)
        if batch_codes is None or batch_lengths is None:
            for i, request_id in enumerate(request_ids):
                if request_id and finished_flags[i]:
                    self._exit_streaming(request_id)
            return results
        if len(valid_indices) <= 1:
            return None

        valid_request_ids = [str(request_ids[i]) for i in valid_indices]
        profile_t0 = time.perf_counter() if self._profile_decoder else 0.0
        if self._profile_decoder:
            self._decoder_profile_stats["batch_streaming_calls"] += 1

        try:
            self._activate_streaming_batch(valid_request_ids)
            codec = self._codec.codec
            decode_result = codec._decode_frame(batch_codes, batch_lengths)
            wavs = self._extract_audio_list(decode_result, len(valid_indices))
            self._save_streaming_batch(valid_request_ids)

            for slot, request_id, wav in zip(valid_indices, valid_request_ids, wavs):
                results[slot] = wav
                if finished_flags[slot]:
                    self._exit_streaming(request_id)
            if not self._logged_streaming_batch_decode:
                logger.info(
                    "[MossTTS Decoder] Enabled batched streaming CAT decode: batch=%d",
                    len(valid_indices),
                )
                self._logged_streaming_batch_decode = True
            return results
        except Exception as exc:
            self._batch_streaming_decode_failed = True
            logger.warning(
                "[MossTTS Decoder] Batched streaming CAT decode failed; "
                "falling back to per-request streaming decode. Error: %s",
                exc,
                exc_info=True,
            )
            self._suspend_active_streaming()
            return None
        finally:
            if self._profile_decoder:
                self._decoder_profile_stats["batch_streaming_wall_s"] += time.perf_counter() - profile_t0

    @torch.inference_mode()
    def _batch_decode(
        self,
        request_codes_list: list[torch.Tensor],
        request_ids: list[str | None] | None = None,
        finished_flags: list[bool] | None = None,
    ) -> list[torch.Tensor]:
        """Decode multiple requests, using per-request streaming state when available."""
        empty = self._empty_audio()
        has_streaming_metadata = (
            request_ids is not None
            and finished_flags is not None
            and len(request_ids) == len(request_codes_list)
            and len(finished_flags) == len(request_codes_list)
        )

        if has_streaming_metadata:
            batched_streaming = self._batch_decode_streaming(request_codes_list, request_ids, finished_flags)
            if batched_streaming is not None:
                return batched_streaming
        else:
            batched_stateless = self._batch_decode_stateless(request_codes_list)
            if batched_stateless is not None:
                return batched_stateless

        results: list[torch.Tensor] = []
        for i, req_codes in enumerate(request_codes_list):
            req_id = request_ids[i] if has_streaming_metadata else None
            finished = finished_flags[i] if has_streaming_metadata else False
            wav = self._decode_one_request(req_codes, request_id=req_id, is_finished=finished)
            results.append(wav if wav.numel() > 0 else empty)
        return results

    # ══════════════════════════════════════════════════════════════════
    #  Forward
    # ══════════════════════════════════════════════════════════════════

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        codes: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> OmniOutput:
        if self._profile_decoder:
            self._decoder_profile_stats["forward_calls"] += 1

        # Resolve runtime_additional_information from alternate kwargs keys
        if runtime_additional_information is None:
            runtime_additional_information = kwargs.get("model_intermediate_buffer") or kwargs.get(
                "runtime_additional_information"
            )

        code_tensor = codes if codes is not None else input_ids

        # When input_ids is empty/None but codes are in runtime_additional_information,
        # extract them directly from code_predictor_codes (async_chunk path).
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

        empty = torch.zeros(0, dtype=torch.float32, device=self.device)

        if code_tensor is None or code_tensor.numel() == 0:
            # Even with no codes, a finished sentinel must close streaming state.
            if runtime_additional_information:
                for info in runtime_additional_information:
                    if not isinstance(info, dict):
                        continue
                    req_id = _runtime_request_id(info)
                    fin = info.get("finished")
                    if isinstance(fin, torch.Tensor):
                        fin = bool(fin.item())
                    if req_id and fin:
                        self._exit_streaming(req_id)
            n = len(runtime_additional_information) if runtime_additional_information else 1
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty] * n},
            )

        # Skip decode during CUDA graph capture
        if torch.cuda.is_current_stream_capturing():
            n = len(runtime_additional_information) if runtime_additional_information else 1
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty] * n},
            )

        ids = code_tensor.reshape(-1).to(torch.long)

        # Split flat tensor into per-request slices
        request_codes_list = _split_per_request(
            ids,
            runtime_additional_information,
            kwargs.get("seq_token_counts"),
        )

        # Extract per-request streaming metadata when available
        request_ids: list[str | None] | None = None
        finished_flags: list[bool] | None = None
        if runtime_additional_information:
            request_ids = []
            finished_flags = []
            for info in runtime_additional_information:
                request_ids.append(_runtime_request_id(info) if isinstance(info, dict) else None)
                fin = info.get("finished") if isinstance(info, dict) else None
                if isinstance(fin, torch.Tensor):
                    fin = bool(fin.item())
                finished_flags.append(bool(fin) if fin is not None else False)

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
            input_ids.shape[0],
            hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _log_decoder_profile(self) -> None:
        if not self._profile_decoder:
            return
        logger.warning(
            "[MossTTSDecoderProfile] forward_calls=%d batch_streaming_calls=%d "
            "batch_streaming_wall_s=%.6f batch_stateless_calls=%d "
            "batch_stateless_wall_s=%.6f single_decode_calls=%d single_decode_wall_s=%.6f",
            self._decoder_profile_stats["forward_calls"],
            self._decoder_profile_stats["batch_streaming_calls"],
            self._decoder_profile_stats["batch_streaming_wall_s"],
            self._decoder_profile_stats["batch_stateless_calls"],
            self._decoder_profile_stats["batch_stateless_wall_s"],
            self._decoder_profile_stats["single_decode_calls"],
            self._decoder_profile_stats["single_decode_wall_s"],
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if hidden_states is None or hidden_states.numel() == 0:
            return None
        vocab_size = self.config.language_config.vocab_size
        return torch.zeros(
            hidden_states.shape[0],
            vocab_size,
            dtype=hidden_states.dtype,
            device=self.device,
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if logits is None or logits.numel() == 0:
            return None
        return self.sampler(logits, sampling_metadata)

    def make_omni_output(self, model_output: Any, **kwargs) -> OmniOutput:
        if isinstance(model_output, OmniOutput):
            return model_output
        empty = torch.zeros(0, dtype=torch.float32, device=self.device)
        if model_output is None:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty]},
            )
        if isinstance(model_output, torch.Tensor):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [model_output.float().reshape(-1)]},
            )
        raise TypeError(f"Unexpected model output type: {type(model_output)}")

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        **kwargs,
    ) -> set[str]:
        # The CAT codec is loaded separately via from_pretrained().
        # No weights to load through vllm's standard mechanism.
        return set()
