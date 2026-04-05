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
# The CAT codec is loaded from the MOSS-Audio-Tokenizer HuggingFace repo:
#   OpenMOSS/MOSS-Audio-Tokenizer
# or the combined checkpoint:
#   OpenMOSS-Team/MOSS-TTS-Local-Transformer
#
# Environment variable:
#   MOSS_AUDIO_TOKENIZER_PATH  — path to the CAT codec checkpoint directory.
#                                Falls back to the main model path if unset.

import logging
import os
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)


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

        #       Typical pattern:
        #         from moss_audio_tokenizer import MossAudioTokenizer
        #         self.codec = MossAudioTokenizer.from_pretrained(codec_path)
        #
        # Fallback using AutoModel (works if the repo registers itself):
        from transformers import AutoModel
        self.codec = AutoModel.from_pretrained(
            codec_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.codec = self.codec.to(self.device).eval()

        self.sample_rate: int = getattr(self.codec.config, "sampling_rate", 24_000)
        self.n_vq: int        = getattr(self.codec.config, "n_vq", 32)

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
        """
        codes = codes.to(self.device)
        # The CAT codec decode API — verify the exact method name from the repo.
        # Common patterns:
        #   self.codec.decode(codes)
        #   self.codec.decoder(codes)
        #   self.codec.detokenize(codes)
        wav = self.codec.decode(codes)   # TODO: confirm API
        return wav.float().reshape(-1).cpu()


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

        self.n_vq: int       = cfg.n_vq           # 32
        self.sample_rate: int = cfg.sampling_rate  # 24000

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Resolve CAT codec checkpoint path
        codec_path = (
            getattr(vllm_config.model_config, "audio_tokenizer_path", None)
            or os.environ.get("MOSS_AUDIO_TOKENIZER_PATH")
            or cfg.name_or_path  # fall back to main model path if codec is co-located
        )
        if not codec_path:
            raise ValueError(
                "[MossTTS Decoder] CAT codec path not found. "
                "Set MOSS_AUDIO_TOKENIZER_PATH or model_config.audio_tokenizer_path."
            )

        self._codec: CATCodecWorker = _get_codec_worker(self.device, codec_path)

        # Dummy logits processor / sampler required by vllm's model protocol
        self.logits_processor = LogitsProcessor(cfg.language_config.vocab_size)
        self.sampler = Sampler()

    # ══════════════════════════════════════════════════════════════════
    #  Core decode logic
    # ══════════════════════════════════════════════════════════════════

    def _decode_one_request(self, flat_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode one request's flat code tensor to a waveform.

        flat_codes : [T * n_vq]  long
        Returns    : [samples]   float32  (empty tensor if codes are invalid)
        """
        empty = torch.zeros(0, dtype=torch.float32, device=self.device)

        if flat_codes is None or flat_codes.numel() == 0:
            return empty

        codes = _parse_flat_codes(flat_codes, self.n_vq)  # [n_vq, T] or None
        if codes is None:
            return empty

        # Skip all-zero code tensors (dummy / padding frames)
        if not codes.any():
            return empty

        try:
            wav = self._codec.decode(codes)   # [samples] float32 CPU
            return wav.to(self.device)
        except Exception as exc:
            logger.error("[MossTTS Decoder] Codec decode failed: %s", exc)
            return empty

    @torch.inference_mode()
    def _batch_decode(
        self,
        request_codes_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Batch-decode multiple requests.

        For efficiency, this calls the codec once per request.
        TODO (post-MVP): batch all requests into a single codec forward pass
        by packing hidden states (same optimisation as MiMo's _batch_decode_waveforms).
        """
        empty = torch.zeros(0, dtype=torch.float32, device=self.device)
        results: list[torch.Tensor] = []
        for req_codes in request_codes_list:
            wav = self._decode_one_request(req_codes)
            results.append(wav if wav.numel() > 0 else empty)
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
        # Resolve runtime_additional_information from alternate kwargs keys
        if runtime_additional_information is None:
            runtime_additional_information = (
                kwargs.get("model_intermediate_buffer")
                or kwargs.get("runtime_additional_information")
            )

        code_tensor = codes if codes is not None else input_ids
        empty       = torch.zeros(0, dtype=torch.float32, device=self.device)

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

        ids = code_tensor.reshape(-1).to(torch.long)

        # Split flat tensor into per-request slices
        request_codes_list = _split_per_request(
            ids,
            runtime_additional_information,
            kwargs.get("seq_token_counts"),
        )

        audios = self._batch_decode(request_codes_list)

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
        empty = torch.zeros(0, dtype=torch.float32, device=self.device)
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
