# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
#
# Unified MOSS-TTS-Local entry point.
# Dispatches to one of two stage-specific sub-models based on the
# `model_stage` field set in the YAML stage config:
#
#   "ar_stage" → MossTTSARStageModel   (global Qwen3 backbone + local transformer)
#   "decoder"  → MossTTSDecoderModel   (CAT codec: RVQ codes → waveform audio)

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.moss_tts.moss_tts_ar_stage import MossTTSARStageModel
from vllm_omni.model_executor.models.moss_tts.moss_tts_decoder import MossTTSDecoderModel
from vllm_omni.model_executor.models.output_templates import OmniOutput


class MossTTSForConditionalGeneration(nn.Module, SupportsPP):
    """
    Unified MOSS-TTS-Local model for vllm-omni multi-stage inference.

    Two-stage pipeline
    ------------------
    Stage 0  "ar_stage"
        Global Qwen3-1.7B transformer produces per-step hidden states.
        A lightweight local transformer (4 blocks, no positional embeddings)
        autoregressively predicts 32 RVQ codes per audio frame.
        Output: OmniOutput carrying code_predictor_codes [B, 1, n_vq, 1].

    Stage 1  "decoder"
        The CAT codec (MOSS-Audio-Tokenizer, 1.6 B) converts the accumulated
        RVQ code sequence [n_vq, T] into a 24 kHz waveform tensor.
        Output: OmniOutput carrying model_outputs (list of float tensors).

    Both stages implement the vllm-omni model protocol:
        embed_input_ids / forward / compute_logits / sample / load_weights
    """

    have_multimodal_outputs = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        # model_stage is injected by the YAML engine_args block
        self._stage: str = getattr(vllm_config.model_config, "model_stage", "ar_stage")

        if self._stage == "ar_stage":
            self._model: nn.Module = MossTTSARStageModel(vllm_config, prefix=prefix)
        elif self._stage == "decoder":
            self._model = MossTTSDecoderModel(vllm_config, prefix=prefix)
        else:
            raise ValueError(
                f"[MossTTS] Unknown model_stage={self._stage!r}. "
                f"Expected 'ar_stage' or 'decoder'."
            )

    # ------------------------------------------------------------------ #
    #  Core forward — delegates entirely to the active stage sub-model    #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "OmniOutput | torch.Tensor | IntermediateTensors":
        return self._model.forward(input_ids=input_ids, **kwargs)

    # ------------------------------------------------------------------ #
    #  vllm model protocol                                                #
    # ------------------------------------------------------------------ #

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: bool = False,
    ) -> torch.Tensor:
        if hasattr(self._model, "embed_input_ids"):
            return self._model.embed_input_ids(
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        raise AttributeError(
            f"Stage model {type(self._model).__name__} does not implement embed_input_ids"
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if hasattr(self._model, "compute_logits"):
            return self._model.compute_logits(hidden_states)
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        if hasattr(self._model, "sample"):
            return self._model.sample(logits, sampling_metadata)
        return None

    def make_omni_output(self, model_output, **kwargs) -> OmniOutput:
        if hasattr(self._model, "make_omni_output"):
            return self._model.make_omni_output(model_output, **kwargs)
        if isinstance(model_output, OmniOutput):
            return model_output
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": []},
        )

    def _clear_warmup_state(self) -> None:
        if hasattr(self._model, "_clear_warmup_state"):
            self._model._clear_warmup_state()

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        **kwargs,
    ) -> set[str]:
        # self._model is stored as self._model, so vLLM names its params with
        # the "_model." prefix.  We translate the module-relative names returned
        # by the stage sub-model into the full names vLLM expects.
        loaded_relative = self._model.load_weights(weights, **kwargs)
        return {f"_model.{name}" for name in loaded_relative}
