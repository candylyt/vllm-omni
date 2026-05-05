# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
#
# Stage 0: AR stage for MOSS-TTS-Delay.
#
# Phase-1 scope:
#   - offline / non-streaming direct TTS
#   - single-stage text scheduling via vLLM
#   - internal reconstruction of the delayed audio channels
#   - CAT decoder reuse in Stage 1

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)

_UNSET_DELAY_LENGTH = -1


@dataclass
class MossTTSDelayRequestState:
    """Per-request Delay FSM state reconstructed on top of vLLM decoding."""

    n_vq: int
    audio_pad_code: int
    is_audio: bool = False
    audio_length: int = 0
    delayed_length: int = _UNSET_DELAY_LENGTH
    decode_steps_generated: int = 0
    pending_audio_row: torch.Tensor = field(init=False)
    audio_history: list[set[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.pending_audio_row = torch.full((self.n_vq,), self.audio_pad_code, dtype=torch.long)
        self.audio_history = [set() for _ in range(self.n_vq)]

    def store_next_audio_row(self, row: torch.Tensor, *, track_history: bool) -> None:
        row = row.detach().to(torch.long).reshape(self.n_vq)
        self.pending_audio_row = row
        if not track_history:
            self.decode_steps_generated += 1
            return
        for ch_idx, token in enumerate(row.cpu().tolist()):
            if token != self.audio_pad_code:
                self.audio_history[ch_idx].add(int(token))
        self.decode_steps_generated += 1


class MossTTSDelayARStageModel(nn.Module, SupportsPP):
    """vLLM-backed Delay-pattern AR stage for MOSS-TTS-Delay."""

    have_multimodal_outputs = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        self.config = cfg

        self.n_vq: int = cfg.n_vq
        self.audio_vocab_size: int = cfg.audio_vocab_size
        self.audio_pad_code: int = cfg.audio_pad_code
        self.pad_token_id: int = cfg.pad_token_id
        self.im_end_token_id: int = cfg.im_end_token_id
        self.audio_start_token_id: int = cfg.audio_start_token_id
        self.audio_end_token_id: int = cfg.audio_end_token_id
        self.audio_assistant_gen_slot_token_id: int = cfg.audio_assistant_gen_slot_token_id
        self.audio_assistant_delay_slot_token_id: int = cfg.audio_assistant_delay_slot_token_id
        self.audio_user_slot_token_id: int = cfg.audio_user_slot_token_id

        lang_cfg = cfg.language_config
        self.hidden_size: int = lang_cfg.hidden_size

        qwen3_vllm_config = copy.deepcopy(vllm_config)
        object.__setattr__(qwen3_vllm_config.model_config, "hf_config", lang_cfg)

        from vllm.model_executor.models.qwen3 import Qwen3Model

        backbone_prefix = f"{prefix}language_model" if prefix else "language_model"
        self.backbone = Qwen3Model(
            vllm_config=qwen3_vllm_config,
            prefix=backbone_prefix,
        )

        audio_vocab_with_pad = self.audio_vocab_size + 1
        self.audio_embedding_weight = nn.Parameter(
            torch.empty(self.n_vq, audio_vocab_with_pad, self.hidden_size)
        )
        nn.init.normal_(self.audio_embedding_weight)
        with torch.no_grad():
            self.audio_embedding_weight[:, self.audio_pad_code].zero_()

        self.text_lm_head = nn.Linear(self.hidden_size, lang_cfg.vocab_size, bias=False)
        self.audio_lm_head_weight = nn.Parameter(
            torch.empty(self.n_vq, audio_vocab_with_pad, self.hidden_size)
        )
        for ch_idx in range(self.n_vq):
            nn.init.kaiming_uniform_(self.audio_lm_head_weight[ch_idx], a=math.sqrt(5))
        self.sampler = Sampler()

        self._request_states: dict[str, MossTTSDelayRequestState] = {}
        self._last_request_ids: list[str] = []
        self._last_seq_lens: list[int] = []

    # ------------------------------------------------------------------ #
    # Request helpers                                                     #
    # ------------------------------------------------------------------ #

    def _new_request_state(self) -> MossTTSDelayRequestState:
        return MossTTSDelayRequestState(
            n_vq=self.n_vq,
            audio_pad_code=self.audio_pad_code,
        )

    def _extract_request_schedule(
        self,
        runtime_additional_information: list[dict] | None,
    ) -> tuple[list[str], list[int], list[int]]:
        try:
            from vllm.forward_context import get_forward_context

            ctx = get_forward_context()
            attn_meta_dict = ctx.attn_metadata
            if not attn_meta_dict:
                return [], [], []
            if isinstance(attn_meta_dict, list):
                attn_meta_dict = attn_meta_dict[0]
            attn_meta = next(iter(attn_meta_dict.values()))
            qsl = attn_meta.query_start_loc.cpu().tolist()
            seq_lens = [qsl[i + 1] - qsl[i] for i in range(len(qsl) - 1)]
            decode_positions = [qsl[i] for i, slen in enumerate(seq_lens) if slen == 1]
        except Exception as exc:
            logger.debug("[MossTTS Delay] Failed to extract request schedule: %s", exc)
            return [], [], []

        if runtime_additional_information:
            request_ids = [info.get("req_id", str(i)) for i, info in enumerate(runtime_additional_information)]
        else:
            request_ids = [str(i) for i in range(len(seq_lens))]
        return request_ids, seq_lens, decode_positions

    def _advance_state_with_text_token(
        self,
        state: MossTTSDelayRequestState,
        token_id: int,
    ) -> None:
        if token_id in (
            self.audio_start_token_id,
            self.audio_assistant_gen_slot_token_id,
            self.audio_assistant_delay_slot_token_id,
        ):
            state.is_audio = True
            state.audio_length += 1

            if token_id == self.audio_assistant_delay_slot_token_id:
                if state.delayed_length == _UNSET_DELAY_LENGTH:
                    state.delayed_length = 1
                else:
                    state.delayed_length += 1
                if state.delayed_length > self.n_vq:
                    state.delayed_length = _UNSET_DELAY_LENGTH
            return

        if token_id == self.audio_end_token_id:
            state.is_audio = False
            state.audio_length = 0
            state.delayed_length = _UNSET_DELAY_LENGTH

    def _reset_prefill_state(
        self,
        request_id: str,
        prompt_tokens: torch.Tensor,
    ) -> MossTTSDelayRequestState:
        state = self._new_request_state()
        tokens = prompt_tokens.reshape(-1).tolist()

        unsupported = any(
            token
            in (
                self.audio_user_slot_token_id,
                self.audio_assistant_gen_slot_token_id,
                self.audio_assistant_delay_slot_token_id,
            )
            for token in tokens
        )
        if unsupported:
            logger.warning(
                "[MossTTS Delay] Request %s contains continuation / cloning prompt tokens. "
                "Phase-1 only guarantees direct TTS prompts.",
                request_id,
            )

        for token in tokens:
            self._advance_state_with_text_token(state, int(token))

        self._request_states[request_id] = state
        return state

    def _prepare_request_states(
        self,
        input_ids: torch.Tensor,
        request_ids: list[str],
        seq_lens: list[int],
    ) -> tuple[list[int], list[MossTTSDelayRequestState]]:
        decode_positions: list[int] = []
        decode_states: list[MossTTSDelayRequestState] = []

        offset = 0
        for request_id, seq_len in zip(request_ids, seq_lens):
            req_tokens = input_ids[offset : offset + seq_len].reshape(-1)
            state = self._request_states.get(request_id)

            if seq_len > 1 or state is None:
                state = self._reset_prefill_state(request_id, req_tokens)
            else:
                self._advance_state_with_text_token(state, int(req_tokens[-1].item()))

            if seq_len == 1:
                decode_positions.append(offset)
                decode_states.append(state)
            offset += seq_len

        self._last_request_ids = list(request_ids)
        self._last_seq_lens = list(seq_lens)
        return decode_positions, decode_states

    def _clear_warmup_state(self) -> None:
        self._request_states.clear()
        self._last_request_ids = []
        self._last_seq_lens = []

    def on_requests_finished(self, request_ids) -> None:
        finished = {str(request_id) for request_id in request_ids}
        for request_id in finished:
            self._request_states.pop(str(request_id), None)
        active = [
            (request_id, seq_len)
            for request_id, seq_len in zip(self._last_request_ids, self._last_seq_lens)
            if request_id not in finished
        ]
        self._last_request_ids = [request_id for request_id, _ in active]
        self._last_seq_lens = [seq_len for _, seq_len in active]

    # ------------------------------------------------------------------ #
    # Embedding                                                           #
    # ------------------------------------------------------------------ #

    def _get_text_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        get_input_embeddings = getattr(self.backbone, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            return get_input_embeddings()(input_ids)
        return self.backbone.embed_tokens(input_ids)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: bool = False,
        request_ids: list[str] | None = None,
        seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        embeds = self._get_text_embedding(input_ids)
        if multimodal_embeddings is not None:
            embeds = embeds + multimodal_embeddings

        if not request_ids or not seq_lens:
            return embeds

        decode_offsets: list[int] = []
        audio_rows: list[torch.Tensor] = []
        offset = 0
        for request_id, seq_len in zip(request_ids, seq_lens):
            if seq_len == 1:
                state = self._request_states.get(request_id)
                if state is not None:
                    decode_offsets.append(offset)
                    audio_rows.append(state.pending_audio_row)
            offset += seq_len

        if audio_rows:
            rows = torch.stack(
                [row.to(device=embeds.device, dtype=torch.long, non_blocking=True) for row in audio_rows],
                dim=0,
            )
            vq_positions = torch.arange(self.n_vq, device=embeds.device, dtype=torch.long).expand(
                rows.shape[0],
                -1,
            )
            audio_embeds = self.audio_embedding_weight[vq_positions, rows].sum(dim=1)
            decode_offsets_t = torch.tensor(decode_offsets, device=embeds.device, dtype=torch.long)
            embeds[decode_offsets_t] = embeds[decode_offsets_t] + audio_embeds
        return embeds

    # ------------------------------------------------------------------ #
    # Audio sampling                                                      #
    # ------------------------------------------------------------------ #

    def _audio_mask(self, state: MossTTSDelayRequestState, device: torch.device) -> torch.Tensor:
        positions = torch.arange(self.n_vq, device=device, dtype=torch.long)
        pre_audio = state.audio_length > positions
        if state.delayed_length == _UNSET_DELAY_LENGTH:
            post_audio = torch.ones_like(pre_audio, dtype=torch.bool)
        else:
            post_audio = positions > (state.delayed_length - 1)
        return pre_audio & post_audio

    def _audio_masks(
        self,
        states: list[MossTTSDelayRequestState],
        device: torch.device,
    ) -> torch.Tensor:
        if not states:
            return torch.empty((0, self.n_vq), dtype=torch.bool, device=device)

        positions = torch.arange(self.n_vq, device=device, dtype=torch.long).unsqueeze(0)
        audio_lengths = torch.tensor(
            [state.audio_length for state in states],
            device=device,
            dtype=torch.long,
        ).unsqueeze(1)
        delayed_lengths = torch.tensor(
            [state.delayed_length for state in states],
            device=device,
            dtype=torch.long,
        ).unsqueeze(1)

        pre_audio = audio_lengths > positions
        post_audio = (delayed_lengths == _UNSET_DELAY_LENGTH) | (positions > (delayed_lengths - 1))
        return pre_audio & post_audio

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        histories: list[set[int]] | None,
        penalty: float,
    ) -> torch.Tensor:
        if penalty <= 1.0 or not histories:
            return logits

        for row_idx, history in enumerate(histories):
            if not history:
                continue
            seen = torch.tensor(sorted(set(history)), device=logits.device, dtype=torch.long)
            seen_logits = logits[row_idx, seen]
            logits[row_idx, seen] = torch.where(
                seen_logits < 0,
                seen_logits * penalty,
                seen_logits / penalty,
            )
        return logits

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.shape[-1]:
            return logits
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p <= 0.0 or top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        remove_mask = torch.zeros_like(sorted_mask)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        return logits.masked_fill(remove_mask, float("-inf"))

    def _sample_audio_from_logits(
        self,
        logits: torch.Tensor,
        histories: list[set[int]] | None,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        logits = logits.clone()
        logits[..., self.audio_pad_code] = float("-inf")
        logits = self._apply_repetition_penalty(logits, histories, repetition_penalty)

        if temperature <= 0.0:
            return logits.argmax(dim=-1)

        logits = logits / temperature
        logits = self._apply_top_k(logits, top_k)
        logits = self._apply_top_p(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _sample_delay_audio_rows(
        self,
        decode_hidden: torch.Tensor,
        decode_states: list[MossTTSDelayRequestState],
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        batch_size = decode_hidden.shape[0]
        device = decode_hidden.device
        next_audio = torch.full(
            (batch_size, self.n_vq),
            self.audio_pad_code,
            dtype=torch.long,
            device=device,
        )

        if batch_size == 0:
            return next_audio

        audio_masks = self._audio_masks(decode_states, device)

        audio_logits = torch.einsum(
            "bh,qvh->bqv",
            decode_hidden,
            self.audio_lm_head_weight,
        )

        active_b, active_q = torch.nonzero(audio_masks, as_tuple=True)
        if active_b.numel() == 0:
            return next_audio

        histories: list[set[int]] | None = None
        if repetition_penalty > 1.0:
            active_b_list = active_b.tolist()
            active_q_list = active_q.tolist()
            histories = [
                decode_states[batch_idx].audio_history[ch_idx]
                for batch_idx, ch_idx in zip(active_b_list, active_q_list)
            ]

        sampled = self._sample_audio_from_logits(
            audio_logits[active_b, active_q, :],
            histories,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        next_audio[active_b, active_q] = sampled.to(torch.long)

        return next_audio

    # ------------------------------------------------------------------ #
    # Forward / logits / sample                                           #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        kv_caches: list | None = None,
        attn_metadata=None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        request_ids, seq_lens, decode_positions = self._extract_request_schedule(
            kwargs.get("runtime_additional_information"),
        )

        decode_states: list[MossTTSDelayRequestState] = []
        if request_ids and seq_lens and input_ids is not None:
            decode_positions, decode_states = self._prepare_request_states(
                input_ids=input_ids,
                request_ids=request_ids,
                seq_lens=seq_lens,
            )

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_input_ids(
                input_ids,
                multimodal_embeddings=kwargs.get("multimodal_embeddings"),
                request_ids=request_ids if request_ids else None,
                seq_lens=seq_lens if seq_lens else None,
            )

        hidden_states = self.backbone(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        multimodal_outputs: dict[str, torch.Tensor | int] = {}
        if decode_positions and decode_states and not torch.cuda.is_current_stream_capturing():
            decode_hidden = hidden_states[torch.tensor(decode_positions, device=hidden_states.device)]
            repetition_penalty = float(kwargs.get("audio_repetition_penalty", 1.0))
            audio_rows = self._sample_delay_audio_rows(
                decode_hidden,
                decode_states,
                temperature=float(kwargs.get("audio_temperature", 1.7)),
                top_k=int(kwargs.get("audio_top_k", 25)),
                top_p=float(kwargs.get("audio_top_p", 0.8)),
                repetition_penalty=repetition_penalty,
            )

            for state, row in zip(decode_states, audio_rows):
                state.store_next_audio_row(row, track_history=repetition_penalty > 1.0)

            multimodal_outputs = {
                "code_predictor_codes": audio_rows.reshape(audio_rows.shape[0], 1, self.n_vq, 1),
                "audio_pad_code": self.audio_pad_code,
            }

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        logits = self.text_lm_head(hidden_states)
        if not self._last_request_ids:
            return logits

        neg_inf = float("-inf")
        for row_idx, request_id in enumerate(self._last_request_ids):
            if row_idx >= logits.shape[0]:
                break
            state = self._request_states.get(request_id)
            if state is None:
                continue

            row = logits[row_idx]

            if state.delayed_length != _UNSET_DELAY_LENGTH and state.delayed_length < self.n_vq:
                keep = row[self.audio_assistant_delay_slot_token_id].clone()
                row.fill_(neg_inf)
                row[self.audio_assistant_delay_slot_token_id] = keep
                continue

            if state.delayed_length == self.n_vq:
                keep = row[self.audio_end_token_id].clone()
                row.fill_(neg_inf)
                row[self.audio_end_token_id] = keep
                continue

            if state.is_audio:
                gen_keep = row[self.audio_assistant_gen_slot_token_id].clone()
                delay_keep = row[self.audio_assistant_delay_slot_token_id].clone()
                row.fill_(neg_inf)
                row[self.audio_assistant_gen_slot_token_id] = gen_keep
                row[self.audio_assistant_delay_slot_token_id] = delay_keep
            else:
                row[self.pad_token_id] = neg_inf
                row[self.audio_assistant_gen_slot_token_id] = neg_inf
                row[self.audio_assistant_delay_slot_token_id] = neg_inf
                row[self.audio_end_token_id] = neg_inf

            if state.decode_steps_generated == 0:
                row[self.audio_assistant_delay_slot_token_id] = neg_inf
            if state.decode_steps_generated <= self.n_vq:
                row[self.im_end_token_id] = neg_inf

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)

    def make_omni_output(self, model_output, **kwargs) -> OmniOutput:
        if isinstance(model_output, OmniOutput):
            return model_output
        empty = torch.zeros((0,), dtype=torch.float32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": [empty]},
        )

    # ------------------------------------------------------------------ #
    # Weight loading                                                      #
    # ------------------------------------------------------------------ #

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        **kwargs,
    ) -> set[str]:
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        params: dict[str, torch.nn.Parameter] = dict(self.named_parameters())
        loaded: set[str] = set()

        backbone_stacked = [
            ("self_attn.q_proj", "self_attn.qkv_proj", "q"),
            ("self_attn.k_proj", "self_attn.qkv_proj", "k"),
            ("self_attn.v_proj", "self_attn.qkv_proj", "v"),
            ("mlp.gate_proj", "mlp.gate_up_proj", 0),
            ("mlp.up_proj", "mlp.gate_up_proj", 1),
        ]

        for ckpt_name, tensor in weights:
            mapped: str | None = None
            shard_id = None
            stacked_param: tuple[str, int] | None = None

            if ckpt_name.startswith("language_model.") or ckpt_name.startswith("model.language_model."):
                relative = ckpt_name.split("language_model.", maxsplit=1)[1]
                for ckpt_suffix, mod_suffix, shard in backbone_stacked:
                    if ckpt_suffix in relative:
                        mapped = "backbone." + relative.replace(ckpt_suffix, mod_suffix)
                        shard_id = shard
                        break
                if mapped is None:
                    mapped = "backbone." + relative
            elif ckpt_name.startswith("emb_ext.") or ckpt_name.startswith("model.emb_ext."):
                relative = ckpt_name.replace("model.", "", 1)
                parts = relative.split(".")
                if len(parts) == 3 and parts[0] == "emb_ext" and parts[2] == "weight":
                    stacked_param = ("audio_embedding_weight", int(parts[1]))
            elif ckpt_name.startswith("lm_heads.") or ckpt_name.startswith("model.lm_heads."):
                relative = ckpt_name.replace("model.", "", 1)
                parts = relative.split(".")
                if len(parts) == 3 and parts[0] == "lm_heads" and parts[2] == "weight":
                    head_idx = int(parts[1])
                    if head_idx == 0:
                        mapped = "text_lm_head.weight"
                    else:
                        stacked_param = ("audio_lm_head_weight", head_idx - 1)

            if stacked_param is not None:
                param_name, ch_idx = stacked_param
                param = params.get(param_name)
                if param is None or ch_idx < 0 or ch_idx >= param.shape[0]:
                    logger.debug(
                        "[MossTTS Delay] Unused stacked checkpoint key %s (mapped=%s[%d])",
                        ckpt_name,
                        param_name,
                        ch_idx,
                    )
                    continue
                if param.data[ch_idx].shape != tensor.shape:
                    logger.warning(
                        "[MossTTS Delay] Shape mismatch for %s: ckpt=%s model=%s",
                        ckpt_name,
                        tuple(tensor.shape),
                        tuple(param.data[ch_idx].shape),
                    )
                    continue
                param.data[ch_idx].copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded.add(param_name)
                continue

            if mapped is None or mapped not in params:
                logger.debug(
                    "[MossTTS Delay] Unused checkpoint key %s (mapped=%s)",
                    ckpt_name,
                    mapped,
                )
                continue

            param = params[mapped]
            if shard_id is not None:
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is None:
                    logger.warning(
                        "[MossTTS Delay] No weight_loader on %s for shard %s",
                        mapped,
                        shard_id,
                    )
                    continue
                weight_loader(param, tensor, shard_id)
                loaded.add(mapped)
                continue

            if param.data.shape != tensor.shape:
                logger.warning(
                    "[MossTTS Delay] Shape mismatch for %s: ckpt=%s model=%s",
                    ckpt_name,
                    tuple(tensor.shape),
                    tuple(param.data.shape),
                )
                continue

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(mapped)

        missing = set(params) - loaded
        if missing:
            logger.warning(
                "[MossTTS Delay] Parameters not loaded from checkpoint:\n%s",
                "\n".join(sorted(missing)[:20]),
            )
        return loaded
