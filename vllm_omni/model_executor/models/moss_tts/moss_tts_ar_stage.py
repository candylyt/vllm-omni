# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
#
# Stage 0: AR stage for MOSS-TTS-Local.
#
# Architecture (mirrors MossTTSDelayModel from modeling_moss_tts.py):
#
#   ┌───────────────────────────────────────────────────────────┐
#   │  Global: Qwen3-1.7B backbone (paged attn via vllm)       │
#   │    input_ids → multi-channel embedding → Qwen3Model       │
#   │    → hidden_states [L, D_global]                          │
#   │                           │                               │
#   │  Local (per decode step): │                               │
#   │    global_hidden[step]    │                               │
#   │        ↓ speech_embedding_to_local_mlp                    │
#   │    for ch in 0..32:                                       │
#   │        append to local_ctx  [B, ch, D_local]             │
#   │        local_transformer forward                          │
#   │        local_to_speech_mlp + norm + lm_head[ch]          │
#   │        sample next_token[ch]                              │
#   │        embed → next local input                           │
#   │    → code_predictor_codes [B, 1, n_vq=32, 1]             │
#   └───────────────────────────────────────────────────────────┘
#
# Weight key mapping (HF checkpoint → this module):
#   model.language_model.*            → backbone.*
#   model.embedding_list.{i}.weight   → embedding_list.{i}.weight
#   local_transformer.*               → local_transformer.*
#   speech_embedding_to_local_mlp.*   → speech_embedding_to_local_mlp.*
#   local_to_speech_embedding_mlps.*  → local_to_speech_embedding_mlps.*
#   layer_norm_before_lm_heads.*      → layer_norm_before_lm_heads.*
#   lm_heads.*                        → lm_heads.*

import copy
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
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
#  Lightweight modules cloned from MOSS-TTS (kept identical for weight compat)
# ═══════════════════════════════════════════════════════════════════════════════

class MossTTSRMSNorm(nn.Module):
    """Root-mean-square layer norm (identical to original)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MossTTSMLP(nn.Module):
    """SwiGLU feed-forward network (identical to original)."""

    def __init__(self, input_size: int, ffn_hidden_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.up_proj   = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MossTTSLocalTransformerWrapper(nn.Module):
    """
    Thin wrapper around MossTTSLocalTransformer from the HF repo.

    We instantiate it from a plain Qwen3Config (the local sub-config) so the
    weight shapes match the original checkpoint exactly.  The local transformer
    has NO positional embeddings and NO token embeddings — it only processes
    input_embeds of shape (B, t, local_hidden_size).
    """

    def __init__(self, local_qwen3_config, model_path: str | None = None):
        super().__init__()
        # Trust-remote-code must be enabled; the HF repo provides the class.
        # Try direct import first (works if moss_tts_local is pip-installed).
        # Fall back to transformers' dynamic module loader, which loads the
        # modeling file from the checkpoint directory (trust_remote_code path).
        try:
            from moss_tts_local.modeling_moss_tts import MossTTSLocalTransformer
        except ModuleNotFoundError:
            if model_path is None:
                raise
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            MossTTSLocalTransformer = get_class_from_dynamic_module(
                "modeling_moss_tts.MossTTSLocalTransformer",
                model_path,
                code_revision=None,
            )
        self.transformer = MossTTSLocalTransformer(local_qwen3_config)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        inputs_embeds : (B, t, local_hidden)   t grows from 1 to n_vq+1
        Returns       : (B, t, local_hidden)
        """
        out = self.transformer(input_ids=None, inputs_embeds=inputs_embeds)
        return out.last_hidden_state


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-request FSM state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MossTTSLocalRequestState:
    """Per-request FSM tracking whether the model is currently emitting audio.

    Mirrors `MossTTSDelayRequestState` from the delay stage but without the
    delay-pattern bookkeeping (no `delayed_length`, no per-channel mask) since
    the local architecture emits all n_vq codes together at every audio step.
    """

    n_vq: int
    audio_pad_code: int
    is_audio: bool = False
    audio_steps_generated: int = 0
    pending_audio_row: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        # Initialised to pad so it is a no-op on prefill / non-audio decode steps.
        self.pending_audio_row = torch.full(
            (self.n_vq,), self.audio_pad_code, dtype=torch.long
        )

    def store_next_audio_row(self, row: torch.Tensor) -> None:
        """Cache the [n_vq] codes just sampled, to be summed into the next decode embedding."""
        self.pending_audio_row = row.detach().to(torch.long).cpu().reshape(self.n_vq)
        self.audio_steps_generated += 1


# ═══════════════════════════════════════════════════════════════════════════════
#  AR Stage Model
# ═══════════════════════════════════════════════════════════════════════════════

class MossTTSARStageModel(nn.Module, SupportsPP):
    """
    Stage 0: autoregressive stage for MOSS-TTS-Local.

    Inference flow per decode step
    --------------------------------
    1. embed_input_ids()
         text_embed(input_ids)
         + Σ audio_embed[i](cached_codes[i])   for gen-slot positions
       → inputs_embeds [L, D_global]

    2. forward()
         vllm Qwen3 backbone (paged attention, KV cache)
       → hidden_states [L, D_global]

    3. _local_forward()  (decode steps only)
         for ch in 0 … 32:
           ctx ← [ctx | project(prev_embed)]
           local_transformer(ctx) → mlp → norm → lm_head[ch] → sample
       → code_predictor_codes [B, 1, n_vq, 1]

    4. compute_logits() + sample()
         Global text token sampled by vllm's standard sampler from lm_heads[0].
    """

    have_multimodal_outputs = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        self.config = cfg

        # ── Scalar constants ────────────────────────────────────────────
        self.n_vq: int            = cfg.n_vq                              # 32
        self.channels: int        = 1 + self.n_vq                        # 33
        self.audio_vocab_size: int = cfg.audio_vocab_size                 # 1024
        self.audio_pad_code: int  = cfg.audio_pad_code                   # 1024
        self.gen_slot_id: int     = cfg.audio_assistant_gen_slot_token_id # 151656
        self.audio_end_id: int    = cfg.audio_end_token_id               # 151653

        # Tokens needed by the FSM / logits gating.
        # All optional — if a config doesn't carry them we just skip the
        # corresponding gating branch (defensive against config drift).
        self.pad_token_id: int          = getattr(cfg, "pad_token_id", -1)
        self.im_end_token_id: int       = getattr(cfg, "im_end_token_id", -1)
        self.audio_start_token_id: int  = getattr(cfg, "audio_start_token_id", -1)
        self.audio_user_slot_token_id: int = getattr(
            cfg, "audio_user_slot_token_id", -1
        )
        logger.info(
            "[MossTTS Local] FSM token ids: pad=%d im_end=%d audio_start=%d "
            "gen_slot=%d audio_end=%d",
            self.pad_token_id, self.im_end_token_id,
            self.audio_start_token_id, self.gen_slot_id, self.audio_end_id,
        )

        lang_cfg = cfg.language_config          # Qwen3Config
        self.hidden_size: int = lang_cfg.hidden_size

        # ── Global Qwen3 backbone ────────────────────────────────────────
        # Substitute the embedded Qwen3Config into a copy of VllmConfig so
        # vllm's Qwen3 model initialises with the correct dimensions.
        # TODO: If VllmConfig becomes immutable, use vllm's public API instead.
        qwen3_vllm_config = copy.deepcopy(vllm_config)
        object.__setattr__(qwen3_vllm_config.model_config, "hf_config", lang_cfg)

        from vllm.model_executor.models.qwen3 import Qwen3Model
        backbone_prefix = f"{prefix}model.language_model" if prefix else "model.language_model"
        self.backbone = Qwen3Model(
            vllm_config=qwen3_vllm_config,
            prefix=backbone_prefix,
        )

        # ── Multi-channel embeddings ─────────────────────────────────────
        # [0]    : text token embedding  (vocab_size  × D_global)
        # [1..32]: audio code embedding  ((audio_vocab_size+1) × D_global)
        #          index audio_vocab_size (=1024) is the pad sentinel
        self.embedding_list = nn.ModuleList()
        self.embedding_list.append(
            nn.Embedding(
                lang_cfg.vocab_size,
                self.hidden_size,
                padding_idx=cfg.pad_token_id,
            )
        )
        for _ in range(self.n_vq):
            self.embedding_list.append(
                nn.Embedding(
                    self.audio_vocab_size + 1,
                    self.hidden_size,
                    padding_idx=self.audio_pad_code,
                )
            )

        # ── Local transformer ────────────────────────────────────────────
        # Build the local Qwen3 sub-config (4 blocks, 1536 hidden).
        #
        # MOSS-TTS's modeling_moss_tts.py builds local_transformer_config by
        # deepcopying language_config and ONLY overriding these three fields.
        # num_attention_heads (16) is intentionally inherited unchanged, giving:
        #   head_dim = local_hidden_size / num_attention_heads = 1536 / 16 = 96
        #
        # Verified from checkpoint:
        #   config.local_hidden_size              = 1536
        #   config.language_config.num_attention_heads = 16
        #   → head_dim                            = 96
        local_cfg = copy.deepcopy(lang_cfg)
        local_cfg.num_hidden_layers = cfg.local_num_layers       # 4
        local_cfg.hidden_size       = cfg.local_hidden_size      # 1536
        local_cfg.intermediate_size = cfg.local_ffn_hidden_size  # 8960
        # num_attention_heads = 16 (inherited from lang_cfg — do NOT override)
        # num_key_value_heads inherited as-is from lang_cfg (respects GQA if any)
        self.local_transformer = MossTTSLocalTransformerWrapper(
            local_cfg, model_path=vllm_config.model_config.model
        )

        # ── Projection: global hidden → local hidden ─────────────────────
        # Used BOTH to project global backbone output and to re-project audio
        # embeddings before feeding them into the local transformer context.
        self.speech_embedding_to_local_mlp = MossTTSMLP(
            input_size=self.hidden_size,
            ffn_hidden_size=cfg.additional_mlp_ffn_hidden_size,  # 2048
            output_size=cfg.local_hidden_size,                   # 1536
        )

        # ── Per-channel: local hidden → global hidden (for LM head) ──────
        self.local_to_speech_embedding_mlps = nn.ModuleList([
            MossTTSMLP(
                input_size=cfg.local_hidden_size,
                ffn_hidden_size=cfg.additional_mlp_ffn_hidden_size,
                output_size=self.hidden_size,
            )
            for _ in range(self.channels)
        ])

        # ── Per-channel layer norms ───────────────────────────────────────
        self.layer_norm_before_lm_heads = nn.ModuleList([
            MossTTSRMSNorm(self.hidden_size) for _ in range(self.channels)
        ])

        # ── Per-channel LM heads ──────────────────────────────────────────
        # lm_heads[0]   : text  → vocab_size
        # lm_heads[1:] : audio → audio_vocab_size + 1
        self.lm_heads = nn.ModuleList()
        self.lm_heads.append(nn.Linear(self.hidden_size, lang_cfg.vocab_size, bias=False))
        for _ in range(self.n_vq):
            self.lm_heads.append(
                nn.Linear(self.hidden_size, self.audio_vocab_size + 1, bias=False)
            )

        # ── vllm sampler for text channel ────────────────────────────────
        self.logits_processor = LogitsProcessor(lang_cfg.vocab_size)
        self.sampler = Sampler()

        # ── Per-request FSM state ─────────────────────────────────────────
        # Replaces the old single-slot `_last_codes_slot` cache. Each request
        # carries its own `MossTTSLocalRequestState` that knows whether the
        # decoder is currently in audio mode and what audio codes were emitted
        # on the previous step (for next-step embedding contribution).
        self._request_states: dict[str, MossTTSLocalRequestState] = {}
        self._last_request_ids: list[str] = []
        self._last_seq_lens: list[int] = []

    # ══════════════════════════════════════════════════════════════════
    #  FSM helpers (per-request audio-mode tracking)
    # ══════════════════════════════════════════════════════════════════

    def _new_request_state(self) -> MossTTSLocalRequestState:
        return MossTTSLocalRequestState(
            n_vq=self.n_vq,
            audio_pad_code=self.audio_pad_code,
        )

    def _advance_state_with_text_token(
        self,
        state: MossTTSLocalRequestState,
        token_id: int,
    ) -> None:
        """Advance the per-request FSM by one observed text-channel token.

        Entry / exit rules:
          - Not in audio:
              * `audio_start` / `gen_slot`  → enter audio mode
              * everything else             → stay outside
          - In audio:
              * `gen_slot`                  → stay in audio mode
              * anything else (including
                `audio_end` and stray vocab)→ leave audio mode

        The "leave on any non-gen_slot token" rule is the key fix for the
        trailing-garbage symptom: the original code had no FSM at all and
        kept generating audio codes for whatever token the text head emitted
        once the model wanted to stop, producing garbage at the tail.
        """
        if state.is_audio:
            # Clean exit on the dedicated audio-end token.
            if token_id == self.audio_end_id:
                state.is_audio = False
                state.pending_audio_row = torch.full(
                    (self.n_vq,), self.audio_pad_code, dtype=torch.long
                )
                return
            # Stray non-gen_slot token mid-audio: only treat as termination
            # AFTER we've actually generated something. The very first decode
            # step after `<|audio_start|>` may not be exactly `gen_slot` in
            # the unconstrained sampling distribution; we don't want to kill
            # audio generation before it starts.
            if token_id != self.gen_slot_id and state.audio_steps_generated >= 1:
                state.is_audio = False
                state.pending_audio_row = torch.full(
                    (self.n_vq,), self.audio_pad_code, dtype=torch.long
                )
            return

        entry_tokens = {self.gen_slot_id}
        if self.audio_start_token_id >= 0:
            entry_tokens.add(self.audio_start_token_id)
        if token_id in entry_tokens:
            state.is_audio = True

    def _reset_prefill_state(
        self,
        request_id: str,
        prompt_tokens: torch.Tensor,
    ) -> MossTTSLocalRequestState:
        """Build a fresh FSM state for a request and replay its prompt tokens."""
        state = self._new_request_state()
        tokens = prompt_tokens.reshape(-1).tolist()

        if self.audio_user_slot_token_id >= 0:
            if any(int(t) == self.audio_user_slot_token_id for t in tokens):
                logger.warning(
                    "[MossTTS Local] Request %s contains continuation prompt tokens. "
                    "Phase-1 only validates direct TTS prompts.",
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
    ) -> tuple[list[int], list[MossTTSLocalRequestState]]:
        """Walk the scheduled batch, prefill-or-decode each request, and return
        the flat decode positions + their FSM states (in the same order)."""
        decode_positions: list[int] = []
        decode_states: list[MossTTSLocalRequestState] = []

        offset = 0
        for request_id, seq_len in zip(request_ids, seq_lens):
            req_tokens = input_ids[offset : offset + seq_len].reshape(-1)
            state = self._request_states.get(request_id)

            if seq_len > 1 or state is None:
                # Prefill (or first time we've seen this request) — rebuild FSM.
                state = self._reset_prefill_state(request_id, req_tokens)
            else:
                # Decode step — advance FSM by the single newly-sampled token.
                self._advance_state_with_text_token(state, int(req_tokens[-1].item()))

            if seq_len == 1:
                decode_positions.append(offset)
                decode_states.append(state)
            offset += seq_len

        self._last_request_ids = list(request_ids)
        self._last_seq_lens = list(seq_lens)
        return decode_positions, decode_states

    # ══════════════════════════════════════════════════════════════════
    #  Embedding
    # ══════════════════════════════════════════════════════════════════

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,             # [L] flat 1-D token IDs (text channel)
        multimodal_embeddings=None,
        is_multimodal: bool = False,
        request_ids: Optional[list[str]] = None,
        seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Build multi-channel embeddings.

        For each decode-step position that is *currently in audio mode*
        (per per-request FSM), the previously-emitted [n_vq] audio codes are
        embedded and summed into the text-channel embedding. Decode steps
        outside audio mode get only the text embedding (no audio contamination).
        Prefill positions also get only the text embedding.
        """
        # Channel 0: text embedding
        embeds = self.embedding_list[0](input_ids)  # [L, D]

        # Add pre-computed audio embeddings if provided by the pipeline
        if multimodal_embeddings is not None:
            embeds = embeds + multimodal_embeddings

        if not request_ids or not seq_lens:
            return embeds

        offset = 0
        for request_id, slen in zip(request_ids, seq_lens):
            if slen == 1:
                state = self._request_states.get(request_id)
                # Only inject audio embeddings if the request is currently
                # producing audio. Otherwise the cached pad row is a no-op,
                # but skipping the loop entirely is cheaper.
                if state is not None and state.is_audio:
                    row = state.pending_audio_row.to(embeds.device)
                    for ch_idx in range(self.n_vq):
                        ch_code = row[ch_idx].unsqueeze(0)
                        ch_emb  = self.embedding_list[ch_idx + 1](ch_code)
                        embeds[offset] = embeds[offset] + ch_emb[0]
            offset += slen

        return embeds

    # ══════════════════════════════════════════════════════════════════
    #  Local transformer: predict n_vq RVQ codes from one global step
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _local_forward(
        self,
        global_hidden: torch.Tensor,   # [B, D_global]  (B = num decode seqs)
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Autoregressively predict (1 text token + 32 audio codes) per batch element.

        Implements the inner loop of MossTTSDelayModel._sample():
            - ch 0  → text token  (text LM head, not used by Stage 1)
            - ch 1..32 → audio codes (audio LM heads, sent to Stage 1)

        Returns
        -------
        audio_codes : Tensor [B, n_vq]  (long)
            The 32 predicted RVQ codes for each sequence in the batch.
        """
        B     = global_hidden.shape[0]
        dev   = global_hidden.device
        dtype = global_hidden.dtype
        local_dim = self.config.local_hidden_size

        # Project global hidden state → first local transformer input
        current_proj = self.speech_embedding_to_local_mlp(global_hidden)  # [B, local_D]
        local_ctx    = torch.zeros(B, 0, local_dim, device=dev, dtype=dtype)

        audio_codes: list[torch.Tensor] = []

        for ch in range(self.channels):   # ch = 0 (text), 1..32 (audio)
            # Grow local context by one token
            local_ctx = torch.cat(
                [local_ctx, current_proj.unsqueeze(1)], dim=1
            )  # [B, ch+1, local_D]

            # Local transformer forward (no positional embeddings, no KV cache)
            local_out = self.local_transformer(local_ctx)   # [B, ch+1, local_D]
            last_h    = local_out[:, -1, :]                 # [B, local_D]

            # Per-channel projection + norm + LM head → logits
            proj_out = self.local_to_speech_embedding_mlps[ch](last_h)  # [B, D_global]
            normed   = self.layer_norm_before_lm_heads[ch](proj_out)     # [B, D_global]
            logits   = self.lm_heads[ch](normed)                         # [B, V]

            # Prevent the model from sampling the pad code in audio channels
            if ch > 0:
                logits[:, self.audio_pad_code] = float("-inf")

            # Sampling (temperature / top-k / top-p)
            # TODO: wire in per-request sampling params from SamplingMetadata
            if temperature > 0.0 and ch > 0:
                logits = logits / temperature
                if top_k > 0:
                    top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values
                    logits[logits < top_k_vals[..., -1:]] = float("-inf")
                probs      = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            else:
                next_token = logits.argmax(dim=-1)  # [B]

            if ch == 0:
                # Text token — skip from audio output, but embed for next local step
                pass
            else:
                audio_codes.append(next_token)

            # Re-embed sampled token → next local step's input
            emb          = self.embedding_list[ch](next_token)             # [B, D_global]
            current_proj = self.speech_embedding_to_local_mlp(emb)        # [B, local_D]

        # Stack: [B, n_vq]
        if audio_codes:
            return torch.stack(audio_codes, dim=1).to(torch.long)
        return torch.zeros(B, self.n_vq, dtype=torch.long, device=dev)

    # ══════════════════════════════════════════════════════════════════
    #  Request-info helpers (vLLM v0.19+)
    # ══════════════════════════════════════════════════════════════════

    def _extract_request_info(
        self,
        runtime_additional_information: Optional[list[dict]] = None,
    ) -> tuple[list[str], list[int], list[int]]:
        """
        Derive per-request schedule info from the vLLM forward context (v0.19+).

        Returns
        -------
        request_ids      : list[str] — req_id for each scheduled request
        seq_lens         : list[int] — #tokens scheduled per request
        decode_positions : list[int] — flat positions of decode-phase requests
        """
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
            num_reqs = len(qsl) - 1
            seq_lens = [qsl[i + 1] - qsl[i] for i in range(num_reqs)]
            decode_positions = [qsl[i] for i, s in enumerate(seq_lens) if s == 1]
        except Exception as exc:
            logger.debug("[MossTTS AR] _extract_request_info failed: %s", exc)
            return [], [], []

        if runtime_additional_information:
            request_ids = [
                info.get("req_id", str(i))
                for i, info in enumerate(runtime_additional_information)
            ]
        else:
            request_ids = [str(i) for i in range(len(seq_lens))]
        return request_ids, seq_lens, decode_positions

    def _clear_warmup_state(self) -> None:
        """Clear any state accumulated during the vLLM profiling / warmup pass."""
        self._request_states.clear()
        self._last_request_ids = []
        self._last_seq_lens = []

    # ══════════════════════════════════════════════════════════════════
    #  Forward
    # ══════════════════════════════════════════════════════════════════

    def forward(
        self,
        input_ids: Optional[torch.Tensor]              = None,
        positions: Optional[torch.Tensor]              = None,
        kv_caches: Optional[list]                      = None,
        attn_metadata                                  = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor]          = None,
        **kwargs,
    ) -> OmniOutput:
        # ── 0. Derive per-request schedule info from vLLM forward context ─
        request_ids, seq_lens_per_req, decode_positions = self._extract_request_info(
            kwargs.get("runtime_additional_information"),
        )

        # ── 1. Per-request FSM bookkeeping ────────────────────────────
        decode_states: list[MossTTSLocalRequestState] = []
        if request_ids and seq_lens_per_req and input_ids is not None:
            decode_positions, decode_states = self._prepare_request_states(
                input_ids=input_ids,
                request_ids=request_ids,
                seq_lens=seq_lens_per_req,
            )

        # ── 2. Build embeddings (multi-channel) ───────────────────────
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_input_ids(
                input_ids,
                multimodal_embeddings=kwargs.get("multimodal_embeddings"),
                request_ids=request_ids if request_ids else None,
                seq_lens=seq_lens_per_req if seq_lens_per_req else None,
            )

        # ── 3. Global Qwen3 backbone ──────────────────────────────────
        hidden_states = self.backbone(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )  # [L, D_global]

        # ── 4. Local transformer — only for decode steps in audio mode ─
        # Outside of audio mode the local transformer's output is garbage
        # for our purposes (downstream Stage 1 would treat it as audio codes).
        # Restrict execution to FSM positions that are currently `is_audio`.
        multimodal_outputs: dict[str, Any] = {}

        if decode_positions and decode_states and not torch.cuda.is_current_stream_capturing():
            audio_mask = [s.is_audio for s in decode_states]
            # Debug: log FSM state + the input token that drove the transition.
            # Remove once the pipeline is verified.
            for p, s in zip(decode_positions, decode_states):
                tok = int(input_ids[p].item()) if input_ids is not None else -1
                logger.warning(
                    "[debug-fsm] tok=%d is_audio=%s steps=%d "
                    "(gen=%d end=%d start=%d)",
                    tok, s.is_audio, s.audio_steps_generated,
                    self.gen_slot_id, self.audio_end_id, self.audio_start_token_id,
                )
            audio_positions = [
                p for p, m in zip(decode_positions, audio_mask) if m
            ]
            audio_states = [s for s, m in zip(decode_states, audio_mask) if m]

            if audio_positions:
                pos_t = torch.tensor(audio_positions, device=hidden_states.device)
                decode_hidden = hidden_states[pos_t]  # [B_audio, D_global]

                codes = self._local_forward(
                    decode_hidden,
                    temperature=kwargs.get("audio_temperature", 1.0),
                    top_k=kwargs.get("audio_top_k", 50),
                    top_p=kwargs.get("audio_top_p", 0.95),
                )  # [B_audio, n_vq]

                # Cache codes per-request for the NEXT decode step's
                # multi-channel embedding contribution.
                for state, row in zip(audio_states, codes):
                    state.store_next_audio_row(row)

                # Shape convention: [B_audio, 1, n_vq, 1]
                B = codes.shape[0]
                multimodal_outputs = {
                    "code_predictor_codes": codes.reshape(B, 1, self.n_vq, 1),
                    "audio_pad_code": self.audio_pad_code,
                }

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    # ══════════════════════════════════════════════════════════════════
    #  vllm model protocol
    # ══════════════════════════════════════════════════════════════════

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        """Text channel logits (channel-0 LM head), with light FSM-driven gating.

        Strategy:
          - In audio mode → DO NOT mask. The model was trained to emit
            `gen_slot` here naturally; over-restricting (e.g. keeping only
            gen_slot/audio_end) skews the relative ranking and can cause
            instant termination on the very first sampling step.
          - Outside audio mode → mask the audio control tokens so they
            cannot leak into text generation.

        The "garbage at end of audio" symptom is fixed by the FSM exiting
        audio mode on any non-`gen_slot` token (see
        `_advance_state_with_text_token`) and by `forward()` skipping
        `_local_forward` for non-audio decode steps.
        """
        logits = self.lm_heads[0](hidden_states)   # [L, vocab_size]
        if not self._last_request_ids:
            return logits

        neg_inf = float("-inf")
        for row_idx, request_id in enumerate(self._last_request_ids):
            if row_idx >= logits.shape[0]:
                break
            state = self._request_states.get(request_id)
            if state is None or state.is_audio:
                # Audio mode: leave the distribution alone.
                continue

            row = logits[row_idx]
            if self.pad_token_id >= 0:
                row[self.pad_token_id] = neg_inf
            row[self.gen_slot_id]  = neg_inf
            row[self.audio_end_id] = neg_inf

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)

    def make_omni_output(self, model_output: Any, **kwargs) -> OmniOutput:
        if isinstance(model_output, OmniOutput):
            return model_output
        empty = torch.zeros((0,), dtype=torch.float32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": [empty]},
        )

    # ══════════════════════════════════════════════════════════════════
    #  Weight loading
    # ══════════════════════════════════════════════════════════════════

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        **kwargs,
    ) -> set[str]:
        """
        Load from a MOSS-TTS-Local HuggingFace checkpoint.

        Returns a set of THIS MODULE's parameter names (as in self.named_parameters())
        that were successfully initialised.  The caller (MossTTSForConditionalGeneration)
        prepends "_model." so that vLLM's default_loader can verify coverage.

        Checkpoint key → this module's parameter name:
            model.language_model.*           → backbone.*
            model.embedding_list.{i}.weight  → embedding_list.{i}.weight
            local_transformer.*              → local_transformer.transformer.*
            speech_embedding_to_local_mlp.*  → speech_embedding_to_local_mlp.*
            local_to_speech_embedding_mlps.* → local_to_speech_embedding_mlps.*
            layer_norm_before_lm_heads.*     → layer_norm_before_lm_heads.*
            lm_heads.*                       → lm_heads.*

        For the Qwen3 backbone, vLLM merges Q/K/V into qkv_proj and
        gate/up into gate_up_proj.  We use each param's weight_loader
        (registered by MergedColumnParallelLinear / QKVParallelLinear)
        to handle the merging correctly.
        """
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        params: dict[str, torch.nn.Parameter] = dict(self.named_parameters())
        loaded_module_names: set[str] = set()

        # Qwen3 backbone: checkpoint has separate q/k/v and gate/up projections;
        # vLLM merges them.  Map (checkpoint suffix → model suffix, shard_id).
        BACKBONE_STACKED = [
            ("self_attn.q_proj", "self_attn.qkv_proj", "q"),
            ("self_attn.k_proj", "self_attn.qkv_proj", "k"),
            ("self_attn.v_proj", "self_attn.qkv_proj", "v"),
            ("mlp.gate_proj",    "mlp.gate_up_proj",   0),
            ("mlp.up_proj",      "mlp.gate_up_proj",   1),
        ]

        for ckpt_name, tensor in weights:
            mapped: str | None = None
            shard_id = None

            if ckpt_name.startswith("model.language_model."):
                relative = ckpt_name[len("model.language_model."):]
                # Check stacked (merged) params first
                for ckpt_sfx, mod_sfx, s_id in BACKBONE_STACKED:
                    if ckpt_sfx in relative:
                        mapped = "backbone." + relative.replace(ckpt_sfx, mod_sfx)
                        shard_id = s_id
                        break
                if mapped is None:
                    mapped = "backbone." + relative

            elif ckpt_name.startswith("model.embedding_list."):
                mapped = "embedding_list." + ckpt_name[len("model.embedding_list."):]

            elif ckpt_name.startswith("local_transformer."):
                mapped = "local_transformer.transformer." + ckpt_name[len("local_transformer."):]

            else:
                # speech_embedding_to_local_mlp.*, local_to_speech_embedding_mlps.*,
                # layer_norm_before_lm_heads.*, lm_heads.*  — no prefix change
                mapped = ckpt_name

            if mapped not in params:
                logger.debug(
                    "[MossTTS AR] Unused checkpoint key %s (mapped to %s)",
                    ckpt_name, mapped,
                )
                continue

            param = params[mapped]

            if shard_id is not None:
                # Use the registered weight_loader for proper shard merging
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, tensor, shard_id)
                    loaded_module_names.add(mapped)
                else:
                    logger.warning(
                        "[MossTTS AR] No weight_loader on %s, cannot merge shard %s — skipping",
                        mapped, shard_id,
                    )
            else:
                if param.data.shape != tensor.shape:
                    logger.warning(
                        "[MossTTS AR] Shape mismatch for %s: ckpt %s vs model %s — skipping.",
                        ckpt_name, tuple(tensor.shape), tuple(param.data.shape),
                    )
                    continue
                wl = getattr(param, "weight_loader", default_weight_loader)
                wl(param, tensor)
                loaded_module_names.add(mapped)

        missing = set(params.keys()) - loaded_module_names
        if missing:
            logger.warning(
                "[MossTTS AR] Parameters not loaded from checkpoint:\n%s",
                "\n".join(sorted(missing)[:20]),
            )
        return loaded_module_names
