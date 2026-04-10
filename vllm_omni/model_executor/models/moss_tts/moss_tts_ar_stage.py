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

        # ── Per-request audio code caches ─────────────────────────────────
        # _last_codes[req_id] = Tensor [n_vq] of the most-recently generated codes.
        # Used to build the multi-channel embedding at the NEXT decode step.
        self._last_codes: dict[str, torch.Tensor] = {}

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

        For positions that are gen_slot tokens, the audio embeddings from the
        previous step are fetched from _last_codes and summed in.  This mirrors
        MosiTTSModel._prepare_multi_modal_inputs() but operates token-by-token
        as vllm feeds individual decode steps.

        For prefill (text input only), multi-channel audio sum is zero-padded
        because no audio codes have been generated yet.
        """
        # Channel 0: text embedding
        embeds = self.embedding_list[0](input_ids)  # [L, D]

        # Add pre-computed audio embeddings if provided by the pipeline
        if multimodal_embeddings is not None:
            embeds = embeds + multimodal_embeddings

        # Add audio embeddings for decode-step gen-slot positions
        # When request_ids and seq_lens are provided, we know which rows in the
        # [L] sequence correspond to which requests, and can look up _last_codes.
        if request_ids is not None and seq_lens is not None:
            offset = 0
            for req_id, slen in zip(request_ids, seq_lens):
                if req_id in self._last_codes and slen == 1:
                    # Single-token decode step: add audio channel embeddings
                    codes = self._last_codes[req_id]  # [n_vq]
                    for ch_idx in range(self.n_vq):
                        ch_code = codes[ch_idx].unsqueeze(0)             # [1]
                        ch_emb  = self.embedding_list[ch_idx + 1](ch_code)  # [1, D]
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
        # ── 1. Build embeddings (multi-channel) ───────────────────────
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(
                input_ids,
                multimodal_embeddings=kwargs.get("multimodal_embeddings"),
                request_ids=kwargs.get("request_ids"),
                seq_lens=kwargs.get("seq_lens"),
            )

        # ── 2. Global Qwen3 backbone ──────────────────────────────────
        hidden_states = self.backbone(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )  # [L, D_global]

        # ── 3. Local transformer (decode positions only) ──────────────
        # num_decode_tokens > 0 means we are in the decode phase.
        # During prefill (text input), we skip local forward entirely.
        code_tensor: Optional[torch.Tensor] = None

        num_decode = getattr(attn_metadata, "num_decode_tokens", 0)
        if num_decode > 0 and not torch.cuda.is_current_stream_capturing():
            # The last `num_decode` rows of hidden_states correspond to decode seqs.
            decode_hidden = hidden_states[-num_decode:]  # [B, D_global]

            codes = self._local_forward(
                decode_hidden,
                temperature=kwargs.get("audio_temperature", 1.0),
                top_k=kwargs.get("audio_top_k", 50),
                top_p=kwargs.get("audio_top_p", 0.95),
            )  # [B, n_vq]

            # Cache codes for the NEXT step's multi-channel embedding.
            request_ids = kwargs.get("request_ids", [])
            for i, req_id in enumerate(request_ids[-num_decode:]):
                if req_id:
                    self._last_codes[req_id] = codes[i].cpu()

            # Shape convention: [B, 1, n_vq, 1]  (matches MiMo's [B, 1, 8, 4])
            B = codes.shape[0]
            code_tensor = codes.reshape(B, 1, self.n_vq, 1)

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=(
                {"code_predictor_codes": code_tensor}
                if code_tensor is not None
                else {}
            ),
        )

    # ══════════════════════════════════════════════════════════════════
    #  vllm model protocol
    # ══════════════════════════════════════════════════════════════════

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Text channel logits (channel-0 LM head)."""
        return self.lm_heads[0](hidden_states)   # [L, vocab_size]

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
