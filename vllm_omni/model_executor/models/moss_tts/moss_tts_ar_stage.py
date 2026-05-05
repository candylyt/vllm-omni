# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
"""
Stage 0 of MOSS-TTS-Local: text → 32-channel RVQ codes.

This module documents Stage 0 in isolation: how the
global Qwen3 backbone and the local transformer cooperate to produce one
audio frame per decode step.

Per-step architecture
---------------------

    ┌───────────────────────────────────────────────────────────┐
    │  Global: Qwen3-1.7B backbone (paged attn via vllm)        │
    │    input_ids → multi-channel embedding → Qwen3Model       │
    │    → hidden_states [L, D_global]                          │
    │                           │                               │
    │  Local (per decode step): │                               │
    │    global_hidden[step]    │                               │
    │        ↓ speech_embedding_to_local_mlp                    │
    │    for ch in 0..32:                                       │
    │        append to local_ctx  [B, ch, D_local]              │
    │        local_transformer forward                          │
    │        local_to_speech_mlp + norm + lm_head[ch]           │
    │        sample next_token[ch]                              │
    │        embed → next local input                           │
    │    → code_predictor_codes [B, 1, n_vq=32, 1]              │
    └───────────────────────────────────────────────────────────┘

The mode is gated by a per-request FSM (see `MossTTSLocalRequestState` and
`_advance_state_with_text_token`).  Outside of audio mode the global backbone
runs as a normal text LM; inside audio mode the local transformer is invoked
to predict 32 RVQ codes per step, and the text head's logits are masked so
vLLM's sampler can only pick `gen_slot` (continue audio) or `audio_end`
(terminate).  The single output every consumer downstream cares about is
`code_predictor_codes` of shape [B, 1, n_vq, 1].

Weight key mapping (HF checkpoint → this module):
    model.language_model.*            → backbone.*
    model.embedding_list.{i}.weight   → embedding_list.{i}.weight
    local_transformer.*               → local_transformer.*
    speech_embedding_to_local_mlp.*   → speech_embedding_to_local_mlp.*
    local_to_speech_embedding_mlps.*  → local_to_speech_embedding_mlps.*
    layer_norm_before_lm_heads.*      → layer_norm_before_lm_heads.*
    lm_heads.*                        → lm_heads.*

The mapping is implemented in `MossTTSARStageModel.load_weights`.
"""

import copy
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.moss_tts._stage0_timing import get_timer
from vllm_omni.model_executor.models.output_templates import OmniOutput

# Per-process Stage-0 timer.  No-op unless MOSS_TTS_TIMING=1 is set in env.
# See _stage0_timing.py for the helper API and report format.
_TIMER = get_timer()

logger = logging.getLogger(__name__)


# Sentinel for "do not pass this kwarg at all" (distinct from None which is
# itself a valid value for past_key_values).  Used by the KV-cache probe.
_OMIT: Any = object()


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

    KV-cache support
    ----------------
    Forward accepts an optional ``past_key_values`` argument and an optional
    ``use_cache`` flag.  When either is set, the wrapper threads them through
    to the underlying HF model and returns ``(hidden_states, past_key_values)``
    so callers can reuse the K/V across successive single-token forwards.
    When neither is set, the wrapper preserves the original behaviour and
    returns ``(hidden_states, None)`` — call sites that don't care about the
    cache can simply ignore the second element.

    The HF MossTTSLocalTransformer is loaded via trust_remote_code, so we
    detect whether its forward signature supports the cache kwargs at init
    time and store the result on ``self.supports_kv_cache``.  Older variants
    that lack the kwargs fall back to the recompute path automatically.
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

        # One-time capability probe: which kwargs does the underlying
        # transformer's forward actually accept?
        #
        #   past_key_values + use_cache  → required for the incremental path
        #   position_ids                 → standard HF Qwen3 kwarg, lets us
        #                                  override the auto-position guess
        #                                  (critical when sending single
        #                                  tokens with a populated cache)
        #   cache_position               → newer HF kwarg; some Qwen3-based
        #                                  models use it to derive RoPE
        import inspect
        try:
            sig_params = inspect.signature(self.transformer.forward).parameters
            self._has_past_kv: bool = "past_key_values" in sig_params
            self._has_use_cache: bool = "use_cache" in sig_params
            self.supports_kv_cache: bool = self._has_past_kv and self._has_use_cache
            self._has_position_ids: bool = "position_ids" in sig_params
            self._has_cache_position: bool = "cache_position" in sig_params
        except (TypeError, ValueError):
            self.supports_kv_cache = False
            self._has_position_ids = False
            self._has_cache_position = False

        # Env-var opt-in.  KV cache is OFF by default after we discovered
        # that the custom MossTTSLocalTransformer's forward produces
        # different outputs when use_cache=True even on the first call —
        # the kwargs are accepted but the semantics diverge from the plain
        # path.  Set MOSS_TTS_LOCAL_KV_CACHE=1 to enable, but only after
        # the init-time probe (below) confirms equivalence.
        import os
        kv_env = os.environ.get("MOSS_TTS_LOCAL_KV_CACHE", "0")  # default OFF
        kv_enabled_by_env = kv_env == "1"
        if not kv_enabled_by_env and self.supports_kv_cache:
            logger.info(
                "[MossTTS Local] KV cache available but DISABLED by default — "
                "set MOSS_TTS_LOCAL_KV_CACHE=1 to enable.  Run the init-time "
                "probe (logged below) to verify equivalence first."
            )
            self.supports_kv_cache = False
        elif kv_enabled_by_env and not self.supports_kv_cache:
            logger.warning(
                "[MossTTS Local] MOSS_TTS_LOCAL_KV_CACHE=1 but the model's "
                "forward does not accept past_key_values/use_cache — staying "
                "on legacy recompute path."
            )

        # Verify mode: when MOSS_TTS_LOCAL_KV_CACHE_VERIFY=1, _local_forward
        # runs both paths on the same input and warns if outputs diverge
        # beyond a tight tolerance.  Slow — for debugging only.
        self._verify_kv_cache: bool = (
            os.environ.get("MOSS_TTS_LOCAL_KV_CACHE_VERIFY", "0") == "1"
            and self.supports_kv_cache
        )

        if self.supports_kv_cache:
            logger.info(
                "[MossTTS Local] KV cache enabled.  position_ids=%s, "
                "cache_position=%s, verify=%s",
                self._has_position_ids,
                self._has_cache_position,
                self._verify_kv_cache,
            )

        # Always run the equivalence probe at init — both to inform the user
        # whether enabling the cache would be safe, and to surface the issue
        # early if they have it enabled.
        self._probe_kv_cache_equivalence()

    @torch.no_grad()
    def _probe_kv_cache_equivalence(self) -> None:
        """Test several kwarg combinations against a plain forward call.

        Goal: find a set of kwargs where calling the local transformer with
        ``inputs_embeds=[B, 1, D]`` plus the kwargs returns the same hidden
        state as calling it with ``inputs_embeds=[B, 1, D]`` alone.  If at
        least one combination matches to within float tolerance, the KV
        cache path can be made equivalent to recompute and is safe to use.
        If every combination diverges, the model fundamentally produces
        different outputs under cache mode — the optimization can't be
        fixed at the wrapper level and would require changes inside the
        custom remote-code transformer itself.

        Runs once at engine init.  Cheap (~milliseconds on a tiny synthetic
        input) and always logged so the user has a definitive answer.
        """
        if not self._has_past_kv and not self._has_use_cache:
            return  # Model can't accept any cache kwargs — nothing to probe

        try:
            params = list(self.transformer.parameters())
            if not params:
                logger.info("[MossTTS Local] KV-cache probe skipped (no params).")
                return
            device = params[0].device
            dtype = params[0].dtype

            # Hidden size from the local config.
            D = getattr(
                self.transformer.config, "hidden_size",
                getattr(self.transformer.config, "local_hidden_size", None),
            )
            if D is None:
                logger.info("[MossTTS Local] KV-cache probe skipped (hidden_size unknown).")
                return

            B, t = 1, 1
            torch.manual_seed(0)
            test_input = torch.randn(B, t, D, device=device, dtype=dtype)

            # Reference: plain forward, no cache kwargs at all.
            ref = self.transformer(
                input_ids=None, inputs_embeds=test_input
            ).last_hidden_state

            # Try to import DynamicCache — modern HF transformers expose this.
            # Some models distinguish between past_key_values=None (no cache,
            # build fresh internally) and an explicit empty DynamicCache().
            try:
                from transformers.cache_utils import DynamicCache
                _DynamicCache: Any = DynamicCache
            except ImportError:
                _DynamicCache = None

            # Helper to build a kwargs dict given which extras to include.
            def build(use_cache: bool, past_kv: Any,
                      with_position_ids: bool, with_cache_position: bool) -> dict:
                kw: dict[str, Any] = {}
                if past_kv is not _OMIT and self._has_past_kv:
                    kw["past_key_values"] = past_kv
                if use_cache and self._has_use_cache:
                    kw["use_cache"] = True
                if with_position_ids and self._has_position_ids:
                    kw["position_ids"] = torch.zeros(B, t, device=device, dtype=torch.long)
                if with_cache_position and self._has_cache_position:
                    kw["cache_position"] = torch.zeros(t, device=device, dtype=torch.long)
                return kw

            # Probe matrix: each entry is (label, kwargs).  The first one
            # exactly matches the no-cache reference (so should always be
            # zero).  Each subsequent variant adds one cache-related kwarg.
            variants: list[tuple[str, dict]] = [
                ("baseline (no cache args)", {}),
                ("use_cache only",
                    build(use_cache=True, past_kv=_OMIT,
                          with_position_ids=False, with_cache_position=False)),
                ("past_key_values=None only",
                    build(use_cache=False, past_kv=None,
                          with_position_ids=False, with_cache_position=False)),
                ("past_key_values=None + use_cache",
                    build(use_cache=True, past_kv=None,
                          with_position_ids=False, with_cache_position=False)),
                ("+ position_ids",
                    build(use_cache=True, past_kv=None,
                          with_position_ids=True, with_cache_position=False)),
                ("+ cache_position",
                    build(use_cache=True, past_kv=None,
                          with_position_ids=False, with_cache_position=True)),
                ("+ both positions",
                    build(use_cache=True, past_kv=None,
                          with_position_ids=True, with_cache_position=True)),
            ]
            if _DynamicCache is not None:
                variants.append((
                    "DynamicCache() + both positions",
                    build(use_cache=True, past_kv=_DynamicCache(),
                          with_position_ids=True, with_cache_position=True),
                ))

            logger.info(
                "[MossTTS Local] KV-cache equivalence probe "
                "(max_abs vs plain forward(inputs_embeds)):"
            )
            best_label = None
            best_max_abs = float("inf")
            # Re-fetch the reference INSIDE this function each variant — to
            # detect whether the previous probe call mutated model state.
            for label, kw in variants:
                try:
                    out = self.transformer(
                        input_ids=None, inputs_embeds=test_input, **kw
                    ).last_hidden_state
                    max_abs = (out - ref).abs().max().item()
                    logger.info(
                        "[MossTTS Local]   variant=%-32s max_abs=%.6e", label, max_abs
                    )
                    if max_abs < best_max_abs:
                        best_max_abs = max_abs
                        best_label = label
                except Exception as e:
                    logger.info(
                        "[MossTTS Local]   variant=%-32s FAILED: %s", label, e
                    )

            # State-leak check: re-run the *baseline* (plain) forward after
            # all the cache-mode variants.  If the model leaks state between
            # calls (e.g. caches a DynamicCache on itself), this re-run will
            # diverge from the original reference even though the call args
            # are identical.
            recheck = self.transformer(
                input_ids=None, inputs_embeds=test_input
            ).last_hidden_state
            recheck_diff = (recheck - ref).abs().max().item()
            logger.info(
                "[MossTTS Local]   recheck (plain forward, post-probe)  "
                "max_abs_vs_initial_ref=%.6e",
                recheck_diff,
            )
            if recheck_diff > 1e-3:
                logger.warning(
                    "[MossTTS Local] State leak detected: a plain forward "
                    "call after running cache-mode variants no longer matches "
                    "the initial baseline (max_abs=%.4e).  The custom local "
                    "transformer is mutating internal state across calls — "
                    "this is the cause of the runtime KV-cache divergence.",
                    recheck_diff,
                )

            verdict = (
                "PASS — KV cache can be made equivalent (safe to enable)"
                if best_max_abs < 1e-3 and recheck_diff < 1e-3
                else "FAIL — divergence detected; do NOT enable KV cache "
                     "without further investigation"
            )
            logger.info(
                "[MossTTS Local] KV-cache probe verdict: %s "
                "(best=%s, max_abs=%.6e, state_leak=%.6e)",
                verdict, best_label, best_max_abs, recheck_diff,
            )
        except Exception as e:
            logger.warning(
                "[MossTTS Local] KV-cache probe could not run: %s", e
            )

    @staticmethod
    def _cache_length(past_key_values: Any) -> int:
        """Return the number of cached tokens in a past_key_values object.

        Handles both the modern ``Cache`` interface (HF transformers ≥ 4.36)
        and the legacy tuple-of-tuples format ``((k0, v0), (k1, v1), …)``.
        Returns 0 when ``past_key_values`` is ``None`` or unparseable.
        """
        if past_key_values is None:
            return 0
        # Modern HF Cache object
        if hasattr(past_key_values, "get_seq_length"):
            try:
                return int(past_key_values.get_seq_length())
            except Exception:
                pass
        # Legacy tuple-of-tuples: ((k0, v0), (k1, v1), …) where each k/v has
        # shape [B, num_heads, seq_len, head_dim].
        try:
            first_layer = past_key_values[0]
            k = first_layer[0] if isinstance(first_layer, (tuple, list)) else first_layer
            return int(k.shape[-2])
        except (TypeError, IndexError, AttributeError):
            return 0

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        """
        Run one forward pass through the local transformer.

        Two calling conventions:

        * Recompute (legacy)
            ``forward(inputs_embeds=[B, t, D])`` with t ≥ 1.
            Returns ``(hidden_states[B, t, D], None)``.

        * Incremental (KV cache)
            ``forward(inputs_embeds=[B, 1, D], past_key_values=prev_kv,
                      use_cache=True)``.
            Returns ``(hidden_states[B, 1, D], new_past_key_values)`` where
            ``new_past_key_values`` is the cache after appending this token.

        Position handling
        -----------------
        When in incremental mode, we explicitly compute the absolute
        position of the new token(s) as ``[past_length, past_length + t)``
        and pass it via ``position_ids`` and/or ``cache_position`` if the
        model accepts them.  This is critical for RoPE-based attention:
        without it the model receives positions ``[0, 1, …, t-1)`` for
        every call and applies the wrong rotary embeddings, producing
        garbage outputs that are still numerically valid (so no error is
        raised — the audio just turns into noise).

        If the underlying HF model does not advertise cache support
        (``self.supports_kv_cache is False``) the cache kwargs are silently
        dropped and the call behaves like the recompute path — callers can
        always pass them and the second tuple element will simply be ``None``.
        """
        want_cache = use_cache or past_key_values is not None
        if want_cache and self.supports_kv_cache:
            B, t, _ = inputs_embeds.shape
            past_length = self._cache_length(past_key_values)
            cache_position = torch.arange(
                past_length, past_length + t,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

            extra: dict[str, Any] = {}
            if self._has_cache_position:
                extra["cache_position"] = cache_position
            if self._has_position_ids:
                extra["position_ids"] = cache_position.unsqueeze(0).expand(B, -1)

            out = self.transformer(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                **extra,
            )
            return out.last_hidden_state, out.past_key_values
        out = self.transformer(input_ids=None, inputs_embeds=inputs_embeds)
        return out.last_hidden_state, None


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-request FSM state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MossTTSLocalRequestState:
    """Per-request FSM tracking whether the model is currently emitting audio.

    All n_vq codes are emitted together at every audio step.
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
        # All values come from the HF model config so they can never drift
        # away from the checkpoint they were trained with.  The numbers in
        # the trailing comments are the verified values for the Phase-1
        # MOSS-TTS-Local checkpoint and are shown for reader convenience —
        # do NOT use them as fallbacks; let the cfg fail loudly if missing.
        #
        #   n_vq                 RVQ depth (32 codebooks per audio frame).
        #   channels             1 (text) + n_vq audio channels = 33.
        #   audio_vocab_size     codebook size; valid codes are 0..1023.
        #   audio_pad_code       the (vocab_size)-th index = pad sentinel,
        #                        embedded with padding_idx=audio_pad_code.
        #   gen_slot_id          placeholder text token forced by the FSM
        #                        every step the model is producing audio.
        #   audio_end_id         text token that exits audio mode when
        #                        sampled (after MIN_AUDIO_FRAMES).
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

        # Per-step text-channel logits captured from the local pipeline's
        # channel-0 path. Populated by `_local_forward` and consumed by
        # `compute_logits`. Cleared at the start of every `forward`.
        # Keyed by request_id; each value is a [vocab_size] tensor on the
        # same device/dtype as the backbone hidden states.
        self._pending_text_logits: dict[str, torch.Tensor] = {}

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
        # In audio mode the only legal text-channel tokens are `gen_slot`
        # (continue) and `audio_end` (terminate). `compute_logits` enforces
        # this via masking, so any other id here would indicate a bug.
        if state.is_audio:
            if token_id == self.audio_end_id:
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
        forced_text_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively predict (1 text token + 32 audio codes) per batch element.

        Mirrors the inner loop of `CustomMixin._sample` in modeling_moss_tts.py:
        every channel (text + n_vq audio) is run through the local transformer
        pipeline before its LM head. The channel-0 ("text") logits are returned
        so the caller can route them into vllm's sampler via `compute_logits`
        — applying `lm_heads[0]` to the raw global hidden state directly is an
        off-distribution shortcut that destroys text conditioning.

        For ch == 1's input we embed `forced_text_token_id` (the token vllm's
        sampler will pick — we force gen_slot via masking) instead of running
        an internal sample on channel 0. This keeps the local transformer's
        per-channel context in sync with what the next decode step will see.

        KV-cache fast path
        ------------------
        When the wrapped transformer advertises ``supports_kv_cache=True``
        (the common case on modern HF Qwen3-based remote code), we feed only
        the new token at each iteration and let the transformer's incremental
        cache hold prior K/V.  This collapses the per-frame attention cost
        from O(channels²) to O(channels): for n_vq=32, ~17× fewer attended
        token-positions across the 33-channel loop.  When the wrapper falls
        back to recompute, the original growing-context path is taken so
        behaviour is preserved exactly.

        The KV cache is local to one frame (one ``_local_forward`` call) and
        is discarded on return — there is no inter-frame dependency for the
        local transformer; the global Qwen3 backbone provides cross-frame
        context.

        Returns
        -------
        audio_codes : Tensor [B, n_vq]   (long)  predicted RVQ codes
        text_logits : Tensor [B, V_text] (model dtype) channel-0 logits
        """
        B     = global_hidden.shape[0]
        dev   = global_hidden.device
        dtype = global_hidden.dtype
        local_dim = self.config.local_hidden_size

        # Project global hidden state → first local transformer input
        current_proj = self.speech_embedding_to_local_mlp(global_hidden)  # [B, local_D]

        use_cache_path = self.local_transformer.supports_kv_cache
        verify_mode = use_cache_path and getattr(
            self.local_transformer, "_verify_kv_cache", False
        )

        # Cache-mode state: incremental K/V across the 33-channel loop.
        past_kv: Any = None
        # Recompute-mode state: growing local context.  Always allocated when
        # we're in verify mode (so we can cross-check), otherwise only when
        # use_cache_path is False.
        local_ctx: torch.Tensor | None = (
            torch.zeros(B, 0, local_dim, device=dev, dtype=dtype)
            if (not use_cache_path or verify_mode)
            else None
        )

        audio_codes: list[torch.Tensor] = []
        text_logits: torch.Tensor | None = None

        for ch in range(self.channels):   # ch = 0 (text), 1..32 (audio)
            with record_function(f"local/ch_{ch:02d}_transformer"), \
                 _TIMER.gpu("local/transformer_per_ch"):
                if use_cache_path:
                    # Incremental path: feed only the new token, reuse past K/V.
                    new_input = current_proj.unsqueeze(1)              # [B, 1, local_D]
                    local_out, past_kv = self.local_transformer(
                        new_input, past_key_values=past_kv, use_cache=True,
                    )                                                  # [B, 1, local_D]
                    last_h = local_out[:, 0, :]                        # [B, local_D]

                    if verify_mode:
                        # Run the recompute path on the same growing context
                        # and compare its last-position output to ours.
                        local_ctx = torch.cat(
                            [local_ctx, current_proj.unsqueeze(1)], dim=1
                        )
                        ref_out, _ = self.local_transformer(local_ctx)
                        ref_h = ref_out[:, -1, :]
                        max_abs = (last_h - ref_h).abs().max().item()
                        rel = max_abs / max(ref_h.abs().max().item(), 1e-6)
                        if max_abs > 1e-2 or rel > 5e-2:
                            logger.warning(
                                "[MossTTS Local] KV-cache vs recompute mismatch "
                                "at ch=%d: max_abs=%.4f rel=%.4f — falling back "
                                "to recompute output for this channel.",
                                ch, max_abs, rel,
                            )
                            last_h = ref_h
                else:
                    # Legacy recompute path: full re-forward over growing context.
                    local_ctx = torch.cat(
                        [local_ctx, current_proj.unsqueeze(1)], dim=1
                    )                                                  # [B, ch+1, local_D]
                    local_out, _ = self.local_transformer(local_ctx)   # [B, ch+1, local_D]
                    last_h = local_out[:, -1, :]                       # [B, local_D]

            # Per-channel projection + norm + LM head → logits
            with record_function(f"local/ch_{ch:02d}_head"), \
                 _TIMER.gpu("local/proj_norm_head_per_ch"):
                proj_out = self.local_to_speech_embedding_mlps[ch](last_h)  # [B, D_global]
                normed   = self.layer_norm_before_lm_heads[ch](proj_out)     # [B, D_global]
                logits   = self.lm_heads[ch](normed)                         # [B, V]

            with record_function(f"local/ch_{ch:02d}_sample"), \
                 _TIMER.gpu("local/sample_per_ch"):
                if ch == 0:
                    # Capture channel-0 logits for vllm's sampler. Defer the
                    # actual sampling to `compute_logits` so the user's
                    # SamplingParams (temp/top-k/top-p, masking) take effect.
                    # Use the forced text token (gen_slot) to drive the next
                    # local step's input — matches what vllm will append to the
                    # global stream after sampling.
                    text_logits = logits
                    next_token = torch.full(
                        (B,), forced_text_token_id, dtype=torch.long, device=dev,
                    )
                else:
                    # Audio channel: prevent the pad code from being sampled.
                    logits[:, self.audio_pad_code] = float("-inf")
                    if temperature > 0.0:
                        logits = logits / temperature
                        if top_k > 0:
                            top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values
                            logits[logits < top_k_vals[..., -1:]] = float("-inf")
                        probs      = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
                    else:
                        next_token = logits.argmax(dim=-1)  # [B]
                    audio_codes.append(next_token)

            # Re-embed sampled token → next local step's input
            with record_function(f"local/ch_{ch:02d}_embed"), \
                 _TIMER.gpu("local/embed_next_per_ch"):
                emb          = self.embedding_list[ch](next_token)             # [B, D_global]
                current_proj = self.speech_embedding_to_local_mlp(emb)        # [B, local_D]

        # Stack: [B, n_vq]
        if audio_codes:
            codes = torch.stack(audio_codes, dim=1).to(torch.long)
        else:
            codes = torch.zeros(B, self.n_vq, dtype=torch.long, device=dev)

        if text_logits is None:
            text_logits = torch.zeros(B, self.lm_heads[0].out_features, device=dev, dtype=dtype)
        return codes, text_logits

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
        # Drop warmup samples so the post-warmup report is clean.
        _TIMER.reset()

        # Re-run the KV-cache equivalence probe AFTER warmup.  If results
        # differ from the init-time probe, vLLM's profile_run has primed
        # the model's internal state in a way that breaks cache equivalence
        # — that's the bug producing runtime divergence.
        logger.info(
            "[MossTTS Local] Re-running KV-cache equivalence probe POST-WARMUP:"
        )
        self.local_transformer._probe_kv_cache_equivalence()

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
        with record_function("stage0/forward"), _TIMER.gpu("stage0/forward_total"):
            # ── 0. Derive per-request schedule info from vLLM forward context ─
            with record_function("stage0/extract_info"), _TIMER.cpu("stage0/extract_info"):
                request_ids, seq_lens_per_req, decode_positions = self._extract_request_info(
                    kwargs.get("runtime_additional_information"),
                )

            # Reset per-step text-logit cache. Populated below by `_local_forward`
            # for any audio-mode decode positions, then consumed by `compute_logits`.
            self._pending_text_logits = {}

            # ── 1. Per-request FSM bookkeeping ────────────────────────────
            with record_function("stage0/fsm_prep"), _TIMER.cpu("stage0/fsm_prep"):
                decode_states: list[MossTTSLocalRequestState] = []
                if request_ids and seq_lens_per_req and input_ids is not None:
                    decode_positions, decode_states = self._prepare_request_states(
                        input_ids=input_ids,
                        request_ids=request_ids,
                        seq_lens=seq_lens_per_req,
                    )

            # ── 2. Build embeddings (multi-channel) ───────────────────────
            with record_function("stage0/embed"), _TIMER.gpu("stage0/embed"):
                if inputs_embeds is None and input_ids is not None:
                    inputs_embeds = self.embed_input_ids(
                        input_ids,
                        multimodal_embeddings=kwargs.get("multimodal_embeddings"),
                        request_ids=request_ids if request_ids else None,
                        seq_lens=seq_lens_per_req if seq_lens_per_req else None,
                    )

            # ── 3. Global Qwen3 backbone ──────────────────────────────────
            with record_function("stage0/backbone"), _TIMER.gpu("stage0/backbone"):
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
                audio_positions = [
                    p for p, m in zip(decode_positions, audio_mask) if m
                ]
                audio_states = [s for s, m in zip(decode_states, audio_mask) if m]
                decode_request_ids = [
                    r for r, sl in zip(request_ids, seq_lens_per_req) if sl == 1
                ]
                audio_request_ids = [
                    r for r, m in zip(decode_request_ids, audio_mask) if m
                ]

                if audio_positions:
                    pos_t = torch.tensor(audio_positions, device=hidden_states.device)
                    decode_hidden = hidden_states[pos_t]  # [B_audio, D_global]

                    with record_function("stage0/local_forward"), _TIMER.gpu("stage0/local_forward"):
                        codes, text_logits = self._local_forward(
                            decode_hidden,
                            forced_text_token_id=self.gen_slot_id,
                            temperature=kwargs.get("audio_temperature", 1.0),
                            top_k=kwargs.get("audio_top_k", 50),
                            top_p=kwargs.get("audio_top_p", 0.95),
                        )  # [B_audio, n_vq], [B_audio, V_text]

                    with record_function("stage0/store_codes"), _TIMER.gpu("stage0/store_codes"):
                        # Cache codes per-request for the NEXT decode step's
                        # multi-channel embedding contribution.  Note: this
                        # currently performs a GPU→CPU sync per request (see
                        # store_next_audio_row); look for the spike in this
                        # phase to confirm the known overhead.
                        for state, row in zip(audio_states, codes):
                            state.store_next_audio_row(row)

                        # Cache channel-0 text logits so `compute_logits` can return
                        # the local-pipeline-processed distribution rather than
                        # reapplying lm_heads[0] to the raw global hidden state.
                        for req_id, tl in zip(audio_request_ids, text_logits):
                            self._pending_text_logits[req_id] = tl

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

    # Minimum number of audio frames to emit before allowing the model to
    # sample `audio_end`. Without this guard the model terminates audio
    # generation almost immediately: empirically the unconstrained text head
    # in vllm-omni's setup does NOT favor `gen_slot` after `<|audio_start|>`
    # (it leans toward random vocab tokens), and once both `gen_slot` and
    # `audio_end` are made the only legal options, `audio_end`'s raw logit
    # tends to win at the boundary.
    #
    # Rough sizing: codec is 24kHz with ~12.5 frames/sec, so 150 frames
    # caps the forced-audio period at ~12 seconds. For longer utterances
    # bump this higher; for very short ones the trailing silence will be
    # modest (and can be trimmed downstream).
    MIN_AUDIO_FRAMES: int = 10

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        """Text channel logits (channel-0 LM head), with FSM-driven gating.

        For audio-mode rows we substitute the channel-0 logits captured by
        `_local_forward` (computed via the full local pipeline:
        speech_embedding_to_local_mlp → local_transformer →
        local_to_speech_embedding_mlps[0] → layer_norm_before_lm_heads[0] →
        lm_heads[0]). Applying lm_heads[0] directly to the raw global hidden
        state — which is what we did originally — is off-distribution and
        produces a degraded "want to stop" signal that fires regardless of
        text length.

        Masking:
          - Audio mode, steps < MIN_AUDIO_FRAMES → force `gen_slot` (warm-up
            guard against premature termination right after `<|audio_start|>`).
          - Audio mode, steps ≥ MIN_AUDIO_FRAMES → unmask only
            `{gen_slot, audio_end}` so vllm's sampler picks freely between
            "continue" and "stop" using the corrected logits.
          - Outside audio mode → mask audio-control tokens.
        """
        with record_function("stage0/compute_logits"), _TIMER.gpu("stage0/compute_logits"):
            logits = self.lm_heads[0](hidden_states)   # [L, vocab_size]
            if not self._last_request_ids:
                return logits

            neg_inf = float("-inf")
            for row_idx, request_id in enumerate(self._last_request_ids):
                if row_idx >= logits.shape[0]:
                    break
                state = self._request_states.get(request_id)
                if state is None:
                    continue

                # Substitute local-pipeline text logits when available.
                cached_tl = self._pending_text_logits.get(request_id)
                if cached_tl is not None:
                    logits[row_idx] = cached_tl.to(logits.dtype)

                row = logits[row_idx]

                if state.is_audio:
                    if state.audio_steps_generated < self.MIN_AUDIO_FRAMES:
                        keep = row[self.gen_slot_id].clone()
                        row.fill_(neg_inf)
                        row[self.gen_slot_id] = keep
                    else:
                        gen_keep = row[self.gen_slot_id].clone()
                        end_keep = row[self.audio_end_id].clone()
                        row.fill_(neg_inf)
                        row[self.gen_slot_id]  = gen_keep
                        row[self.audio_end_id] = end_keep
                else:
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
