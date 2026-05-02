# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
"""
MOSS-TTS-Local integration for vllm-omni.

This package wires the MOSS-TTS-Local two-stage TTS pipeline into vllm-omni's
multi-stage runtime.  Read this file first — it is the only place where the
full integration is described end-to-end.  Every other file in the integration
points back here for the big picture.

================================================================================
  Pipeline (text → audio)
================================================================================

    user prompt (text + chat template + "<|audio_start|>")
            │
            ▼
    ┌───────────────────────────────────────────────────────────┐
    │  Stage 0   "ar_stage"                                     │
    │  ─────────────────────                                    │
    │  Class : MossTTSARStageModel  (moss_tts_ar_stage.py)      │
    │  Worker: vllm_omni AR worker  (worker_type: "ar")         │
    │                                                           │
    │  Per decode step:                                         │
    │    Qwen3-1.7B global backbone produces hidden state.      │
    │    A 4-block local transformer autoregressively predicts  │
    │    32 RVQ codes (one frame).  An FSM gates entry/exit     │
    │    between text and audio modes via logits masking.       │
    │                                                           │
    │  Output (per request, per step):                          │
    │      code_predictor_codes  Tensor [B, 1, n_vq=32, 1]      │
    └────────────────────┬──────────────────────────────────────┘
                         │
                         │  Stage-0 outputs are flattened into a
                         │  row-major (frame-first) token stream:
                         │      [t0v0, t0v1, …, t0v31,
                         │       t1v0, …,           t1v31, …]
                         │
                         │  Bridge function (in stage_input_processors/
                         │  moss_tts.py):
                         │    • llm2decoder              — batch (sync)
                         │    • llm2decoder_async_chunk  — streaming (async)
                         │
                         ▼
    ┌───────────────────────────────────────────────────────────┐
    │  Stage 1   "decoder"                                      │
    │  ─────────────────                                        │
    │  Class : MossTTSDecoderModel  (moss_tts_decoder.py)       │
    │  Worker: vllm_omni generation worker (worker_type:        │
    │          "generation")                                    │
    │                                                           │
    │  Wraps the MOSS CAT codec (1.6 B, separate checkpoint).   │
    │  Reshapes the flat token stream into [n_vq, T] and decodes│
    │  to a 24 kHz waveform.  Streaming mode keeps a per-request│
    │  KV-cache so successive chunks decode with continuous     │
    │  causal context.                                          │
    │                                                           │
    │  Output: 1-D float32 audio tensor at 24 kHz               │
    └───────────────────────────────────────────────────────────┘

================================================================================
  File map
================================================================================

  __init__.py                     This file.  Architectural overview.
  moss_tts.py                     Dispatcher class (registered with vLLM).
                                  Routes to AR or Decoder based on `model_stage`.
  moss_tts_ar_stage.py            Stage 0: AR + local transformer (~940 LOC).
  moss_tts_decoder.py             Stage 1: CAT codec wrapper (~600 LOC).

  Outside this package:
  ../stage_input_processors/moss_tts.py   Bridge: AR codes → decoder input prompt.
  ../stage_configs/moss_tts.yaml          Batch (sync) pipeline config.
  ../stage_configs/moss_tts_async.yaml    Streaming (async-chunk) pipeline config.
  ../models/registry.py                   Registers the dispatcher under two arch names.

================================================================================
  Integration touch-points
================================================================================

The runtime reads these YAML keys to wire the pipeline together.  Each maps to
a specific extension point inside vllm-omni:

  model_arch: MossTTSForConditionalGeneration
        One arch is registered (see registry.py).  Both stages instantiate the
        same dispatcher class; the dispatcher picks the right sub-model from
        `model_stage`.

  model_stage: "ar_stage" | "decoder"
        Read by MossTTSForConditionalGeneration.__init__ in moss_tts.py.
        Selects MossTTSARStageModel vs MossTTSDecoderModel.

  worker_type: "ar" | "generation"
        Resolved by vllm_omni/engine/stage_init_utils.py:resolve_worker_cls.
        Picks the AR worker (multi-step token generation) or the generic
        generation worker (codec decode).

  scheduler_cls: OmniARScheduler | OmniGenerationScheduler
        Per-stage scheduler.  AR scheduler handles the global+local
        autoregressive loop; generation scheduler handles plain decode.

  engine_output_type: "latent" | "audio"
        Tells the runtime what shape of output to expect from `forward()`.
        Stage 0 returns multimodal `code_predictor_codes`; Stage 1 returns
        the final waveform tensor.

  engine_input_source: [0]   (Stage 1 only)
        List of stage indices that feed this stage.  `[0]` means "Stage 1
        consumes Stage 0's outputs."

  custom_process_input_func: …llm2decoder              (batch mode)
  custom_process_next_stage_input_func: …llm2decoder_async_chunk  (streaming)
        The Stage-0 to Stage-1 bridge functions, defined in
        stage_input_processors/moss_tts.py.  Batch mode runs the bridge once
        at end of generation; streaming mode runs it after every decode step
        and ships chunks through a SharedMemoryConnector.

  enable_prefix_caching: false
        Required.  Audio code streams are not stable prefixes — a tiny
        difference in early codes diverges the entire downstream waveform.
        Do NOT flip this to true.

================================================================================
  Operational notes  (see INFERENCE_GUIDE.md for the full guide)
================================================================================

The CAT codec is a separate checkpoint from the main MOSS-TTS-Local model.
Set `MOSS_AUDIO_TOKENIZER_PATH` (or the YAML `audio_tokenizer_path`) to point
at it.  See examples/offline_inference/moss_tts/INFERENCE_GUIDE.md for the
full operational walk-through (env vars, weight downloads, batch vs streaming
trade-offs, troubleshooting).
"""

from vllm_omni.model_executor.models.moss_tts.moss_tts import MossTTSForConditionalGeneration

__all__ = ["MossTTSForConditionalGeneration"]
