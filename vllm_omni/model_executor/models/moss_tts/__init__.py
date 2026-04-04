# Copyright 2025 OpenMOSS / vllm-omni contributors.
#
# Licensed under the Apache License, Version 2.0.
#
# Entry point for the MOSS-TTS-Local vllm-omni integration.
# Exposes the unified dispatch class that vllm-omni's model registry resolves.

from vllm_omni.model_executor.models.moss_tts.moss_tts import MossTTSForConditionalGeneration

__all__ = ["MossTTSForConditionalGeneration"]
