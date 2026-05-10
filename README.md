# HPML Final Project: MOSS-TTS Integration for vLLM-Omni

> **Course:** High Performance Machine Learning  
> **Semester:** Spring 2026  
> **Instructor:** Dr. Kaoutar El Maghraoui  

This repository is a course-project fork of `vllm-omni` focused on integrating
the MOSS-TTS model family into vLLM-Omni and benchmarking inference-serving
performance against native HuggingFace-style baselines.

## Team Information
**Team Name:** Omnipresent

| Member | UNI | Primary contributions |
| --- | --- | --- |
| Andrew Chung | ac5905 | Metric scripts, W&B integration, experiment tracking |
| Yuting Liu | yl5961 | MOSS-TTS-Local integration; Stage 0 local-transformer KV-cache; Stage 1 batch decoding work |
| Xingru Lu | xl3602 | MOSS-TTS-Delay async-chunk integration support; profiling; preliminary CUDA Graph work |
| Yutao Mao | ym3019 | MOSS-TTS-Delay integration; shared Stage 1 decoder optimization |

## Submission
- **GitHub Repository:** <https://github.com/candylyt/vllm-omni>
- **Final presentation:** [`deliverables/HPML Final Project Presentation.pptx`](deliverables/HPML%20Final%20Project%20Presentation.pptx)
- **Final report source:** [`deliverables/Report.pdf`](deliverables/Report.pdf)
- **Experiment tracking dashboard:** <https://wandb.ai/ac5905-columbia-university/hpml-final-project/runs/xfa9dt56?nw=nwuserac5905>

## Branch Map

The work is split across branches:

| Branch | Purpose |
| --- | --- |
| `moss-tts-delay` | MOSS-TTS-Delay model integration, sync and async configs, Delay-specific optimizations, offline example, tests |
| `moss-tts-local` | MOSS-TTS-Local model integration, sync and async configs, offline example |
| `metric-branch` | Experiment scripts for throughput, RTF/FCL profiling, W&B logging, and runbook notes |
| `main` | Submission README and deliverables |

## 1. Problem Statement

MOSS-TTS is a state-of-the-art text-to-speech model family, but its architecture
does not plug directly into the standard vLLM-Omni model path. The project
targets **inference serving**, not training: we integrate MOSS-TTS-Local and
MOSS-TTS-Delay into vLLM-Omni, then optimize throughput, real-time factor (RTF),
and first chunk latency/time to first audio (FCL/TTFA). The main bottlenecks are
autoregressive audio-code generation, Stage 1 codec decoding, inter-stage
handoff overhead, and the lack of native continuous batching in the
HuggingFace-style baseline.

## 2. Model and System Description

### Models

- **MOSS-TTS-Local:** Global-latent + local-transformer TTS model. The Stage 0
  path uses a global Qwen3-style temporal transformer plus a local depth
  transformer to emit one complete 32-codebook RVQ audio frame per audio decode
  step. The model is approximately 1.7B parameters in the presentation.
- **MOSS-TTS-Delay:** Delay-pattern TTS model using a single large transformer
  backbone, approximately 8B parameters. Stage 0 emits delayed RVQ rows, which
  must be de-delayed into frame-major `[T, 32]` codec codes before audio decode.
- **MOSS-Audio-Tokenizer:** Shared Stage 1 CAT codec decoder. It consumes
  32-layer RVQ codec codes and produces 24 kHz waveform audio.

### vLLM-Omni Integration

Both MOSS models are integrated as two-stage vLLM-Omni TTS pipelines:

1. **Stage 0, AR stage:** consumes the text prompt and generates discrete RVQ
   audio codes.
2. **Stage 1, decoder stage:** consumes flattened codec codes and decodes them
   into 24 kHz waveform audio.

The stage configs bind Stage 0 to `OmniARScheduler` and Stage 1 to
`OmniGenerationScheduler`. The sync configs invoke a stage input processor after
full Stage 0 generation. The async configs use `async_chunk: true`, a
`SharedMemoryConnector`, and a custom next-stage processor so Stage 1 can decode
chunks while Stage 0 continues generating later audio codes.

### Important Files by Branch

#### `moss-tts-delay`

- `vllm_omni/model_executor/models/moss_tts/moss_tts.py`
- `vllm_omni/model_executor/models/moss_tts/moss_tts_delay_ar_stage.py`
- `vllm_omni/model_executor/models/moss_tts/moss_tts_decoder.py`
- `vllm_omni/model_executor/stage_input_processors/moss_tts.py`
- `vllm_omni/model_executor/stage_configs/moss_tts_delay.yaml`
- `vllm_omni/model_executor/stage_configs/moss_tts_delay_async.yaml`
- `examples/offline_inference/moss_tts_delay/end2end.py`
- `examples/offline_inference/moss_tts_delay/benchmark_async_chunk.py`
- `tests/model_executor/stage_input_processors/test_moss_tts_delay.py`
- `tests/model_executor/models/moss_tts/test_moss_tts_decoder_batch_decode.py`

#### `moss-tts-local`

- `vllm_omni/model_executor/models/moss_tts/moss_tts_local.py`
- `vllm_omni/model_executor/models/moss_tts/moss_tts_local_ar_stage.py`
- `vllm_omni/model_executor/models/moss_tts/moss_tts_local_decoder.py`
- `vllm_omni/model_executor/stage_input_processors/moss_tts_local.py`
- `vllm_omni/model_executor/stage_configs/moss_tts_local.yaml`
- `vllm_omni/model_executor/stage_configs/moss_tts_local_async.yaml`
- `examples/offline_inference/moss_tts_local/end2end.py`

#### `metric-branch`

- `HPML-metrics/moss_tts_profile.py`
- `HPML-metrics/throughput_moss_tts_local.py`
- `HPML-metrics/throughput_moss_tts_delay.py`
- `HPML-metrics/torch_profile_local.py`
- `HPML-metrics/hf_profile.ipynb`
- `HPML-metrics/hf_throughput.ipynb`
- `HPML-metrics/uni_functions.py`
- `HPML-metrics/running_on_vast_ai.md`

## 3. Optimizations Implemented

### System-Level Optimizations

- **Stage-level batching:** vLLM-Omni batches work at stage boundaries,
  especially in the audio decoder stage, reducing per-request overhead.
- **Async two-stage decoding:** Stage 0 streams codec chunks to Stage 1 so audio
  can be emitted before the full utterance finishes. This is the primary TTFA
  optimization.
- **Stage 1 batch decoding:** The decoder path supports batched decode of
  multiple full code sequences in sync mode and streaming decode in async mode.
- **vLLM Qwen3 backend benefits:** The global transformer portions benefit from
  vLLM scheduling, PagedAttention, and fused kernels where they use vLLM-native
  Qwen3 components.

### MOSS-TTS-Local Optimizations

- **Local-transformer KV-cache management:** In `moss-tts-local-optimization`,
  the local transformer avoids recomputing the full growing local context every
  audio step. The presentation reports about an 8% CUDA-time reduction in the
  profiled Stage 0 local-transformer section.
- **Async chunking:** `moss_tts_local_async.yaml` flushes complete frame-major
  RVQ rows through `llm2decoder_async_chunk`. Since Local emits one complete
  frame per audio decode step, the first chunk can be as small as
  `codec_chunk_frames: 3`.

### MOSS-TTS-Delay Optimizations

- **Incremental async de-delay:** The async processor scatters each newly
  generated delay row directly into restored frame slots using a bounded
  per-request ring buffer, avoiding full matrix rebuild and full de-delay every
  step.
- **Vectorized sync de-delay:** `_apply_de_delay_pattern()` uses tensor index
  grids instead of looping over the 32 codebooks.
- **Packed audio embeddings:** Delay Stage 0 replaces 32 separate audio
  embedding modules with `audio_embedding_weight[n_vq, vocab, hidden]`.
- **Packed audio heads:** Delay Stage 0 uses
  `audio_lm_head_weight[n_vq, vocab, hidden]` and a batched `einsum` to compute
  logits for all codebooks.
- **Vectorized sampling:** Sampling is vectorized over active request/codebook
  pairs instead of looping over all 32 codebooks.
- **Avoided CPU history copies:** When `repetition_penalty=1.0`, the benchmark
  setting, Delay Stage 0 skips copying generated audio rows to CPU for
  repetition-history bookkeeping.
- **Reduced async transfer overhead:** The Delay async path skips unnecessary
  Stage 0 hidden-state CPU copies and suppresses unnecessary per-step pooling
  payloads to the client accumulator while still feeding the chunk-transfer
  adapter.
- **Connector cleanup:** Per-request Delay async state caches are cleaned after
  terminal chunks.
- **Dynamic async chunking:** `moss_tts_delay_async.yaml` uses
  `initial_codec_chunk_frames: 3` for TTFA and `codec_chunk_frames: 50` for
  steadier-state decoder efficiency.

### Preliminary/Not Fully Adopted Work

- **CUDA Graph:** Preliminary Delay experiments applied CUDA Graph mainly to the
  AR backbone and showed a 44% self-CPU-time reduction, but this is not part of
  the main benchmark table. Multiple batch sizes remain difficult because of
  incompatibility between `torch.compile` and the current `OmniOutput` path.
- **Full vLLM-native Local transformer:** The Local transformer is still not
  fully converted into a vLLM-native transformer, limiting its ability to use
  PagedAttention and fused kernels as effectively as the global Qwen3 backbone.

## 4. Experiment Methodology

### Data

- **RTF/FCL profiling:** 20 examples from HuggingFace WikiText
  `wikitext-103-raw-v1` test split, sampled with varying sentence lengths. The
  report uses 10-60 words as the final range.
- **Throughput:** A fixed sentence to reduce variance:
  `"The weather is so nice today and the birds are singing in the trees."`

### Metrics

- **RTF:** `wall_time / generated_audio_duration`. Lower is better.
- **FCL/TTFA:** elapsed time until the first audio chunk appears. For sync and
  HF baselines, this equals total latency because audio is returned only after
  full generation.
- **Throughput:** `generated_audio_duration / wall_time`, reported as
  audio-seconds per second. Higher is better.
- **Speedup:** relative to batch size 1 within the same model/mode.

### Hardware and Runtime

- NVIDIA A100 80GB GPU.
- Vast AI and Google Colab A100 environments were used.
- At least 100GB disk was needed for model weights and intermediate outputs.
- The project runbook notes CUDA 12.8 and vLLM 0.19.1 as the working runtime.

### Profiling and Tracking Tools

- PyTorch Profiler / `torch.profiler` for HF paths.
- vLLM profiler integration and vLLM logs for integrated paths.
- Weights & Biases for run-level logging.
- Experiments were averaged across three runs or multiple examples where
  practical; the metric scripts in `metric-branch` use one repeat for some
  throughput sweeps, with W&B logging enabled.

## 5. Final Results Summary

### Headline Results

| Result | Baseline | vLLM-Omni result | Improvement |
| --- | ---: | ---: | ---: |
| MOSS-TTS-Delay throughput at BS=256 | 25.006 audio-s/s HF | 69.956 audio-s/s sync | 2.8x |
| MOSS-TTS-Local throughput at BS=256 | 15.148 audio-s/s HF | 26.718 audio-s/s sync | 1.8x |
| MOSS-TTS-Local FCL/TTFA | 21.860 s HF | 0.557 s async | 39.2x lower |
| MOSS-TTS-Delay FCL/TTFA | 5.002 s HF | 0.970 s async | 5.2x lower |
| Local Stage 0 local-transformer CUDA time | before KV-cache | after KV-cache | ~8% lower |
| Delay AR backbone self CPU time | eager/pre-CUDA-Graph | preliminary CUDA Graph | 44% lower |

**Hardware:** [1× NVIDIA A100 80GB SXM, CUDA 12.8]

**Headline result:** By inferencing the MOSS-TTS-Local and MOSS-TTS-Delay models through vLLM-Omni, leveraging both native vLLM optimizations and newly introduced techniques such as stage-level batching and asynchronous two-stage decoding, we achieved up to 2.8× higher throughput and 39.2× lower first-chunk latency (FCL/TTFA) compared to the original Hugging Face implementations.

### Throughput Tables

#### MOSS-TTS-Local

| Batch size | HF tput | HF speedup | Async tput | Async speedup | Sync tput | Sync speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.299 | 1.00x | 0.536 | 1.00x | 0.528 | 1.00x |
| 4 | 0.973 | 3.25x | 0.705 | 1.32x | 1.664 | 3.09x |
| 16 | 3.402 | 11.38x | 1.944 | 3.63x | 4.002 | 7.57x |
| 64 | 10.139 | 33.91x | 7.470 | 14.91x | 13.574 | 25.69x |
| 128 | 11.610 | 38.83x | 11.866 | 22.15x | 21.290 | 40.29x |
| 256 | 15.148 | 50.66x | 18.265 | 34.09x | 26.718 | 50.56x |

#### MOSS-TTS-Delay

| Batch size | HF tput | HF speedup | Async tput | Async speedup | Sync tput | Sync speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.713 | 1.00x | 1.664 | 1.00x | 1.566 | 1.00x |
| 4 | 2.877 | 4.04x | 3.590 | 2.16x | 3.839 | 2.45x |
| 16 | 7.088 | 9.94x | 13.145 | 7.90x | 14.152 | 9.04x |
| 64 | 18.612 | 26.10x | 19.055 | 11.45x | 40.291 | 25.73x |
| 128 | 21.667 | 30.39x | 20.204 | 12.14x | 54.463 | 34.78x |
| 256 | 25.006 | 35.07x | 21.817 | 13.11x | 69.956 | 44.68x |

The async Delay path hit OOM behavior at batch sizes 128 and 256 and fell back
to single-request processing after memory pressure, which limited throughput at
the largest batch sizes. We expect multi-GPU separation of Stage 0 and Stage 1
to improve this operating point.

### RTF/FCL Summary

| Metric | Local async | Local sync | Delay async | Delay sync | HF Delay | HF Local |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTF mean | 1.876 | 1.202 | 0.470 | 0.489 | 1.730 | 3.292 |
| Total latency mean (s) | 11.859 | 10.931 | 2.901 | 2.958 | 5.002 | 21.860 |
| First chunk latency (s) | 0.557 | 10.931 | 0.970 | 2.948 | 5.002 | 21.860 |
| Number of chunks mean | 27.00 | 1.00 | 3.25 | 1.00 | 1.00 | 1.00 |

## 6. Reproducing the Work

### Environment Setup

The root [`environment.yml`](environment.yml) captures the reported runtime
setup as closely as the submitted artifacts allow:

```bash
conda env create -f environment.yml
conda activate moss-tts-hpml
```

The runbook on `metric-branch` used the following equivalent setup on Vast AI:

```bash
conda create -n moss-tts python=3.11 -y
conda activate moss-tts

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

pip install uv --quiet
pip install datasets wandb soundfile huggingface_hub
uv pip install vllm==0.19.1 --torch-backend=auto
uv pip install setuptools_scm
uv pip install -e . --no-build-isolation --no-cache-dir
```

On Vast AI, the runbook also required adding Torch's bundled `nvjitlink`
directory to `LD_LIBRARY_PATH`.

### Download Model Weights

```bash
mkdir -p $HOME/weights

hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --local-dir $HOME/weights/moss-tts-local

hf download OpenMOSS-Team/MOSS-TTS \
  --local-dir $HOME/weights/moss-tts-delay

hf download OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --local-dir $HOME/weights/moss-audio-tokenizer

export MOSS_TTS_LOCAL_PATH=$HOME/weights/moss-tts-local
export MOSS_TTS_DELAY_PATH=$HOME/weights/moss-tts-delay
export MOSS_AUDIO_TOKENIZER_PATH=$HOME/weights/moss-audio-tokenizer
```

### Run MOSS-TTS-Local Inference

```bash
git checkout moss-tts-local
export REPO=$(pwd)
export PYTHONPATH=$REPO

python examples/offline_inference/moss_tts_local/end2end.py \
  --model "$MOSS_TTS_LOCAL_PATH" \
  --stage-configs-path vllm_omni/model_executor/stage_configs/moss_tts_local.yaml \
  --text "The weather is so nice today." \
  --output-dir ./output_audio_local
```

Use `moss_tts_local_async.yaml` for async chunking.

### Run MOSS-TTS-Delay Inference

```bash
git checkout moss-tts-delay
export REPO=$(pwd)
export PYTHONPATH=$REPO

python examples/offline_inference/moss_tts_delay/end2end.py \
  --model "$MOSS_TTS_DELAY_PATH" \
  --stage-configs-path vllm_omni/model_executor/stage_configs/moss_tts_delay.yaml \
  --text "The weather is so nice today." \
  --output-dir ./output_audio_delay
```

Use `moss_tts_delay_async.yaml` for async chunking.

### Run Profiling and Throughput Scripts

The root [`scripts/`](scripts/) folder contains the reproducibility scripts for
the reported throughput and RTF/FCL tables. See
[`scripts/README.md`](scripts/README.md) for the full command set and worktree
layout.

```bash
# The MOSS implementation files live on model branches. Use worktrees so the
# root scripts can run against the correct branch checkout.
git worktree add ../vllm-omni-local moss-tts-local
git worktree add ../vllm-omni-delay moss-tts-delay

export VLLM_LOGGING_LEVEL=WARNING
export WANDB_DISABLED=true  # remove this line if logging to W&B

# RTF/FCL profiling on WikiText samples for Local.
export REPO=$(realpath ../vllm-omni-local)
export PYTHONPATH=$REPO
python scripts/profile_moss_tts.py \
  --model-type local \
  --model "$MOSS_TTS_LOCAL_PATH" \
  --repo "$REPO" \
  --modes async sync \
  --n 20 \
  --min-words 10 \
  --max-words 60 \
  --output-json results/local_profile.json

# RTF/FCL profiling on WikiText samples for Delay.
export REPO=$(realpath ../vllm-omni-delay)
export PYTHONPATH=$REPO
python scripts/profile_moss_tts.py \
  --model-type delay \
  --model "$MOSS_TTS_DELAY_PATH" \
  --repo "$REPO" \
  --modes async sync \
  --n 20 \
  --min-words 10 \
  --max-words 60 \
  --output-json results/delay_profile.json

# Throughput sweeps with fixed text.
export REPO=$(realpath ../vllm-omni-local)
export PYTHONPATH=$REPO
python scripts/throughput_moss_tts.py \
  --model-type local \
  --model "$MOSS_TTS_LOCAL_PATH" \
  --repo "$REPO" \
  --mode sync \
  --batch-sizes 1 4 16 64 128 256 \
  --output-json results/local_sync_throughput.json

export REPO=$(realpath ../vllm-omni-delay)
export PYTHONPATH=$REPO
python scripts/throughput_moss_tts.py \
  --model-type delay \
  --model "$MOSS_TTS_DELAY_PATH" \
  --repo "$REPO" \
  --mode async \
  --batch-sizes 1 4 16 64 128 256 \
  --output-json results/delay_async_throughput.json
```

The scripts log to the W&B project `hpml-final-project` when W&B is configured
and `WANDB_DISABLED` is not set.

## 7. Observations and Lessons Learned

- vLLM-Omni's stage-aware batching and scheduling are the largest contributors
  to high-batch throughput improvements.
- Async chunking substantially reduces TTFA/FCL because Stage 1 can emit audio
  before Stage 0 completes the whole utterance.
- Sync mode generally has better high-batch throughput because it avoids
  streaming overhead and decodes larger complete code sequences.
- Delay async has an inherent ramp-up cost because complete frame-major rows are
  only available after enough shifted quantizer positions arrive.
- The HuggingFace baseline lacks native continuous batching, so we compare
  against static batching for throughput.
- PyTorch profiling was less directly comparable between HF and vLLM-Omni
  because the integrated two-stage execution path has different scopes and
  boundaries.

## 8. Limitations and Future Work

- Support reference-audio continuation, ambience/noise prompts, and richer
  prompting styles beyond direct TTS.
- Test multi-GPU setups, especially separating Stage 0 and Stage 1 to reduce
  memory pressure for async Delay at large batch sizes.
- Make the MOSS-TTS-Local local transformer more fully vLLM-native so it can
  better use PagedAttention and fused kernels.
- Improve CUDA Graph support once the `OmniOutput` / `torch.compile`
  incompatibility is resolved.
- Run more realistic online-serving benchmarks with random arrivals and more
  varied text lengths.

## AI Use Disclosure

**Did your team use any AI tool in completing this project?**
- [ ] No, we did not use any AI tool.
- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** ChatGPT, Claude, GitHub Copilot

**Specific purpose:** We used AI tools to understand the two-stage pipeline of vLLM-Omni, debugging compatibility issues between vLLM-Omni and MOSS-TTS. It was also used to understand the codebase in general, especially for how profiling was implemented and the internal workings of vLLM/vLLM-Omni. We also utilized AI such as copilot's autofill function for loops, boilerplate code, and cleaning up prose. Also, debugging environment setup for running models on Vast AI. 

**Sections affected:** - vllm_omni/model_executor/models/moss_tts/moss_tts_local_ar_stage.py (moss-tts-local branch), vllm_omni/model_executor/models/moss_tts/moss_tts_local_decoder.py (moss-tts-local branch), vllm_omni/model_executor/models/moss_tts/moss_tts_delay_ar_stage.py (moss-tts-delay branch), vllm_omni/model_executor/models/moss_tts/moss_tts_decoder.py (moss-tts-delay branch),
benchmarking scripts under HPML-metrics (metric-branch)

**How we verified correctness:** We validated all AI-assited outputs by manually reviewing the generated code, running and reproducing all experiments ourselves, cross-checking the outputs benchmark results against the actual execution logs.

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

## License

This repository is a course-project fork of vLLM-Omni. The upstream project is
released under the Apache License 2.0; see [`LICENSE`](LICENSE).

## References

- OpenMOSS Team, MOSS-TTS Family: <https://github.com/OpenMOSS/MOSS-TTS>
- OpenMOSS Team, MOSS-Audio-Tokenizer:
  <https://arxiv.org/abs/2602.10934>
- vLLM-Omni paper: <https://arxiv.org/abs/2602.02204>
- PagedAttention/vLLM paper: <https://arxiv.org/abs/2309.06180>
- Qwen3-TTS technical report: <https://arxiv.org/abs/2601.15621>

### Citation
If you build on this work, please cite:
```bibtex
@misc{Omnipresent2026hpml,
title = {MOSS-TTS Integration for vLLM-Omni},
author = {Chung, Andrew and Liu, Yuting and Lu, Xingru and Mao Yutao},
year = {2026},
note = {HPML Spring 2026 Final Project, Columbia University},
url = {https://github.com/candylyt/vllm-omni}
}
```
### Contact
Open a GitHub Issue or email *[ym3019@columbia.edu]*.
---
*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
