# MOSS-TTS-Local: Complete Inference Guide

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Two Inference Modes Compared](#2-two-inference-modes-compared)
3. [Prerequisites](#3-prerequisites)
   - [3.0 Instance requirements](#30-instance-requirements)
   - [3.1 Software installation](#31-software-installation)
   - [3.2 Weights](#32-weights)
   - [3.3 Code](#33-code)
   - [3.4 One-time codec config patch](#34-one-time-codec-config-patch)
4. [Running Inference](#4-running-inference)
   - [Option A — Interactive (direct)](#option-a--interactive-direct)
   - [Option B — Slurm job submission](#option-b--slurm-job-submission)
5. [Expected Output and Log Interpretation](#5-expected-output-and-log-interpretation)
6. [YAML Configuration Reference](#6-yaml-configuration-reference)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Architecture Overview

MOSS-TTS-Local inference consists of two sequential stages:

```
Text input
   │
   ▼
┌──────────────────────────────────────┐
│  Stage 0 — AR stage                  │  Qwen3-1.7B global backbone + local
│  MossTTSForConditionalGeneration     │  transformer → one row of n_vq=32
│  (ar_stage)                          │  RVQ codebook entries per decode step
└────────────────┬─────────────────────┘
                 │  SharedMemoryConnector
                 │  async: flush every codec_chunk_frames (=3) frames
                 │  sync:  transfer once, after Stage 0 finishes completely
                 ▼
┌──────────────────────────────────────┐
│  Stage 1 — CAT Codec decoder         │  RVQ codes → 24 kHz audio waveform
│  MossTTSForConditionalGeneration     │  (streaming KV-cache maintains causal
│  (decoder)                           │   context across chunk boundaries)
└────────────────┬─────────────────────┘
                 │
                 ▼
            output.wav
```

**First-chunk latency — no delay-pattern penalty:**
Unlike the delay variant, MOSS-TTS-Local emits a complete 32-code row at every
decode step, so **the first chunk is ready after just `codec_chunk_frames`
decode steps** (default 3). There is no 33-step ramp-up and no (n_vq − 1)
codebook fill — the first audio sample arrives almost immediately once Stage 0
begins generating.

---

## 2. Two Inference Modes Compared

|                             | **async_chunk mode (streaming)**                                | **sync batch mode**                                                    |
| --------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| YAML config                 | `moss_tts_async.yaml`                                           | `moss_tts.yaml`                                                        |
| `--mode` flag               | `async`                                                         | `sync`                                                                 |
| `async_chunk` flag in YAML  | `true`                                                          | `false`                                                                |
| Stage 0 → Stage 1 transfer  | Flush every 3 frames via `SharedMemoryConnector`                | Transfer once after Stage 0 **finishes completely**, via `llm2decoder` |
| Stage 0 processing function | `llm2decoder_async_chunk` (triggered on Stage 0 side)           | None (Stage 0 side does nothing)                                       |
| Stage 1 processing function | Streaming KV-cache, receives chunks one at a time               | `llm2decoder` (Stage 1 side, one-shot)                                 |
| `async_scheduling`          | `true` on both stages                                           | `false` on both stages                                                 |
| `num_chunks` in output      | **> 1** (typically ~9–10)                                       | **= 1**                                                                |
| First-chunk latency         | Stage 1 starts decoding while Stage 0 is still running          | Stage 1 only starts after Stage 0 finishes                             |
| Output audio quality        | Identical to sync (streaming KV-cache preserves causal context) | Reference quality                                                      |

> **Audio-quality note:** Async mode uses the CAT codec's streaming KV-cache
> so that causal state is maintained across chunk boundaries. With this
> enabled, async and sync produce sample-equivalent audio — no boundary
> glitches. See [§7 Troubleshooting](#7-troubleshooting) if you hear artifacts.

### Execution timeline (schematic)

```
async_chunk mode:
  time ──────────────────────────────────────────────────>
  Stage 0:  [generating tokens ··· chunk1 ··· chunk2 ··· ... done]
  Stage 1:              [decode chunk1][decode chunk2][...]

sync batch mode:
  time ──────────────────────────────────────────────────>
  Stage 0:  [generating tokens .............................done]
  Stage 1:                                              [decode all]
```

---

## 3. Prerequisites

### 3.0 Instance requirements

| Requirement | Minimum / Tested            |
| ----------- | --------------------------- |
| GPU VRAM    | 80 GB+ (tested: A100 80 GB) |
| CUDA        | 12.1-12.6                   |
| System RAM  | 24 GB+                      |
| Disk        | 80 GB free                  |
| OS          | Linux (x86_64)              |

---

### 3.1 Software installation

These steps set up a fresh Python environment with all dependencies.

**a. Install `uv` (fast pip replacement):**

```bash
which uv || pip install uv --quiet
```

**b. Install vllm:**

```bash
uv pip install vllm --torch-backend=auto
```

**c. Install missing build dependency:**

```bash
uv pip install setuptools_scm
```

**d. Install vllm-omni in editable mode:**

```bash
# REPO must already be set (see §3.3)
uv pip install -e ${REPO} --no-build-isolation --no-cache-dir
```

> On the cluster (Slurm environment), activate your conda env first:
>
> ```bash
> source /path/to/miniconda3/etc/profile.d/conda.sh
> conda activate /path/to/conda_env
> module load cuda
> export CUDA_HOME=/usr/local/cuda
> export PATH=/usr/local/cuda/bin:$PATH
> export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
> ```

---

### 3.2 Weights

Two checkpoints are required:

| Checkpoint             | HuggingFace repo                           | Role                                                                |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------------------- |
| `moss-tts-local`       | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` | Stage 0 — AR model (Qwen3-1.7B global + local transformer, ~6.1 GB) |
| `moss-audio-tokenizer` | `OpenMOSS-Team/MOSS-Audio-Tokenizer`       | Stage 1 — CAT codec decoder                                         |

**Download:**

```bash
mkdir -p /path/to/weights

hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --local-dir /path/to/weights/moss-tts-local

hf download OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --local-dir /path/to/weights/moss-audio-tokenizer
```

**Set environment variables** (all commands below depend on these):

```bash
export MOSS_TTS_LOCAL_PATH=/path/to/weights/moss-tts-local
export MOSS_AUDIO_TOKENIZER_PATH=/path/to/weights/moss-audio-tokenizer
```

### 3.3 Code

```bash
git clone https://github.com/candylyt/vllm-omni.git
cd vllm-omni
git checkout moss-tts-local
export REPO=$(pwd)
```

### 3.4 One-time codec config patch

The `moss-audio-tokenizer` HuggingFace config contains type annotations incompatible
with newer versions of `transformers`. Run this patch once before first use
(idempotent — safe to re-run):

```bash
python3 - << 'EOF'
import os
path = os.path.join(os.environ["MOSS_AUDIO_TOKENIZER_PATH"],
                    "configuration_moss_audio_tokenizer.py")
bad = {
    "    sampling_rate: int\n",
    "    downsample_rate: int\n",
    "    causal_transformer_context_duration: float\n",
    "    encoder_kwargs: list[dict[str, Any]]\n",
    "    decoder_kwargs: list[dict[str, Any]]\n",
    "    quantizer_type: str\n",
    "    quantizer_kwargs: dict[str, Any]\n",
}
lines = open(path).readlines()
out = [l for l in lines if l not in bad]
if len(out) < len(lines):
    open(path, "w").writelines(out)
    print(f"Patched: removed {len(lines)-len(out)} lines")
else:
    print("Already patched, skipping")
EOF
```

---

## 4. Running Inference

The entry-point script handles both modes:

```
${REPO}/examples/offline_inference/moss_tts/benchmark_async_chunk.py
```

Switch between modes by changing `--mode` and `--stage-configs-path`.

---

### Option A — Interactive (direct)

#### async_chunk mode (streaming, recommended)

```bash
PYTHONPATH=${REPO} \
MOSS_AUDIO_TOKENIZER_PATH=${MOSS_AUDIO_TOKENIZER_PATH} \
CUDA_VISIBLE_DEVICES=0 \
python ${REPO}/examples/offline_inference/moss_tts/benchmark_async_chunk.py \
  --model              ${MOSS_TTS_LOCAL_PATH} \
  --mode               async \
  --stage-configs-path ${REPO}/vllm_omni/model_executor/stage_configs/moss_tts_async.yaml \
  --text               "The weather is so nice today." \
  --output-dir         ./output_async
```

#### sync batch mode

```bash
PYTHONPATH=${REPO} \
MOSS_AUDIO_TOKENIZER_PATH=${MOSS_AUDIO_TOKENIZER_PATH} \
CUDA_VISIBLE_DEVICES=0 \
python ${REPO}/examples/offline_inference/moss_tts/benchmark_async_chunk.py \
  --model              ${MOSS_TTS_LOCAL_PATH} \
  --mode               sync \
  --stage-configs-path ${REPO}/vllm_omni/model_executor/stage_configs/moss_tts.yaml \
  --text               "The weather is so nice today." \
  --output-dir         ./output_sync
```

#### Optional arguments

| Argument                | Default | Description                                                       |
| ----------------------- | ------- | ----------------------------------------------------------------- |
| `--init-sleep-seconds`  | `20`    | Wait time (s) after model load. Increase to `30` on slow storage. |
| `--batch-timeout`       | `5`     | Scheduler batch timeout (s).                                      |
| `--init-timeout`        | `5000`  | Inter-stage handshake timeout (steps).                            |
| `--shm-threshold-bytes` | `65536` | SharedMemory threshold in bytes.                                  |
| `--num-prompts`         | `1`     | Number of concurrent requests.                                    |

---

### Option B — Slurm job submission

The scripts below mirror the MOSS-TTS-Delay layout; only paths and config filenames change.

#### async_chunk mode — `run-bench-async.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=moss-local-bench-async
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=moss-local-bench-async-%j.out

set -euo pipefail

REPO=/path/to/vllm-omni
WEIGHTS_DIR=/path/to/weights

module purge
module load cuda

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda_env

# Codec config patch (idempotent)
python3 - << 'EOF'
import os
path = "/path/to/weights/moss-audio-tokenizer/configuration_moss_audio_tokenizer.py"
bad = {
    "    sampling_rate: int\n", "    downsample_rate: int\n",
    "    causal_transformer_context_duration: float\n",
    "    encoder_kwargs: list[dict[str, Any]]\n",
    "    decoder_kwargs: list[dict[str, Any]]\n",
    "    quantizer_type: str\n", "    quantizer_kwargs: dict[str, Any]\n",
}
lines = open(path).readlines()
out = [l for l in lines if l not in bad]
if len(out) < len(lines):
    open(path, "w").writelines(out)
    print(f"Patched: removed {len(lines)-len(out)} lines")
else:
    print("Already patched, skipping")
EOF

MPLCONFIGDIR=/tmp/matplotlib \
PYTHONPATH=${REPO} \
CUDA_VISIBLE_DEVICES=0 \
MOSS_AUDIO_TOKENIZER_PATH=${WEIGHTS_DIR}/moss-audio-tokenizer \
VLLM_LOGGING_LEVEL=INFO \
python ${REPO}/examples/offline_inference/moss_tts/benchmark_async_chunk.py \
  --model              ${WEIGHTS_DIR}/moss-tts-local \
  --mode               async \
  --stage-configs-path ${REPO}/vllm_omni/model_executor/stage_configs/moss_tts_async.yaml \
  --text               "The weather is so nice today." \
  --output-dir         ${REPO}/bench_output_async \
  --init-sleep-seconds 30
```

Submit:

```bash
sbatch run-bench-async.slurm
```

#### sync batch mode — `run-bench-sync.slurm`

Same script as above; change only three lines:

```bash
#SBATCH --job-name=moss-local-bench-sync
  --mode               sync \
  --stage-configs-path ${REPO}/vllm_omni/model_executor/stage_configs/moss_tts.yaml \
  --output-dir         ${REPO}/bench_output_sync \
```

Submit:

```bash
sbatch run-bench-sync.slurm
```

---

## 5. Expected Output and Log Interpretation

### 5.1 Terminal output

**async_chunk mode (representative):**

```
============================================================
  mode               : async
  total_time         : 4.812s
  audio_dur          : 2.240s
  RTF                : 2.1482  (slower than real-time)
  num_chunks         : 9  (streaming pipelined)
  first_chunk_latency: 1.349s  (time from generate() start to first audio sample available)
  wav                : ./output_async/<request_id>.wav
============================================================
```

**sync mode (representative):**

```
============================================================
  mode               : sync
  total_time         : 4.866s
  audio_dur          : 2.240s
  RTF                : 2.1723  (slower than real-time)
  num_chunks         : 1  (single batch)
  first_chunk_latency: 4.866s  (time from generate() start to first audio sample available)
  wav                : ./output_sync/<request_id>.wav
============================================================
```

**How to confirm the mode is working correctly:**

- `num_chunks > 1` — streaming pipeline is active (async mode only).
- `num_chunks = 1` — Stage 1 ran after Stage 0 finished (sync mode).
- In async mode, `first_chunk_latency` should be noticeably lower than
  `total_time` (the whole point of streaming). In sync mode they are equal.

> **Numbers are illustrative.** Your exact values depend on GPU, text length,
> and sampled AR length. Don't compare across different prompts or hardware.

### 5.2 Output file layout

```
output_async/
  ├── <request_id>.wav             # generated audio (24 kHz, mono)
  ├── _first_chunk/                # wall-clock stamps for first-chunk latency
  │    └── <request_id>.first_chunk.ts  (or first.first_chunk.ts fallback)
  └── bench_async.json             # summary: total_time_s, audio_dur_s, rtf,
                                   # num_chunks, first_chunk_latency_s

output_sync/
  ├── <request_id>.wav
  └── bench_sync.json
```

### 5.3 Reading TIMING logs

Set `VLLM_LOGGING_LEVEL=INFO` to get `[TIMING]` lines from Stage 1 that mark
exactly when the first non-empty waveform is produced.

**Grep command:**

```bash
grep "\[TIMING\]" moss-local-bench-async-JOBID.out
```

**Typical async output:**

```
[MossTTS Decoder][TIMING] non-empty wav produced at wall=<ts> request_id=run_1
[MossTTS Decoder][TIMING] non-empty wav produced at wall=<ts> request_id=run_1
[MossTTS Decoder][TIMING] non-empty wav produced at wall=<ts> request_id=run_1
... (one line per chunk; same run_N key across a single request)
```

`request_id=run_N` is a **synthetic key** minted on the Stage 1 side — see
[§7 Troubleshooting](#7-troubleshooting) for why. The fact that the same
`run_N` appears across chunks confirms the streaming KV-cache is active.

---

## 6. YAML Configuration Reference

The two YAML files differ in these key ways:

### `moss_tts_async.yaml` (async_chunk mode)

```yaml
async_chunk: true # global switch that enables streaming mode

stage_args:
  - stage_id: 0
    engine_args:
      async_scheduling: true
      # flush function on Stage 0: fires every codec_chunk_frames valid frames
      custom_process_next_stage_input_func: >-
        vllm_omni.model_executor.stage_input_processors.moss_tts.llm2decoder_async_chunk
    output_connectors:
      to_stage_1: connector_of_shared_memory # transfer via shared memory

  - stage_id: 1
    engine_args:
      async_scheduling: true

runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        codec_streaming: true
        codec_chunk_frames: 3 # frames per flush (smaller = lower first-chunk latency)
        codec_left_context_frames: 0 # MossTTSDecoderModel needs no left context
        connector_get_max_wait_first_chunk: 3000
        connector_get_max_wait: 300
```

### `moss_tts.yaml` (sync mode)

```yaml
async_chunk: false # streaming disabled

stage_args:
  - stage_id: 0
    engine_args:
      async_scheduling: false
    # no output_connectors, no custom_process_next_stage_input_func

  - stage_id: 1
    engine_args:
      async_scheduling: false
    # Stage 1 runs the full codes → waveform conversion in one call
    custom_process_input_func: >-
      vllm_omni.model_executor.stage_input_processors.moss_tts.llm2decoder
```

### Tunable parameters

| Parameter                           | Location             | Default          | Effect                                                                                   |
| ----------------------------------- | -------------------- | ---------------- | ---------------------------------------------------------------------------------------- |
| `codec_chunk_frames`                | async YAML connector | `3`              | Frames flushed per chunk. Smaller → lower first-chunk latency, more scheduling overhead. |
| `gpu_memory_utilization` (Stage 0)  | both YAMLs           | `0.3`            | VRAM fraction for the AR model.                                                          |
| `gpu_memory_utilization` (Stage 1)  | both YAMLs           | `0.25`           | VRAM fraction for the codec decoder.                                                     |
| `devices` (Stage 0 / Stage 1)       | `runtime.process`    | `"0"` / `"0"`    | GPU assignment. Set to different IDs for true parallelism.                               |
| `max_model_len` (Stage 0 / Stage 1) | `engine_args`        | `4096` / `18192` | KV-cache budget. Stage 1 cap bounds `max_tokens` (≤ `floor(18192/32)=568`).              |

**Single-GPU note:** the two `gpu_memory_utilization` values default to
`0.3 + 0.25 = 0.55`, leaving plenty of headroom on a 24 GB card. Bump them
if you see KV-cache pressure at long outputs.

**Dual-GPU parallel setup (reduces per-stage contention):**

```yaml
stage_args:
  - stage_id: 0
    runtime:
      devices: "0" # Stage 0 on GPU 0
  - stage_id: 1
    runtime:
      devices: "1" # Stage 1 on GPU 1
```

---

## 7. Troubleshooting

**`num_chunks = 1` in async mode**
The streaming pipeline is not active. Confirm you passed
`moss_tts_async.yaml` (not `moss_tts.yaml`) to `--stage-configs-path` and that
`async_chunk: true` is at the top of the file.

**Audio has glitches / clicks at chunk boundaries in async mode**
This means the codec's streaming KV-cache is **not** being engaged. Root
cause: `SharedMemoryConnector` currently strips the processor's `request_id`
and `finished` keys, so Stage 1 never learns which chunks belong to the same
request. The decoder works around this by synthesizing a stable key
(`run_N`) whose lifetime is bounded by `generated_len` resets — see
`MossTTSDecoderModel.forward()` in
`vllm_omni/model_executor/models/moss_tts/moss_tts_decoder.py`. If you
still hear boundary artifacts:

- Grep Stage 1 logs for `[MossTTS Decoder][TIMING]` lines; confirm the same
  `request_id=run_N` appears for all chunks of one request.
- If a fresh `run_N` is minted per chunk, the reset-detection heuristic fired
  incorrectly — file an issue with the `[DEBUG]` dump of
  `runtime_additional_information` keys.
- Long-term fix: have the connector forward `request_id` / `finished` so the
  synthetic-key workaround can be removed.

**`configuration_moss_audio_tokenizer.py` ImportError**
Run the codec config patch from [§3.4](#34-one-time-codec-config-patch).

**`PreTrainedConfig` / `PretrainedConfig` AttributeError on codec load**
The cached HF module import uses the legacy (transformers < 4.40) name.
The decoder injects an alias automatically — if you still see this,
check that `transformers` is importable from the active env.

**CUDA out of memory**

- Lower `gpu_memory_utilization` for one or both stages (sum < 0.94).
- Or move Stage 1 to a second GPU (see §6 dual-GPU setup).

**Stage 1 hangs / never receives a chunk**

- Verify `MOSS_AUDIO_TOKENIZER_PATH` is set correctly.
- Increase `connector_get_max_wait_first_chunk` (default 3000).
- Check Stage 0 is actually producing tokens (look for AR-side progress logs).

**Model load timeout / initialization failure**
Increase `--init-sleep-seconds` (try 30 or higher) on slow network storage.

**Audio sounds distorted even in sync mode**

- Check the AR text decoded in the terminal output — if it's garbage the
  AR stage is misconfigured (bad `stop_token_ids`, wrong prompt template,
  or a model-load mismatch). For MOSS-TTS-Local the stop set is
  `[151653, 151645, 151643]` (audio_end, im_end, endoftext).
- Ensure you patched `configuration_moss_audio_tokenizer.py` (§3.4).
