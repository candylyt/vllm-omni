# Reproduction Scripts

These scripts reproduce the MOSS-TTS/vLLM-Omni experiment tables reported in
the project README and final report.

The implementation branches are split:

- `moss-tts-local` or `moss-tts-local-optimization` contains the Local model
  integration and stage configs.
- `moss-tts-delay` contains the Delay model integration and stage configs.
- `main` contains this submission README, environment file, and deliverables.

Use a checkout/worktree of the relevant model branch as `--repo`, and run these
scripts from any branch after setting `PYTHONPATH` to that model-branch checkout.

## Environment

```bash
conda env create -f environment.yml
conda activate moss-tts-hpml
```

Download model weights:

```bash
mkdir -p "$HOME/weights"
hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --local-dir "$HOME/weights/moss-tts-local"
hf download OpenMOSS-Team/MOSS-TTS \
  --local-dir "$HOME/weights/moss-tts-delay"
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --local-dir "$HOME/weights/moss-audio-tokenizer"

export MOSS_TTS_LOCAL_PATH="$HOME/weights/moss-tts-local"
export MOSS_TTS_DELAY_PATH="$HOME/weights/moss-tts-delay"
export MOSS_AUDIO_TOKENIZER_PATH="$HOME/weights/moss-audio-tokenizer"
```

## Recommended Worktree Layout

```bash
git worktree add ../vllm-omni-local moss-tts-local
git worktree add ../vllm-omni-delay moss-tts-delay

# Install each model branch in the environment before running its experiments.
cd ../vllm-omni-local
uv pip install -e . --no-build-isolation --no-cache-dir

cd ../vllm-omni-delay
uv pip install -e . --no-build-isolation --no-cache-dir
```

## Throughput Tables

Throughput is reported as generated audio seconds per wall-clock second.

```bash
# Local, sync and async.
export REPO="$(realpath ../vllm-omni-local)"
export PYTHONPATH="$REPO"
python scripts/throughput_moss_tts.py \
  --model-type local \
  --model "$MOSS_TTS_LOCAL_PATH" \
  --repo "$REPO" \
  --mode sync \
  --batch-sizes 1 4 16 64 128 256 \
  --output-json results/local_sync_throughput.json

python scripts/throughput_moss_tts.py \
  --model-type local \
  --model "$MOSS_TTS_LOCAL_PATH" \
  --repo "$REPO" \
  --mode async \
  --batch-sizes 1 4 16 64 128 256 \
  --output-json results/local_async_throughput.json

# Delay, sync and async.
export REPO="$(realpath ../vllm-omni-delay)"
export PYTHONPATH="$REPO"
python scripts/throughput_moss_tts.py \
  --model-type delay \
  --model "$MOSS_TTS_DELAY_PATH" \
  --repo "$REPO" \
  --mode sync \
  --batch-sizes 1 4 16 64 128 256 \
  --output-json results/delay_sync_throughput.json

python scripts/throughput_moss_tts.py \
  --model-type delay \
  --model "$MOSS_TTS_DELAY_PATH" \
  --repo "$REPO" \
  --mode async \
  --batch-sizes 1 4 16 64 128 256 \
  --output-json results/delay_async_throughput.json
```

## RTF/FCL Table

```bash
export REPO="$(realpath ../vllm-omni-local)"
export PYTHONPATH="$REPO"
python scripts/profile_moss_tts.py \
  --model-type local \
  --model "$MOSS_TTS_LOCAL_PATH" \
  --repo "$REPO" \
  --modes async sync \
  --n 20 \
  --min-words 10 \
  --max-words 60 \
  --output-json results/local_profile.json

export REPO="$(realpath ../vllm-omni-delay)"
export PYTHONPATH="$REPO"
python scripts/profile_moss_tts.py \
  --model-type delay \
  --model "$MOSS_TTS_DELAY_PATH" \
  --repo "$REPO" \
  --modes async sync \
  --n 20 \
  --min-words 10 \
  --max-words 60 \
  --output-json results/delay_profile.json
```

## Generate Markdown/LaTeX Tables

```bash
python scripts/summarize_moss_tts_results.py \
  --throughput-json results/local_sync_throughput.json \
  --throughput-json results/local_async_throughput.json \
  --throughput-json results/delay_sync_throughput.json \
  --throughput-json results/delay_async_throughput.json \
  --profile-json results/local_profile.json \
  --profile-json results/delay_profile.json \
  --out-dir results/tables
```

The summary script emits Markdown tables for README-style reporting and LaTeX
table bodies suitable for the IEEE report.

## Notes

- HuggingFace baseline values in the report were collected separately with
  notebooks on `metric-branch` (`HPML-metrics/hf_profile.ipynb` and
  `HPML-metrics/hf_throughput.ipynb`). This folder contains the vLLM-Omni
  reproduction path and table-generation utilities.
- Set `WANDB_DISABLED=true` to avoid W&B logging.
- For the Delay async run, batch sizes 128 and 256 may hit memory pressure on a
  single A100 80GB and fall back to lower effective concurrency, matching the
  reported limitation.
