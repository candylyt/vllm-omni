#!/usr/bin/env bash
set -euo pipefail

# Run all vLLM-Omni MOSS-TTS experiments used for the reported tables.
#
# Required environment variables:
#   MOSS_TTS_LOCAL_PATH
#   MOSS_TTS_DELAY_PATH
#   MOSS_AUDIO_TOKENIZER_PATH
#
# Optional environment variables:
#   LOCAL_REPO   path to a checkout/worktree of moss-tts-local
#   DELAY_REPO   path to a checkout/worktree of moss-tts-delay
#   OUT_DIR      output directory, default: results/reported
#   WANDB_DISABLED=true to disable W&B logging

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_REPO="${LOCAL_REPO:-${ROOT_DIR}}"
DELAY_REPO="${DELAY_REPO:-${ROOT_DIR}}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/reported}"

mkdir -p "${OUT_DIR}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
}

require_env MOSS_TTS_LOCAL_PATH
require_env MOSS_TTS_DELAY_PATH
require_env MOSS_AUDIO_TOKENIZER_PATH

run_local() {
  export PYTHONPATH="${LOCAL_REPO}"
  python "${ROOT_DIR}/scripts/throughput_moss_tts.py" \
    --model-type local \
    --model "${MOSS_TTS_LOCAL_PATH}" \
    --repo "${LOCAL_REPO}" \
    --mode "$1" \
    --batch-sizes 1 4 16 64 128 256 \
    --output-json "${OUT_DIR}/local_${1}_throughput.json"
}

run_delay() {
  export PYTHONPATH="${DELAY_REPO}"
  python "${ROOT_DIR}/scripts/throughput_moss_tts.py" \
    --model-type delay \
    --model "${MOSS_TTS_DELAY_PATH}" \
    --repo "${DELAY_REPO}" \
    --mode "$1" \
    --batch-sizes 1 4 16 64 128 256 \
    --output-json "${OUT_DIR}/delay_${1}_throughput.json"
}

run_profile_local() {
  export PYTHONPATH="${LOCAL_REPO}"
  python "${ROOT_DIR}/scripts/profile_moss_tts.py" \
    --model-type local \
    --model "${MOSS_TTS_LOCAL_PATH}" \
    --repo "${LOCAL_REPO}" \
    --modes async sync \
    --n 20 \
    --min-words 10 \
    --max-words 60 \
    --output-dir "${OUT_DIR}/local_profile_raw" \
    --output-json "${OUT_DIR}/local_profile.json"
}

run_profile_delay() {
  export PYTHONPATH="${DELAY_REPO}"
  python "${ROOT_DIR}/scripts/profile_moss_tts.py" \
    --model-type delay \
    --model "${MOSS_TTS_DELAY_PATH}" \
    --repo "${DELAY_REPO}" \
    --modes async sync \
    --n 20 \
    --min-words 10 \
    --max-words 60 \
    --output-dir "${OUT_DIR}/delay_profile_raw" \
    --output-json "${OUT_DIR}/delay_profile.json"
}

run_local sync
run_local async
run_delay sync
run_delay async
run_profile_local
run_profile_delay

python "${ROOT_DIR}/scripts/summarize_moss_tts_results.py" \
  --throughput-json "${OUT_DIR}/local_sync_throughput.json" \
  --throughput-json "${OUT_DIR}/local_async_throughput.json" \
  --throughput-json "${OUT_DIR}/delay_sync_throughput.json" \
  --throughput-json "${OUT_DIR}/delay_async_throughput.json" \
  --profile-json "${OUT_DIR}/local_profile.json" \
  --profile-json "${OUT_DIR}/delay_profile.json" \
  --out-dir "${OUT_DIR}/tables"

echo "Done. Tables written to ${OUT_DIR}/tables"
