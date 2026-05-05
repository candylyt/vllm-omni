# MOSS-TTS-Local: Deployment & Profiling Runbook (vLLM 0.19.1)
## 1. Instance Requirements
* **GPU:** 80 GB+ VRAM (A100 80 GB recommended)
* **Disk:** 100 GB free
* **VLLM:** 0.19.1

NEED VERSION 12.8 CUDA OR ELSE DOES NOT RUN

## 2. Environment Setup
### 2.1 Create Env
```bash
conda create -n moss-tts python=3.11 -y
conda activate moss-tts
```

### 2.2 Configure CUDA Paths
Add these to `~/.bashrc`:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
source ~/.bashrc
```

## 3. Python Dependencies (vLLM 0.19.1)
```bash
pip install uv --quiet
pip install datasets
pip install wandb
pip install soundfile
uv pip install vllm==0.19.1 --torch-backend=auto
uv pip install setuptools_scm
```

## 4. Clone & Install vllm-omni
```bash
git clone https://github.com/candylyt/vllm-omni.git
cd vllm-omni
git checkout moss-tts-local
export REPO=$(pwd)
 
uv pip install -e ${REPO} --no-build-isolation --no-cache-dir
```

## 5. Download Model Weights
```bash
pip install huggingface_hub
mkdir -p $HOME/weights

# --- Local/Async Model Weights ---
hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer --local-dir $HOME/weights/moss-tts-local

# --- Delay Model Weights ---
hf download OpenMOSS-Team/MOSS-TTS --local-dir $HOME/weights/moss-tts-delay

# --- Audio Tokenizer (Codec) Weights ---
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir $HOME/weights/moss-audio-tokenizer

# Persist Environment Variables
export MOSS_TTS_LOCAL_PATH=$HOME/weights/moss-tts-local
export MOSS_TTS_DELAY_PATH=$HOME/weights/moss-tts-delay
export MOSS_AUDIO_TOKENIZER_PATH=$HOME/weights/moss-audio-tokenizer
```

## 6. One-Time Codec Config Patch
```bash
python3 - << 'EOF'
import os
path = os.path.join(os.environ['MOSS_AUDIO_TOKENIZER_PATH'], 'configuration_moss_audio_tokenizer.py')
bad = {
    '    sampling_rate: int\n',
    '    downsample_rate: int\n',
    '    causal_transformer_context_duration: float\n',
    '    encoder_kwargs: list[dict[str, Any]]\n',
    '    decoder_kwargs: list[dict[str, Any]]\n',
    '    quantizer_type: str\n',
    '    quantizer_kwargs: dict[str, Any]\n',
}
lines = open(path).readlines()
out = [l for l in lines if l not in bad]
if len(out) < len(lines):
    open(path, 'w').writelines(out)
    print(f'Patched: removed {len(lines)-len(out)} lines')
else:
    print('Already patched, skipping')
EOF
```

## 7. Critical: Fix CUDA Library Path (Vast.ai)
```bash
# Find torch's bundled nvjitlink lib
TORCH_LIB_PATH=$(find /venv/moss-tts/lib -name 'libnvJitLink*' 2>/dev/null \
  | head -1 | xargs dirname)
echo "Found: $TORCH_LIB_PATH"
 
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH

echo 'export LD_LIBRARY_PATH=/venv/moss-tts/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x  True


```

## 8. Profiling Execution
```bash
# Local/ Delay would be the same thing but DELAY path and delay model type
PYTHONPATH=${REPO} \
MOSS_AUDIO_TOKENIZER_PATH=${MOSS_AUDIO_TOKENIZER_PATH} \
CUDA_VISIBLE_DEVICES=0 \
python moss_tts_profile.py \
  --model    ${MOSS_TTS_LOCAL_PATH} \
  --model-type local \
  --repo     ${REPO} \
  --n        20 \
  --min-words 10 \
  --max-words 50 \
  --output-dir ./profiling_results_local \
  --init-sleep-seconds 30
```