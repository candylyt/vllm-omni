"""
Usage:
  PYTHONPATH=${REPO} \
  MOSS_AUDIO_TOKENIZER_PATH=${MOSS_AUDIO_TOKENIZER_PATH} \
  CUDA_VISIBLE_DEVICES=0 \
  python3 moss_tts_profile.py \
    --model-type  local \
    --model       ${MOSS_TTS_LOCAL_PATH} \
    --repo        ${REPO} \
    --n           20 \
    --min-words   10 \
    --max-words   50 \
    --output-dir  ./profiling_results

  # or for the delay model:
  python3 moss_tts_profile.py \
    --model-type  delay \
    --model       ${MOSS_TTS_DELAY_PATH} \
    --repo        ${REPO} \
    --n           2 \
    --min-words   10 \
    --max-words   60 \
    --output-dir  ./profiling_results
"""
import wandb

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni
from uni_functions import build_tts_prompt, get_stop_ids

# load sentences from wikitext
def get_wikitext_sentences(min_words, max_words, n, seed=42):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    texts = [r["text"].strip() for r in ds
             if min_words < len(r["text"].split()) <= max_words]
            
    random.seed(seed)
    selected = random.sample(texts, min(n, len(texts)))

    return selected

# run a sample and return timing results for total time, audio duration, RTF, and first chunk latency
def run_one_sample(omni, ar_params, decoder_params, text, first_chunk_dir, output_dir, sample_idx):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # cleanup any existing first chunk ts files for this sample idx in the first chunk dir to avoid confusion with leftover files from previous runs
    if first_chunk_dir and os.path.exists(first_chunk_dir):
        for f in os.listdir(first_chunk_dir):
            if f.endswith(".first_chunk.ts"):
                os.remove(os.path.join(first_chunk_dir, f))

    prompt = build_tts_prompt(text, omni.model)
    prompts = [{"prompt": prompt}]

    t0Wall = time.time()
    t0 = time.perf_counter()

    omniOut = omni.generate(prompts, [ar_params, decoder_params])

    audioTensor = None
    t1 = time.perf_counter()
    totalTime = t1 - t0

    # for each stage in the output, find the one with audio output and get the audio tensor. if multiple chunks, concatenate them
    for stage in omniOut:

        if stage.final_output_type != "audio":
            continue

        output = stage.request_output
        request_id = output.request_id
        audioTensor = output.outputs[0].multimodal_output["audio"]
        numChunks = 1

        if isinstance(audioTensor, list):
            numChunks = len(audioTensor)
            audioTensor = torch.cat(audioTensor, dim=0)

        break
    
    #if no audio returned we failed
    if audioTensor is None:
        return {"ok": False, "error": "no audio"}
    
    # covnert audio to a numpy and save as wav
    audioNP = audioTensor.float().detach().cpu().numpy()

    if audioNP.ndim > 1:
        audioNP = audioNP.flatten()

    # compute audio duration and RTF. 24000 is the sample rate for moss
    audioDur = len(audioNP) / 24000
    rtf = totalTime / audioDur

    wavPath = os.path.join(output_dir, f"sample_{sample_idx:03d}.wav")
    sf.write(wavPath, audioNP, samplerate=24000)

    # calc FCL 
    fcl = totalTime

    if first_chunk_dir:
        for key in (request_id, "first"):
            ts_path = os.path.join(first_chunk_dir, f"{key}.first_chunk.ts")
            
            if os.path.exists(ts_path):
                with open(ts_path) as f:
                    ts = float(f.read().strip())

                #FCL = fc ts - start ts
                fcl = max(0.0, ts - t0Wall)

                break

    return {"ok": True,
            "text": text,
            "total_time_s": totalTime,
            "audio_dur_s": audioDur,
            "rtf": rtf,
            "num_chunks": numChunks,
            "first_chunk_latency_s": fcl,
            "wav": wavPath}


# run the samples in either sync or async mode and collect results
def profile_mode(mode, sentences, model, repo, output_base, init_sleep, mtype):
    # get correct config yaml
    if mtype == "delay":
        yaml = "moss_tts_delay_async.yaml" if mode == "async" else "moss_tts_delay.yaml"
        stage_cfg = os.path.join(repo, f"vllm_omni/model_executor/stage_configs/{yaml}")
        
        if not os.path.exists(stage_cfg):
            yaml = "moss_tts_async.yaml" if mode == "async" else "moss_tts.yaml"
            stage_cfg = os.path.join(repo, f"vllm_omni/model_executor/stage_configs/{yaml}")
    else:
        yaml = "moss_tts_async.yaml" if mode == "async" else "moss_tts.yaml"
        stage_cfg = os.path.join(repo, f"vllm_omni/model_executor/stage_configs/{yaml}")

    out = output_base / f"raw_{mode}"
    out.mkdir(parents=True, exist_ok=True)

    first_chunk_dir = str(output_base / f"_first_chunk_{mode}")
    os.makedirs(first_chunk_dir, exist_ok=True)
    os.environ["MOSS_FIRST_CHUNK_DIR"] = first_chunk_dir

    ar_stop_ids = get_stop_ids(model)
    ar_params = None

    wandbRun = wandb.init(project="hpml-final-project", name= f"vllm-local-inference-profile-{mode}", 
                          config={"model": model, "mode": mode})

    # set sampling params for AR stage for model
    if mtype == "delay":
        ar_params = SamplingParams(temperature=1.5, top_p=1.0, top_k=50, max_tokens=900, seed=42, repetition_penalty=1.0)
    else:
        ar_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=50, max_tokens=500, seed=42, stop_token_ids=ar_stop_ids if ar_stop_ids else None)

    decoder_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=18192, seed=42, detokenize=False,)

    omni = Omni(model=model, stage_configs_path=stage_cfg, init_sleep_seconds=init_sleep)
    omni.model = model

    # collect res for each sample

    n = len(sentences)

    for i, text in enumerate(sentences):
        print(f"[{i+1}/{n}] {text[:60]}  ", end=" ", flush=True)

        # run thesample
        run = run_one_sample(omni=omni, ar_params=ar_params, decoder_params=decoder_params, text=text, first_chunk_dir=first_chunk_dir, output_dir=str(out / f"sample_{i:03d}"), sample_idx=i)

        if run["ok"]:
            print(f"RTF={run['rtf']:.3f}  dur={run['audio_dur_s']:.2f}s fcl={run['first_chunk_latency_s']:.2f}s")
            wandbRun.log({"rtf": run["rtf"], "audio_dur_s": run["audio_dur_s"], "first_chunk_latency_s": run["first_chunk_latency_s"], "num_chunks": run["num_chunks"], 'total_time_s': run["total_time_s"]})
        else:
            print(f"Oopsies!  {run['error'][:60]}")

    wandbRun.finish()

def main():
    p = argparse.ArgumentParser(description="Profile MOSS-TTS via Omni")
    p.add_argument("--model-type", default="local", choices=["local", "delay"])
    p.add_argument("--model", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--min-words", type=int, default=10)
    p.add_argument("--max-words", type=int, default=50)
    p.add_argument("--output-dir", default="./profiling_results")
    p.add_argument("--init-sleep-seconds", type=int, default=30)
    p.add_argument("--modes", nargs="+", default=["async", "sync"], choices=["async", "sync"])
    
    args = p.parse_args()

    os.environ["VLLM_LOGGING_LEVEL"] = os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sentences = get_wikitext_sentences(args.min_words, args.max_words, args.n)

    for mode in args.modes:
        profile_mode(mode=mode, sentences=sentences, model=args.model,  
                     repo=args.repo, output_base=out, init_sleep=args.init_sleep_seconds, mtype=args.model_type)


if __name__ == "__main__":
    main()
