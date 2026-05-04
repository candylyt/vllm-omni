#!/usr/bin/env python3
"""
Usage:
  PYTHONPATH=${REPO} \
  MOSS_AUDIO_TOKENIZER_PATH=${MOSS_AUDIO_TOKENIZER_PATH} \
  CUDA_VISIBLE_DEVICES=0 \
  VLLM_LOGGING_LEVEL=WARNING \
  python3 throughput_moss_tts_local.py \
    --model  ${MOSS_TTS_LOCAL_PATH} \
    --repo   ${REPO} \
    --output-dir ./throughput_results_vllm_local
"""

import wandb
import argparse
import copy
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoConfig
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni
from uni_functions import build_tts_prompt, get_stop_ids




FIXED_TEXT  = "The weather is so nice today and the birds are singing in the trees."
BATCH_SIZES = [1, 4, 8, 16, 64, 128, 256]
N_REPEATS   = 1
MAX_AR_TOKENS = 200
INIT_SLEEP_S  = 30


#generate yaml config for models to set max num seqs which is batch size and also can set gpu mem and max batched tokens
def make_yaml(base_yaml_path, max_num_seqs, gpu_mem_stage0=None, gpu_mem_stage1=None, max_num_batched_tokens=None):
    with open(base_yaml_path) as f:
        cfg = yaml.safe_load(f)

    for stage in cfg.get("stage_args", []):
        stage_id = stage.get("stage_id", 0)
        engine_args = stage.get("engine_args", {})

        engine_args["max_num_seqs"] = max_num_seqs

        # set the max num batched tokens for stage 0 and 1.
        if stage_id == 1:
            engine_args["max_num_batched_tokens"] = engine_args.get("max_num_batched_tokens", 18192)
        else:
            if max_num_batched_tokens is not None:
                engine_args["max_num_batched_tokens"] = max_num_batched_tokens
            else:
                engine_args["max_num_batched_tokens"] = max(engine_args.get("max_num_batched_tokens", 4096), max_num_seqs * 512)

        # per-stage GPU memory utilization overrides
        if stage_id == 0 and gpu_mem_stage0 is not None:
            engine_args["gpu_memory_utilization"] = gpu_mem_stage0

        if stage_id == 1 and gpu_mem_stage1 is not None:
            engine_args["gpu_memory_utilization"] = gpu_mem_stage1

        stage["engine_args"] = engine_args

    # make a temp yaml file for this config
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix=f"moss_bs{max_num_seqs}_")
    yaml.dump(cfg, tmp)
    tmp.flush()

    return tmp.name



def run_batch(model, repo, mode, bs, prompt, ar_params, decoder_params, init_sleep, gpu_mem_stage0=None, gpu_mem_stage1=None, max_num_batched_tokens=None):
    yaml = "moss_tts_async.yaml" if mode == "async" else "moss_tts.yaml"
    baseYAML = os.path.join(repo, f"vllm_omni/model_executor/stage_configs/{yaml}")
    pathYAML = make_yaml(baseYAML, max_num_seqs=bs, gpu_mem_stage0=gpu_mem_stage0, gpu_mem_stage1=gpu_mem_stage1, max_num_batched_tokens=max_num_batched_tokens)


    omni = Omni(model=model, stage_configs_path=pathYAML, init_sleep_seconds=init_sleep)
    prompts = [copy.deepcopy({"prompt": prompt}) for _ in range(bs)]

    t0 = time.perf_counter()
    omniOut = omni.generate(prompts, [ar_params, decoder_params])
    wall = time.perf_counter() - t0

    audioTotal = 0.0


    # calc total audio duration and save wavs for all samples in the batch
    for stage in omniOut:
        if stage.final_output_type != "audio":
            continue

        audioTensor = stage.request_output.outputs[0].multimodal_output["audio"]

        if audioTensor is None:
            continue

        if isinstance(audioTensor, list):
            audioTensor = torch.cat(audioTensor, dim=0)

        audioNP = audioTensor.float().detach().cpu().numpy().flatten()

        # compute audio duration and accumulate total audio duration for the batch. 24000 is the sample rate for moss
        audioTotal += len(audioNP) / 24000


    # cleanup the current omni instance and yaml file to free GPU memory before the next batch
    del omni
    os.unlink(pathYAML)
    torch.cuda.empty_cache()

    return wall, audioTotal



def run_benchmark(args):

    #initialize the wandbrun
    wandbRun = wandb.init(project="hpml-final-project", name= "vllm-local-throughput", 
                        config={"model": "vllm-local", "fixed_text": FIXED_TEXT,  "max_ar_tokens": MAX_AR_TOKENS,
                                "batch_sizes": BATCH_SIZES,
                                "n_repeats": N_REPEATS})

    # build prompts and sampling params
    prompt = build_tts_prompt(FIXED_TEXT, args.model)
    ar_stop_ids = get_stop_ids(args.model)

    ar_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=50, max_tokens=MAX_AR_TOKENS, seed=42, stop_token_ids=ar_stop_ids if ar_stop_ids else None)
    decoder_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=18192, seed=42, detokenize=False)

    print(f"  Mode: {args.mode.upper()}")
    print(f"  Text: \"{FIXED_TEXT[:55]}...\"")
    print("\n\n")
    print(f"  {'BS':>4}  {'wall(s)':>10}  {'audio(s)':>10}  {'tput(a/s)':>10}  {'speedup':>8}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

    # for each batch size and num of repeats run the benchmark and collect results for wall time, total audio duration, and throughput (audio duration / wall time)

    tp0 = None

    for bs in BATCH_SIZES:
        walls, audios, tps = [], [], []

        for _ in range(N_REPEATS):
            wall, audio = run_batch(model=args.model, repo=args.repo, mode=args.mode, bs=bs, prompt=prompt, ar_params=ar_params, decoder_params=decoder_params,
                                    init_sleep=args.init_sleep_seconds, gpu_mem_stage0=args.gpu_mem_stage0, gpu_mem_stage1=args.gpu_mem_stage1, max_num_batched_tokens=args.max_num_batched_tokens)

            tp = audio / wall if wall > 0 else 0
            walls.append(wall); audios.append(audio); tps.append(tp)

        meanWall = float(np.mean(walls))
        meanAudio = float(np.mean(audios))
        tp = float(np.mean(tps))

        #first run
        if tp0 is None:
            tp0 = tp

        speedup = tp / tp0 if tp0 > 0 else 1.0

        wandbRun.log({"bs": bs, "wall_s": meanWall, "audio_s": meanAudio, "throughput_audio_per_s": tp, "speedup": speedup})

        print(f"{bs:>4} {meanWall:>10.2f} {meanAudio:>10.2f} {tp:>10.3f} {speedup:>7.2f}x")


    wandbRun.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--mode", default="async", choices=["async", "sync"])
    p.add_argument("--init-sleep-seconds", type=int, default=INIT_SLEEP_S)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--gpu-mem-stage0", type=float, default=None)
    p.add_argument("--gpu-mem-stage1", type=float, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    args = p.parse_args()
    
    global BATCH_SIZES



    BATCH_SIZES = args.batch_sizes
    os.environ["VLLM_LOGGING_LEVEL"] = os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")

    run_benchmark(args)

if __name__ == "__main__":
    main()
