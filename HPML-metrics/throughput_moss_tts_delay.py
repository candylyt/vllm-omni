#!/usr/bin/env python3
"""
Usage:
  PYTHONPATH=${REPO} \
  MOSS_AUDIO_TOKENIZER_PATH=${MOSS_AUDIO_TOKENIZER_PATH} \
  CUDA_VISIBLE_DEVICES=0 \
  VLLM_LOGGING_LEVEL=WARNING \
  python3 throughput_moss_tts_delay.py \
    --model  ${MOSS_TTS_DELAY_PATH} \
    --repo   ${REPO} \
    --output-dir ./throughput_results_vllm_delay
"""
import wandb

import argparse
import copy
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni
from uni_functions import build_tts_prompt


FIXED_TEXT = "The weather is so nice today and the birds are singing in the trees."
N_REPEATS = 1
MAX_AR_TOKENS = 900
MAX_DECODER_TOKENS = 32768
INIT_SLEEP_S = 30


def make_yaml(
    base_yaml_path,
    max_num_seqs,
    gpu_mem_stage0=None,
    gpu_mem_stage1=None,
    max_num_batched_tokens=None,
):
    with open(base_yaml_path) as f:
        cfg = yaml.safe_load(f)

    for stage in cfg.get("stage_args", []):
        stage_id = stage.get("stage_id", 0)
        engine_args = stage.get("engine_args", {})

        engine_args["max_num_seqs"] = max_num_seqs

        if stage_id == 1:
            engine_args["max_num_batched_tokens"] = engine_args.get(
                "max_num_batched_tokens", MAX_DECODER_TOKENS
            )
        else:
            if max_num_batched_tokens is not None:
                engine_args["max_num_batched_tokens"] = max_num_batched_tokens
            else:
                engine_args["max_num_batched_tokens"] = max(
                    engine_args.get("max_num_batched_tokens", 4096),
                    max_num_seqs * 1024,
                )

        if stage_id == 0 and gpu_mem_stage0 is not None:
            engine_args["gpu_memory_utilization"] = gpu_mem_stage0

        if stage_id == 1 and gpu_mem_stage1 is not None:
            engine_args["gpu_memory_utilization"] = gpu_mem_stage1

        stage["engine_args"] = engine_args

    runtime_defaults = cfg.get("runtime", {}).get("defaults")
    if isinstance(runtime_defaults, dict) and "max_inflight" in runtime_defaults:
        runtime_defaults["max_inflight"] = max_num_seqs

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        prefix=f"moss_delay_bs{max_num_seqs}_",
    )
    yaml.dump(cfg, tmp)
    tmp.flush()
    return tmp.name


def run_batch(
    model,
    repo,
    mode,
    bs,
    prompt,
    ar_params,
    decoder_params,
    init_sleep,
    output_dir,
    gpu_mem_stage0=None,
    gpu_mem_stage1=None,
    max_num_batched_tokens=None,
):
    yaml_name = "moss_tts_delay_async.yaml" if mode == "async" else "moss_tts_delay.yaml"
    base_yaml = os.path.join(
        repo, "vllm_omni", "model_executor", "stage_configs", yaml_name
    )
    if not os.path.exists(base_yaml):
        raise FileNotFoundError(f"Missing delay stage config: {base_yaml}")

    path_yaml = make_yaml(
        base_yaml,
        max_num_seqs=bs,
        gpu_mem_stage0=gpu_mem_stage0,
        gpu_mem_stage1=gpu_mem_stage1,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    omni = Omni(model=model, stage_configs_path=path_yaml, init_sleep_seconds=init_sleep)
    prompts = [copy.deepcopy({"prompt": prompt}) for _ in range(bs)]

    t0 = time.perf_counter()
    omniOut = omni.generate(prompts, [ar_params, decoder_params])
    wall = time.perf_counter() - t0

    audioDur = 0.0
    idx = 0

    for stage in omniOut:
        if stage.final_output_type != "audio":
            continue

        audioTensor = stage.request_output.outputs[0].multimodal_output["audio"]

        if isinstance(audioTensor, list):
            audioTensor = torch.cat(audioTensor, dim=0)

        audioNP = audioTensor.float().detach().cpu().numpy().flatten()
        audioDur += len(audioNP) / 24000

        sf.write(
            os.path.join(output_dir, f"sample_{idx:03d}.wav"),
            audioNP,
            samplerate=24000,
        )
        idx += 1

    del omni
    os.unlink(path_yaml)
    torch.cuda.empty_cache()

    return wall, audioDur


def run_benchmark(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prompt = build_tts_prompt(FIXED_TEXT, args.model)
    ar_params = SamplingParams(
        temperature=1.5,
        top_p=1.0,
        top_k=50,
        max_tokens=MAX_AR_TOKENS,
        seed=42,
        repetition_penalty=1.0,
    )
    decoder_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_DECODER_TOKENS,
        seed=42,
        detokenize=False,
    )

    wandbRun = wandb.init(
        project="hpml-final-project",
        name=f"moss-tts-delay-{args.mode}-throughput",
        config={
            "model": "moss-tts-delay",
            "mode": args.mode,
            "fixed_text": FIXED_TEXT,
            "max_ar_tokens": MAX_AR_TOKENS,
            "max_decoder_tokens": MAX_DECODER_TOKENS,
            "batch_sizes": BATCH_SIZES,
            "n_repeats": N_REPEATS})

    print(f"  Model: MOSS-TTS-DELAY")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Text: \"{FIXED_TEXT[:55]}...\"")
    print("\n\n")
    print(
        f"  {'BS':>4}  {'wall(s)':>10}  {'audio(s)':>10}  "
        f"{'tput(a/s)':>10}  {'speedup':>8}"
    )
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    res = []
    tp0 = None

    for bs in BATCH_SIZES:
        walls, audios, tps = [], [], []

        for _ in range(N_REPEATS):
            wall, audio = run_batch(
                model=args.model,
                repo=args.repo,
                mode=args.mode,
                bs=bs,
                prompt=prompt,
                ar_params=ar_params,
                decoder_params=decoder_params,
                init_sleep=args.init_sleep_seconds,
                output_dir=str(out / f"raw_bs{bs}"),
                gpu_mem_stage0=args.gpu_mem_stage0,
                gpu_mem_stage1=args.gpu_mem_stage1,
                max_num_batched_tokens=args.max_num_batched_tokens,
            )

            tp = audio / wall if wall > 0 else 0
            walls.append(wall)
            audios.append(audio)
            tps.append(tp)

        mean_wall = float(np.mean(walls))
        mean_audio = float(np.mean(audios))
        tp = float(np.mean(tps))

        if tp0 is None:
            tp0 = tp

        speedup = tp / tp0 if tp0 > 0 else 1.0
        res.append(
            {
                "bs": bs,
                "wall_s": mean_wall,
                "audio_s": mean_audio,
                "throughput_audio_per_s": tp,
                "speedup": speedup,
            }
        )

        print(f"{bs:>4} {mean_wall:>10.2f} {mean_audio:>10.2f} {tp:>10.3f} {speedup:>7.2f}x")
        wandbRun.log({"batch_size": bs, "wall_s": mean_wall, "audio_s": mean_audio, "throughput_audio_per_s": tp, "speedup": speedup})

    wandbRun.finish()

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--mode", default="async", choices=["async", "sync"])
    parser.add_argument("--output-dir", default="./throughput_results_vllm_delay")
    parser.add_argument("--init-sleep-seconds", type=int, default=INIT_SLEEP_S)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 16, 32, 64, 128, 256])
    parser.add_argument("--gpu-mem-stage0", type=float, default=None)
    parser.add_argument("--gpu-mem-stage1", type=float, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    args = parser.parse_args()

    global BATCH_SIZES
    BATCH_SIZES = args.batch_sizes
    os.environ["VLLM_LOGGING_LEVEL"] = os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")

    run_benchmark(args)


if __name__ == "__main__":
    main()
