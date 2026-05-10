#!/usr/bin/env python3
"""Throughput benchmark for MOSS-TTS Local/Delay in vLLM-Omni.

Throughput is generated audio seconds per wall-clock second.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni

from moss_tts_common import (
    FIXED_TEXT,
    audio_duration_seconds,
    build_tts_prompt,
    concat_audio_tensor,
    find_stage_config,
    get_stop_ids,
    maybe_init_wandb,
    wandb_finish,
    wandb_log,
)


def make_yaml(
    base_yaml_path: str,
    max_num_seqs: int,
    model_type: str,
    gpu_mem_stage0: float | None = None,
    gpu_mem_stage1: float | None = None,
    max_num_batched_tokens: int | None = None,
) -> str:
    with open(base_yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    decoder_default = 32768 if model_type == "delay" else 18192
    stage0_scale = 1024 if model_type == "delay" else 512

    for stage in cfg.get("stage_args", []):
        stage_id = int(stage.get("stage_id", 0))
        engine_args = stage.get("engine_args", {})
        engine_args["max_num_seqs"] = max_num_seqs
        if stage_id == 1:
            engine_args["max_num_batched_tokens"] = engine_args.get(
                "max_num_batched_tokens",
                decoder_default,
            )
        else:
            engine_args["max_num_batched_tokens"] = (
                max_num_batched_tokens
                if max_num_batched_tokens is not None
                else max(engine_args.get("max_num_batched_tokens", 4096), max_num_seqs * stage0_scale)
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
        prefix=f"moss_{model_type}_bs{max_num_seqs}_",
        encoding="utf-8",
    )
    yaml.safe_dump(cfg, tmp)
    tmp.flush()
    return tmp.name


def sampling_params(model_type: str, model_path: str) -> tuple[SamplingParams, SamplingParams]:
    if model_type == "delay":
        ar_params = SamplingParams(
            temperature=1.5,
            top_p=1.0,
            top_k=50,
            max_tokens=900,
            seed=42,
            repetition_penalty=1.0,
        )
        decoder_max = 32768
    else:
        ar_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            max_tokens=200,
            seed=42,
            stop_token_ids=get_stop_ids(model_path) or None,
        )
        decoder_max = 18192
    decoder_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=decoder_max,
        seed=42,
        detokenize=False,
    )
    return ar_params, decoder_params


def run_batch(args: argparse.Namespace, batch_size: int, prompt: str) -> tuple[float, float]:
    base_yaml = find_stage_config(args.repo, args.model_type, args.mode)
    path_yaml = make_yaml(
        base_yaml,
        max_num_seqs=batch_size,
        model_type=args.model_type,
        gpu_mem_stage0=args.gpu_mem_stage0,
        gpu_mem_stage1=args.gpu_mem_stage1,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    ar_params, decoder_params = sampling_params(args.model_type, args.model)

    try:
        omni = Omni(model=args.model, stage_configs_path=path_yaml, init_sleep_seconds=args.init_sleep_seconds)
        prompts = [copy.deepcopy({"prompt": prompt}) for _ in range(batch_size)]
        t0 = time.perf_counter()
        outputs = omni.generate(prompts, [ar_params, decoder_params])
        wall = time.perf_counter() - t0

        audio_total = 0.0
        for stage in outputs:
            if stage.final_output_type != "audio":
                continue
            audio = concat_audio_tensor(stage.request_output.outputs[0].multimodal_output.get("audio"))
            if audio is not None:
                audio_total += audio_duration_seconds(audio)
        return wall, audio_total
    finally:
        try:
            del omni  # type: ignore[name-defined]
        except Exception:
            pass
        if os.path.exists(path_yaml):
            os.unlink(path_yaml)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_benchmark(args: argparse.Namespace) -> dict:
    os.environ["VLLM_LOGGING_LEVEL"] = os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")
    prompt = build_tts_prompt(args.text, args.model)
    run = maybe_init_wandb(
        project=args.wandb_project,
        name=f"moss-tts-{args.model_type}-{args.mode}-throughput",
        config=vars(args),
    )

    rows = []
    baseline_tp = None
    print(f"model_type={args.model_type} mode={args.mode} repeats={args.repeats}")
    print(f"{'BS':>5} {'wall_s':>10} {'audio_s':>10} {'tput':>10} {'speedup':>9}")

    for bs in args.batch_sizes:
        walls: list[float] = []
        audios: list[float] = []
        tputs: list[float] = []
        for _ in range(args.repeats):
            wall, audio = run_batch(args, bs, prompt)
            throughput = audio / wall if wall > 0 else 0.0
            walls.append(wall)
            audios.append(audio)
            tputs.append(throughput)

        row = {
            "batch_size": bs,
            "wall_s_mean": float(np.mean(walls)),
            "audio_s_mean": float(np.mean(audios)),
            "throughput_audio_per_s": float(np.mean(tputs)),
            "wall_s_values": walls,
            "audio_s_values": audios,
            "throughput_values": tputs,
        }
        if baseline_tp is None:
            baseline_tp = row["throughput_audio_per_s"]
        row["speedup_vs_bs1"] = row["throughput_audio_per_s"] / baseline_tp if baseline_tp else 1.0
        rows.append(row)
        wandb_log(run, row)
        print(
            f"{bs:>5} {row['wall_s_mean']:>10.2f} {row['audio_s_mean']:>10.2f} "
            f"{row['throughput_audio_per_s']:>10.3f} {row['speedup_vs_bs1']:>8.2f}x"
        )

    wandb_finish(run)
    result = {
        "model_type": args.model_type,
        "mode": args.mode,
        "text": args.text,
        "repeats": args.repeats,
        "rows": rows,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, choices=["local", "delay"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--mode", required=True, choices=["async", "sync"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--text", default=FIXED_TEXT)
    parser.add_argument("--init-sleep-seconds", type=int, default=30)
    parser.add_argument("--gpu-mem-stage0", type=float, default=None)
    parser.add_argument("--gpu-mem-stage1", type=float, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--wandb-project", default="hpml-final-project")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
