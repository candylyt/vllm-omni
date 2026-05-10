#!/usr/bin/env python3
"""Profile MOSS-TTS RTF, total latency, first chunk latency, and chunks."""

from __future__ import annotations

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
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni

from moss_tts_common import (
    SAMPLE_RATE,
    audio_duration_seconds,
    build_tts_prompt,
    concat_audio_tensor,
    find_stage_config,
    get_stop_ids,
    maybe_init_wandb,
    wandb_finish,
    wandb_log,
)


def get_wikitext_sentences(min_words: int, max_words: int, n: int, seed: int = 42) -> list[str]:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = [r["text"].strip() for r in ds if min_words < len(r["text"].split()) <= max_words]
    random.seed(seed)
    return random.sample(texts, min(n, len(texts)))


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
            max_tokens=500,
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


def run_one_sample(
    omni: Omni,
    ar_params: SamplingParams,
    decoder_params: SamplingParams,
    text: str,
    first_chunk_dir: str,
    output_dir: Path,
    sample_idx: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in Path(first_chunk_dir).glob("*.first_chunk.ts"):
        path.unlink()

    prompt = build_tts_prompt(text, omni.model)
    t0_wall = time.time()
    t0 = time.perf_counter()
    outputs = omni.generate([{"prompt": prompt}], [ar_params, decoder_params])
    total_time = time.perf_counter() - t0

    audio = None
    request_id = "first"
    num_chunks = 1
    for stage in outputs:
        if stage.final_output_type != "audio":
            continue
        output = stage.request_output
        request_id = output.request_id
        raw_audio = output.outputs[0].multimodal_output.get("audio")
        if isinstance(raw_audio, list):
            num_chunks = len(raw_audio)
        audio = concat_audio_tensor(raw_audio)
        break

    if audio is None:
        return {"ok": False, "text": text, "error": "no audio"}

    audio_np = audio.float().detach().cpu().numpy().reshape(-1)
    wav_path = output_dir / f"sample_{sample_idx:03d}.wav"
    sf.write(wav_path, audio_np, samplerate=SAMPLE_RATE)
    audio_dur = audio_duration_seconds(audio)

    first_chunk_latency = total_time
    for key in (request_id, "first"):
        ts_path = Path(first_chunk_dir) / f"{key}.first_chunk.ts"
        if ts_path.exists():
            first_chunk_latency = max(0.0, float(ts_path.read_text().strip()) - t0_wall)
            break

    return {
        "ok": True,
        "text": text,
        "total_time_s": total_time,
        "audio_dur_s": audio_dur,
        "rtf": total_time / audio_dur if audio_dur else 0.0,
        "num_chunks": num_chunks,
        "first_chunk_latency_s": first_chunk_latency,
        "wav": str(wav_path),
    }


def profile_mode(args: argparse.Namespace, mode: str, sentences: list[str]) -> dict:
    stage_cfg = find_stage_config(args.repo, args.model_type, mode)
    first_chunk_dir = Path(args.output_dir) / f"_first_chunk_{args.model_type}_{mode}"
    first_chunk_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MOSS_FIRST_CHUNK_DIR"] = str(first_chunk_dir)
    ar_params, decoder_params = sampling_params(args.model_type, args.model)

    run = maybe_init_wandb(
        project=args.wandb_project,
        name=f"moss-tts-{args.model_type}-{mode}-profile",
        config={**vars(args), "mode": mode},
    )
    omni = Omni(model=args.model, stage_configs_path=stage_cfg, init_sleep_seconds=args.init_sleep_seconds)
    omni.model = args.model

    samples = []
    for idx, text in enumerate(sentences):
        result = run_one_sample(
            omni,
            ar_params,
            decoder_params,
            text,
            str(first_chunk_dir),
            Path(args.output_dir) / f"raw_{args.model_type}_{mode}" / f"sample_{idx:03d}",
            idx,
        )
        samples.append(result)
        if result.get("ok"):
            wandb_log(run, {k: result[k] for k in ("rtf", "audio_dur_s", "first_chunk_latency_s", "num_chunks", "total_time_s")})
            print(
                f"[{mode} {idx + 1}/{len(sentences)}] "
                f"RTF={result['rtf']:.3f} FCL={result['first_chunk_latency_s']:.3f}s "
                f"chunks={result['num_chunks']}"
            )
        else:
            print(f"[{mode} {idx + 1}/{len(sentences)}] failed: {result.get('error')}")

    wandb_finish(run)
    ok_samples = [s for s in samples if s.get("ok")]
    summary = {
        "rtf_mean": float(np.mean([s["rtf"] for s in ok_samples])) if ok_samples else 0.0,
        "total_latency_s_mean": float(np.mean([s["total_time_s"] for s in ok_samples])) if ok_samples else 0.0,
        "first_chunk_latency_s_mean": float(np.mean([s["first_chunk_latency_s"] for s in ok_samples])) if ok_samples else 0.0,
        "num_chunks_mean": float(np.mean([s["num_chunks"] for s in ok_samples])) if ok_samples else 0.0,
        "num_ok": len(ok_samples),
        "num_total": len(samples),
    }
    return {"mode": mode, "summary": summary, "samples": samples}


def run_profile(args: argparse.Namespace) -> dict:
    os.environ["VLLM_LOGGING_LEVEL"] = os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")
    sentences = get_wikitext_sentences(args.min_words, args.max_words, args.n, args.seed)
    result = {
        "model_type": args.model_type,
        "n": args.n,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "modes": [],
    }
    for mode in args.modes:
        result["modes"].append(profile_mode(args, mode, sentences))
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
    parser.add_argument("--modes", nargs="+", default=["async", "sync"], choices=["async", "sync"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./profiling_results")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--init-sleep-seconds", type=int, default=30)
    parser.add_argument("--wandb-project", default="hpml-final-project")
    return parser.parse_args()


if __name__ == "__main__":
    run_profile(parse_args())
