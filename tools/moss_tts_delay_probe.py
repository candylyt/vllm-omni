#!/usr/bin/env python3

import argparse
import os
import sys
import time
import yaml

import torch
from vllm import SamplingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from throughput_moss_tts_delay import (
    FIXED_TEXT,
    MAX_AR_TOKENS,
    MAX_DECODER_TOKENS,
    build_tts_prompt,
    make_yaml,
)
from vllm_omni.entrypoints.omni import Omni


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--mode", default="async", choices=["async", "sync"])
    parser.add_argument("--init-sleep-seconds", type=int, default=5)
    parser.add_argument("--max-ar-tokens", type=int, default=MAX_AR_TOKENS)
    parser.add_argument("--initial-codec-chunk-frames", type=int, default=None)
    parser.add_argument("--codec-chunk-frames", type=int, default=None)
    args = parser.parse_args()

    yaml_name = "moss_tts_delay_async.yaml" if args.mode == "async" else "moss_tts_delay.yaml"
    yaml_path = make_yaml(
        os.path.join(
            args.repo,
            "vllm_omni",
            "model_executor",
            "stage_configs",
            yaml_name,
        ),
        max_num_seqs=1,
    )
    if args.mode == "async" and (
        args.initial_codec_chunk_frames is not None or args.codec_chunk_frames is not None
    ):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        extra = cfg["runtime"]["connectors"]["connector_of_shared_memory"]["extra"]
        if args.initial_codec_chunk_frames is not None:
            extra["initial_codec_chunk_frames"] = args.initial_codec_chunk_frames
        if args.codec_chunk_frames is not None:
            extra["codec_chunk_frames"] = args.codec_chunk_frames
        with open(yaml_path, "w") as f:
            yaml.safe_dump(cfg, f)

    prompt = build_tts_prompt(FIXED_TEXT, args.model)
    ar_params = SamplingParams(
        temperature=1.5,
        top_p=1.0,
        top_k=50,
        max_tokens=args.max_ar_tokens,
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

    omni = Omni(model=args.model, stage_configs_path=yaml_path, init_sleep_seconds=args.init_sleep_seconds)
    try:
        start = time.perf_counter()
        outputs = omni.generate([{"prompt": prompt}], [ar_params, decoder_params])
        wall = time.perf_counter() - start
        print(f"wall_s={wall:.6f}")
        print(f"outputs={len(outputs)}")
        for idx, output in enumerate(outputs):
            request_output = getattr(output, "request_output", None)
            print(
                "stage",
                idx,
                "final_type",
                getattr(output, "final_output_type", None),
                "metrics",
                getattr(output, "metrics", None),
            )
            if request_output is None:
                continue
            print(
                " request_id",
                getattr(request_output, "request_id", None),
                "finished",
                getattr(request_output, "finished", None),
                "metrics",
                getattr(request_output, "metrics", None),
                "prompt_tokens",
                len(getattr(request_output, "prompt_token_ids", []) or []),
            )
            for comp_idx, completion in enumerate(getattr(request_output, "outputs", []) or []):
                token_ids = getattr(completion, "token_ids", []) or []
                print(
                    " completion",
                    comp_idx,
                    "tokens",
                    len(token_ids),
                    "finish",
                    getattr(completion, "finish_reason", None),
                )
                multimodal_output = getattr(completion, "multimodal_output", None)
                if isinstance(multimodal_output, dict) and multimodal_output.get("audio") is not None:
                    audio = multimodal_output["audio"]
                    if isinstance(audio, list):
                        print(" audio_list_samples", [item.numel() for item in audio])
                    else:
                        print(" audio_samples", audio.numel())
    finally:
        close = getattr(omni, "close", None)
        if callable(close):
            close()
        os.unlink(yaml_path)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
