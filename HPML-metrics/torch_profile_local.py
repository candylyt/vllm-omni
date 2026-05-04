"""
Usage:
    PYTHONPATH=${REPO} \
    MOSS_AUDIO_TOKENIZER_PATH=${MOSS_AUDIO_TOKENIZER_PATH} \
    CUDA_VISIBLE_DEVICES=0 \
    python3 torch_profile_local.py \
    --model       ${MOSS_TTS_LOCAL_PATH} \
    --repo        ${REPO} \
    --mode        async \
    --trace-dir   ./profiler_traces \
    --samples     3 \
    --warmup      2 \
    --delay-iterations   2 \
    --active-iterations  5

  # open res in https://ui.perfetto.dev
"""

import argparse
import gc
import os
import time

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams
from vllm.config import ProfilerConfig
from vllm_omni.entrypoints.omni import Omni

#example(s) you want to test
SAMPLES = [
    "profiling example here",
]


USER_INST_TEMPLATE = """\
<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""


# make prompt based on Moss TTS template
def build_tts_prompt(text, model_path):
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(model_path), trust_remote_code=True)

    content = USER_INST_TEMPLATE.format(reference="None", instruction="None",  tokens="None", 
                                        quality="None", sound_event="None", ambient_sound="None", language="None", text=str(text))
    
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    return prompt + "<|audio_start|>"


# get stop ids for decoding from model config. audio end token and/or eos token
def get_stop_ids(model_path):
    cfg = AutoConfig.from_pretrained(os.path.abspath(model_path), trust_remote_code=True)

    audio_end_id = getattr(cfg, "audio_end_token_id", None)
    eos_id = getattr(cfg, "eos_token_id", None)

    if isinstance(eos_id, list):
        eos_ids = eos_id
    elif eos_id is not None:
        eos_ids = [eos_id]
    else:
        eos_ids = []

    return list(dict.fromkeys(([audio_end_id] if audio_end_id is not None else []) + eos_ids))



def main():
    p = argparse.ArgumentParser(description="Minimal PyTorch Profiler for MOSS-TTS")
    p.add_argument("--model", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--mode", required=True, choices=["async", "sync"])
    p.add_argument("--trace-dir", default="./profiler_traces")
    p.add_argument("--init-sleep-seconds", type=int, default=30)
    p.add_argument("--max-ar-tokens", type=int, default=500)
    p.add_argument("--delay-iterations", type=int, default=0)
    p.add_argument("--active-iterations", type=int, default=50)
    p.add_argument("--record-shapes", choices=["True", "False"], default="False")
    p.add_argument("--with-flops", choices=["True", "False"], default="False")
    p.add_argument("--with-mem", choices=["True", "False"], default="False")
    p.add_argument("--use-gzip", choices=["True", "False"], default="False")
    p.add_argument("--with-stack", choices=["True", "False"], default="False")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--samples", type=int, default=None)

    args = p.parse_args()

    audioTokenizerPath =  os.environ.get("MOSS_AUDIO_TOKENIZER_PATH", "")
    os.makedirs(args.trace_dir, exist_ok=True)
    
    # get req num samples or all
    samples = SAMPLES[:args.samples] if args.samples else SAMPLES

    ar_stop_ids = get_stop_ids(args.model)
    ar_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=50, max_tokens=args.max_ar_tokens, seed=42,stop_token_ids=ar_stop_ids if ar_stop_ids else None)
    decoder_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=18192, seed=42, detokenize=False)

    mode = args.mode
    print(f"\nmode: {mode}")

    yamlName = "moss_tts_async.yaml" if mode == "async" else "moss_tts.yaml"
    stageCfg = os.path.join(args.repo, f"vllm_omni/model_executor/stage_configs/{yamlName}")

    modeTraceDir = os.path.join(args.trace_dir, mode)
    os.makedirs(modeTraceDir, exist_ok=True)

    #use porfiler config from vllm 
    profilerCfg = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=modeTraceDir,
        torch_profiler_use_gzip=args.use_gzip == "True",
        torch_profiler_record_shapes=args.record_shapes == "True",
        torch_profiler_with_memory=args.with_mem == "True",
        torch_profiler_with_stack=args.with_stack == "True",
        torch_profiler_with_flops=args.with_flops == "True",
        delay_iterations=args.delay_iterations,
        active_iterations=args.active_iterations,
    )
    
    engine = Omni(
        model=args.model,
        audio_tokenizer=audioTokenizerPath,
        stage_configs_path=stageCfg,
        profiler_config=profilerCfg,
        init_sleep_seconds=args.init_sleep_seconds,
    )

    engine.model = args.model

    #warmup
    if args.warmup > 0:
        for _ in range(args.warmup):
            for _ in engine.generate([{"prompt": build_tts_prompt(samples[0], args.model)}], [ar_params, decoder_params]):
                pass

        print("warmup complete")

        gc.collect()
        torch.cuda.synchronize()

    # start the profiler and go through the samples
    engine.start_profile()

    for idx, text in enumerate(samples):
        shortText = text[:60] + ("..." if len(text) > 60 else "")
        print(f"  [{idx + 1} / {len(samples)}] Executing: \"{shortText}\"")

        for _ in engine.generate([{"prompt": build_tts_prompt(text, args.model)}], [ar_params, decoder_params]):
            pass

    print(f"Done: profiler and exporting traces")
    
    engine.stop_profile()
    
    if args.use_gzip == "True":
        time.sleep(3)
    
    # cleanup cleanup everybody everwhere woohoo
    del engine
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
