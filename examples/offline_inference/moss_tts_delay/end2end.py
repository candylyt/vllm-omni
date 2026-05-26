"""
Offline direct-TTS inference with MOSS-TTS-Delay on vllm-omni.

Usage:
  cd /workspace/vllm-omni/examples/offline_inference/moss_tts_delay
  MOSS_AUDIO_TOKENIZER_PATH=/workspace/vllm-omni/weights/moss-audio-tokenizer \
  python end2end.py \
    --model /workspace/vllm-omni/weights/moss-tts-delay \
    --stage-configs-path ../../../vllm_omni/model_executor/stage_configs/moss_tts_delay.yaml \
    --text "The weather is so nice today." \
    --output-dir ./output_audio

Download example:
  hf download OpenMOSS-Team/MOSS-TTS --local-dir /workspace/vllm-omni/weights/moss-tts-delay
  hf download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir /workspace/vllm-omni/weights/moss-audio-tokenizer
"""

from __future__ import annotations

import copy
import os

import torch
import soundfile as sf
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni

SEED = 42

_USER_INST_TEMPLATE = """\
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


def build_tts_prompt(
    text: str,
    model_path: str,
    instruction=None,
    tokens=None,
    quality=None,
    sound_event=None,
    ambient_sound=None,
    language=None,
) -> str:
    """Build the direct-TTS prompt used for the Phase-1 Delay path."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.abspath(model_path),
        trust_remote_code=True,
    )

    content = (
        _USER_INST_TEMPLATE
        .replace("{reference}", "None")
        .replace("{instruction}", str(instruction))
        .replace("{tokens}", str(tokens))
        .replace("{quality}", str(quality))
        .replace("{sound_event}", str(sound_event))
        .replace("{ambient_sound}", str(ambient_sound))
        .replace("{language}", str(language))
        .replace("{text}", str(text))
    )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "<|audio_start|>"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/vllm-omni/weights/moss-tts-delay")
    parser.add_argument(
        "--stage-configs-path",
        default="../../../vllm_omni/model_executor/stage_configs/moss_tts_delay.yaml",
    )
    parser.add_argument("--text", default="The weather is so nice today.")
    parser.add_argument("--output-dir", default="./output_audio")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=5000)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    args = parser.parse_args()

    if not os.environ.get("MOSS_AUDIO_TOKENIZER_PATH"):
        sibling = os.path.join(
            os.path.dirname(os.path.abspath(args.model)),
            "moss-audio-tokenizer",
        )
        if os.path.exists(sibling):
            os.environ["MOSS_AUDIO_TOKENIZER_PATH"] = sibling
            print(f"[Info] MOSS_AUDIO_TOKENIZER_PATH auto-set -> {sibling}")
        else:
            print("[Warn] MOSS_AUDIO_TOKENIZER_PATH not set; decoder will fall back to HF resolution.")

    os.makedirs(args.output_dir, exist_ok=True)

    prompt = build_tts_prompt(args.text, args.model)
    print(f"[Info] Prompt ({len(prompt)} chars):\n{prompt}\n")

    ar_params = SamplingParams(
        temperature=1.5,
        top_p=1.0,
        top_k=50,
        max_tokens=900,
        seed=SEED,
        repetition_penalty=1.0,
    )
    decoder_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=32768,
        seed=SEED,
        detokenize=False,
    )

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    prompts = [copy.deepcopy({"prompt": prompt}) for _ in range(args.num_prompts)]
    print(f"[Info] Running {len(prompts)} prompt(s)...")
    omni_outputs = omni.generate(prompts, [ar_params, decoder_params])

    for stage_outputs in omni_outputs:
        output = stage_outputs.request_output
        request_id = output.request_id

        if stage_outputs.final_output_type == "text":
            text_out = output.outputs[0].text
            print(f"[{request_id}] AR text: {text_out[:300]!r}")
            continue

        if stage_outputs.final_output_type != "audio":
            continue

        audio_tensor = output.outputs[0].multimodal_output.get("audio")
        if audio_tensor is None:
            print(f"[{request_id}] No audio in output.")
            continue

        # async_chunk mode accumulates a list of per-chunk tensors
        if isinstance(audio_tensor, list):
            chunks = [t for t in audio_tensor if isinstance(t, torch.Tensor) and t.numel() > 0]
            if not chunks:
                print(f"[{request_id}] No audio chunks in output.")
                continue
            audio_tensor = torch.cat(chunks, dim=0)

        audio_np = audio_tensor.float().detach().cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        wav_path = os.path.join(args.output_dir, f"{request_id}.wav")
        sf.write(wav_path, audio_np, samplerate=24000, format="WAV")
        duration = len(audio_np) / 24000
        print(f"[{request_id}] Saved -> {wav_path} ({duration:.2f}s)")


if __name__ == "__main__":
    main()
