"""
Offline TTS inference with MOSS-TTS-Local on vllm-omni.

Usage:
  python end2end.py \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --stage-configs-path ../../../vllm_omni/model_executor/stage_configs/moss_tts.yaml \
    --text "Hello, this is a test of MOSS TTS." \
    --output-dir ./output_audio
"""
import os
import soundfile as sf
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni
from transformers import AutoTokenizer, AutoConfig

def build_tts_prompt(text: str, model_id: str) -> str:
    """
    Build the tokenized prompt string using MOSS-TTS's chat template.
    The MossTTSDelayProcessor handles inserting gen_slot tokens etc.
    """
    from moss_tts_local.processing_moss_tts import MossTTSDelayProcessor
    processor = MossTTSDelayProcessor.from_pretrained(model_id, trust_remote_code=True)
    messages = [
        {"role": "user", "content": text},
    ]
    # apply_chat_template returns token IDs ready for the model
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-TTS-Local-Transformer")
    parser.add_argument("--stage-configs-path",
                        default="../../../vllm_omni/model_executor/stage_configs/moss_tts.yaml")
    parser.add_argument("--text", default="The weather is so nice today.")
    parser.add_argument("--output-dir", default="./output_audio")
    parser.add_argument("--num-prompts", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
    )

    prompt = build_tts_prompt(args.text, args.model)
    print(f"Prompt: {prompt[:200]}...")

    ar_params = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=50,
        max_tokens=18192, seed=42, repetition_penalty=1.1,
    )
    decoder_params = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1,
        max_tokens=18192, seed=42, detokenize=False,
    )

    prompts = [{"prompt": prompt}] * args.num_prompts
    outputs = omni.generate(prompts, [ar_params, decoder_params])

    for stage_out in outputs:
        out = stage_out.request_output
        if stage_out.final_output_type == "audio":
            audio = out.outputs[0].multimodal_output.get("audio")
            if audio is None:
                print(f"[{out.request_id}] No audio output.")
                continue
            wav_path = os.path.join(args.output_dir, f"{out.request_id}.wav")
            sf.write(wav_path, audio.float().cpu().numpy().flatten(),
                     samplerate=24000, format="WAV")
            print(f"[{out.request_id}] Saved → {wav_path}")

if __name__ == "__main__":
    main()
