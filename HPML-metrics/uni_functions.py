
import argparse
import gc
import os
import time

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams
from vllm.config import ProfilerConfig
from vllm_omni.entrypoints.omni import Omni

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
