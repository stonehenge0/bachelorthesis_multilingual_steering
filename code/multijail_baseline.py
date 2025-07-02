"""Generate responses for the MultiJail dataset. It tests to how many unsafe prompts a model responds to in 10 typologically diverse languages.
"""

# imports
import os
import json

import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import set_seed
from tqdm import tqdm
import numpy as np
from huggingface_hub import login


# HF login and seed
with open(os.path.expanduser('~/.cache/huggingface/token'), 'r') as f:
    hf_token = f.read().strip()
login(token=hf_token)

set_seed(42)

# paths and globals
MULTIJAIL_PATH = "/scratch-scc/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/data/MultiJail.csv"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
OUT_FILE = "MultiJail_results.csv"
BATCH_SIZE = 4

# language ids in MultiJail
lang_ids = [
    "en",
    "zh",
    "it",
    "vi",
    "ar",
    "ko",
    "th",
    "bn",
    "sw",
    "jv"
]

SYSTEM_PROMPT = "You are a helpful assistant."

# load model and tokenizer from hf
def load_model_and_tokenizer(MODEL):
    """Load model and tokenizer from a given Hugging Face model."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    return model, tokenizer


# get batched responses for prompts for one lang. column
def get_batched_responses_for_language(prompts, model, tokenizer):
    """Generate model responses for a single batch of prompts."""
    batch_messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    texts = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in batch_messages
    ]

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return responses


# apply get_batched_responses_for_language to every column
def process_dataframe(df, model, tokenizer, lang_ids, batch_size=8):
    """Processes a DataFrame and returns a dict of language -> list of {prompt, response} dicts."""
    text_data = {}

    for lang in tqdm(lang_ids, desc="Processing languages"):
        if lang not in df.columns:
            print(f"Warning: Language '{lang}' not found in DataFrame columns")
            continue

        prompts = df[lang].tolist()
        results_for_lang = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            batch_responses = get_batched_responses_for_language(
                prompts=batch_prompts,
                model=model,
                tokenizer=tokenizer
                )

            for prompt, response in zip(batch_prompts, batch_responses):
                results_for_lang.append({"prompt": prompt, "response": response})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        text_data[lang] = results_for_lang

    return text_data


if __name__ == "__main__":

    # 1. load model and tokenizer
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(MODEL)

    # 2. read in our data
    df = pd.read_csv(MULTIJAIL_PATH)

    # 3. generate responses for every language
    text_data= process_dataframe(
        df, loaded_model, loaded_tokenizer, lang_ids, batch_size=BATCH_SIZE
    )

    # 4. save outputs
    with open("output.json", "w", encoding="utf-8") as f:
      json.dump(text_data, f, ensure_ascii=False, indent=4) # No ascii, since we have many different scripts in our answer data


    # 4.5 Sanity check prints
    print("Model:", loaded_model)
    print()
    print(f"Output has been written to: {OUT_FILE}")
