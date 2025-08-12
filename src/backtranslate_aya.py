# Translate the LLM answers on Or-bench back to English for
# evaluation using Aya101.

import os
import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from huggingface_hub import login
import jsonlines

from utils import seed_everything

# seeds
seed_everything(42)

# hf login
with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
    hf_token = f.read().strip()
login(token=hf_token)

# Globals
BATCH_SIZE = 16
DEVICE = "cuda:0"

# Prompt for the llm-judge. This prompt is from the original MultiJail paper.
TRANSLATION_INSTRUCTION = """Translate to English: """


def load_model(hf_path):
    """Load model and tokenizer from huggingface."""
    tokenizer = AutoTokenizer.from_pretrained(hf_path, padding_side="left")
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_path).to(DEVICE)
    return tokenizer, model


def set_pad_ids(tokenizer, model):
    """Set pad token ids to eos token id."""
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id


def validate_inputs(df):
    """Print columns, NAs and length of dataframe. Optional, for debugging."""

    print("\nInformation on the Dataframe being processed:")
    print(f"Columns: {df.columns}")
    print(f"Number of samples: {len(df)}")
    print(f"NAs values:\n {df.isna().sum()}\n")


def folderpath_to_files_dict(folderpath):
    """Extract samples from MultiJail and Or-bench from parent folderpath.
    Expects samples to be named samples_multijail<timestamp>.jsonl and samples_or_bench<timestamp>.jsonl
    """
    files_dict = {}

    # Find all .jsonl files
    folder = Path(folderpath)

    for file_path in folder.rglob("*.jsonl"):
        filename = file_path.name

        # Parent directory indicates the steer type
        parent_1_level_up = file_path.parent.name
        parent_2_level_up = file_path.parent.parent.name

        if "multijail" in filename or "or_bench" in filename:
            files_dict[parent_2_level_up] = file_path

    print("Processing these files (should be multijail and or_bench):")
    for k, v in files_dict.items():
        print(f"{k}: {v}")

    return files_dict


def read_in_jsonl_to_df(filepath):
    """Read in jsonl object and turn to dataframe.
    - Clean LLM responses
    - Add col for combined prompt and LLM answer as input for LLM judge"""

    # jsonl to df
    data = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            data.append(obj)

    df = pd.DataFrame(data)

    # clean and add combined prompt + answer prompt as new col
    df["filtered_resps"] = (
        df["filtered_resps"]
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
    )
    df["prompt"] = df["doc"].apply(lambda x: x["prompt"])
    df["prompt_and_answer"] = (
        "Query: " + df["prompt"] + " Response: " + df["filtered_resps"]
    )
    # this is optional, but useful in debugging.
    validate_inputs(df)
    return df


def batch_generator(texts, tokenizer, batch_size=16):
    """Generator that yields batches of formated text inputs.
    Texts is a pandas Series."""

    batch = []

    for text in texts:
        # Format in chat template
        messages = [{"role": "user", "content": f"{JUDGE_INSTRUCTION} {text}"}]
        formated_texts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # batching
        batch.append(formated_texts)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Add final batch if it's shorter than batch size.
    if batch:
        yield batch


def generate_batched_answers(batch, model, tokenizer):
    """Generate llm-judge answers for a single batch of text."""

    # tokenize
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(
        DEVICE
    )

    # predict
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.01)

    # decode
    outputs = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return outputs


def save_pretty_output_df(full_df, task_name, out_path):
    """Save a df with answers to the specified path. It also makes the output df more readable for analysis"""

    cols_to_drop = [
        "doc_id",
        "target",
        "arguments",
        "resps",
        "filter",
        "metrics",
        "doc_hash",
        "prompt_hash",
        "target_hash",
        "bypass",
        "text",
        "prompt",  # this is not the actual prompt col we are dropping here, we keep "filtered responses" which is what we care about
        "category",
    ]

    # Lm-eval returns the samples a bit weird: the [doc] column is a json with different
    # values so we need to unpack here.
    doc_df = pd.json_normalize(full_df["doc"])
    full_df = pd.concat([full_df.drop("doc", axis=1), doc_df], axis=1)

    # drop uneccesary cols for readability.
    for col in cols_to_drop:
        if col in full_df.columns:
            full_df.drop(col, inplace=True)
        else:
            print(
                f"Colum: {col} not in the dataframe and was not dropped. Df has the following columns: {full_df.columns}"
            )

    full_out_path = f"{out_path}{task_name}_translated.csv"
    full_df.to_csv(full_out_path)
    print(f"Done! Finished dataframe has been written to: {full_out_path}")


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Judge for safety assessment.")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Huggingface model path to use for LLM judge.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/",
        help="Path to save the output dataframe.",
    )

    parser.add_argument(
        "--folderpath",
        type=str,  # Keep as string, parse later
        required=True,
        help="Path to the folder containing multijail and or_bench samples. Parent dir of multijail/or_bench is expected to indicate the task and steer level in its name.",
    )

    args = parser.parse_args()
    args.folderpath = json.loads(args.folderpath)

    return args


if __name__ == "__main__":

    args = get_args()
    # 0. Setup model and tokenizer
    print(f"Loading model {args.model} and tokenizer...")
    tokenizer, model = load_model(args.model)
    set_pad_ids(tokenizer, model)

    files_to_process_dict = folderpath_to_files_dict(args.folderpath)

    # processing dataframes
    for task, filepath in files_to_process_dict.items():
        print(
            f"\n============Starting processing============\nTask:{task}\nFile: {filepath}"
        )
        # 1. Read in and preprocess dataframe
        df = read_in_jsonl_to_df(filepath)

        # 2. Setup generators for batches
        generator_prompts = batch_generator(
            df["prompt"], tokenizer=tokenizer, batch_size=BATCH_SIZE
        )
        generator_answers = batch_generator(
            df["filtered_resps"], tokenizer=tokenizer, batch_size=BATCH_SIZE
        )

        # 3. Generate Translations
        # Translate prompts
        prompt_translations = []
        for batch in generator_prompts:
            aya_translations = generate_batched_answers(
                batch=batch, model=model, tokenizer=tokenizer
            )
            prompt_translations.extend(aya_translations)
        df["prompt_translated"] = prompt_translations

        # translate answers
        answer_translations = []
        for batch in generator_answers:
            aya_translations = generate_batched_answers(
                batch=batch, model=model, tokenizer=tokenizer
            )
            answer_translations.extend(aya_translations)
        df["answer_translated"] = answer_translations

        # 4. Save output df
        save_pretty_output_df(df, task_name=task, out_path=args.out_path)
