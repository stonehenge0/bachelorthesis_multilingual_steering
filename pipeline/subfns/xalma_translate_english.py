# Translate samples to English for evaluation (Or-bench) and analysis (Multijail)
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import sentencepiece
import argparse
from datasets import load_dataset
from utils import seed_everything

_model_cache = {}

def get_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()

    # Task and model
    parser.add_argument(
        "--source_langs",
        required=True,
        type=str,
        nargs="+",
        help="ISO codes for the languages to translate from. For options see X-Alma docs.",
    )

    parser.add_argument("--or_bench_folder", type=str, help="Path to the folder with or bench results for all steering strengths.")

    return parser.parse_args()


args = get_args()


TARGET_LANGUAGES = args.source_langs
OUT_PATH = f"/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/data/or_bench_translated_{'_'.join(TARGET_LANGUAGES)}.csv"

# Language grouping as required by the model
GROUP2LANG = {
    1: ["da", "nl", "de", "is", "no", "sv", "af"],
    2: ["ca", "ro", "gl", "it", "pt", "es"],
    3: ["bg", "mk", "sr", "uk", "ru"],
    4: ["id", "ms", "th", "vi", "mg", "fr"],
    5: ["hu", "el", "cs", "pl", "lt", "lv"],
    6: ["ka", "zh", "ja", "ko", "fi", "et"],
    7: ["gu", "hi", "mr", "ne", "ur"],
    8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
}

# map ISO lang codes to full language names for prompt.
ISO_TO_NAME = {
    "ka": "Georgian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fi": "Finnish",
    "et": "Estonian",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
}

LANG2GROUP = {lang: str(group) for group, source_langs in GROUP2LANG.items() for lang in source_langs} ### double check this w. the source_langs

def folderpath_to_file_dicts(all_results_folderpath):
    """Filter out or bench answer files."""

    or_bench_dict = {}

    for file in os.listdir(all_results_folderpath):
        if "samples_or_bench" in file:
            or_bench_dict[file] = os.path.join(all_results_folderpath,file)

    print(f"Files to process in or_bench_dict: {or_bench_dict}")
    return or_bench_dict

def load_model_for_lang(lang_code):
    """Load model and tokenizer based on language group with caching."""
    print("=== Starting Load model ===")
    print(f"lang_code for model load: {lang_code}")

    # Check if model is already cached
    if lang_code in _model_cache:
        print(f"Model for {lang_code} already loaded, using cached version")
        return _model_cache[lang_code]

    group_id = LANG2GROUP.get(lang_code)
    if group_id is None:
        raise ValueError(f"Language '{lang_code}' not supported.")

    model_name = f"haoranxu/X-ALMA-13B-Group{group_id}"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # Cache the loaded model
    _model_cache[lang_code] = (model, tokenizer)

    print(f"model:{model_name}\n ===load_model has run sucessfully===")
    return model, tokenizer


def translate_batch(texts, source_lang, target_lang, model, tokenizer):
    """Translate a batch of texts using pre-loaded model."""
    translations = []

    for text in texts:
        # Get full language names
        src_name = ISO_TO_NAME.get(source_lang, source_lang)
        tgt_name = ISO_TO_NAME.get(target_lang, target_lang)

        # Format prompt
        prompt = f"Translate this from {src_name} to {tgt_name}:\n{src_name}: {text}\n{tgt_name}:"
        chat_prompt = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            chat_prompt, tokenize=False, add_generation_prompt=True
        )

        input_ids = tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=256, truncation=True
        ).input_ids.cuda()

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                num_beams=5,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        translation = outputs[0].strip()
        translation = translation.replace("[/INST]", "")
        translations.append(translation)

    return translations


def translate_dataframe(df, prompt_column, source_langs):
    """
    Optimized translation that loads each model only once and processes all prompts
    for that language before moving to the next language.

    Args:
        df (pd.DataFrame): Input DataFrame with prompts.
        prompt_column (str): Name of the column containing prompts.
        source_langs (List[str]): List of ISO codes to translate from.

    Returns:
        pd.DataFrame: DataFrame with translated samples, ids, and language codes.
    """
    print("=== Starting optimized translate_dataframe ===")
    translations = []

    # Extract all prompts and their indices
    prompts = df[prompt_column].tolist()
    indices = df.index.tolist()

    # Process one language at a time
    for lang in source_langs:
        print(f"\n--- Processing language: {lang} ---")

        try:
            # Load model once for this language
            model, tokenizer = load_model_for_lang(lang)

            # Translate all prompts for this language
            translated_texts = translate_batch(
                prompts,
                source_lang=lang,
                target_lang="en",
                model=model,
                tokenizer=tokenizer,
            )

            # Store results
            for idx, original_idx in enumerate(indices):
                translations.append(
                    {
                        "id": f"{original_idx}_{lang}",
                        "lang": lang,
                        "text": translated_texts[idx],
                    }
                )

            print(f"Completed {len(translated_texts)} translations for {lang}")

        except Exception as e:
            print(f"Error processing language {lang}: {e}")
            continue

    print(f"\nTotal translations collected: {len(translations)}")
    print(f"First couple translations: {translations[:4]}")
    print("=== Finished optimized translate_dataframe ===")

    return pd.DataFrame(translations)


if __name__ == "__main__":

    # seed everything
    seed_everything(42)

    # 1. Load in all steering files: 
    or_bench_files_dict = folderpath_to_file_dicts(args.or_bench_folder)

    # 2. Iterate over all files and translate.
    for name, filepath in or_bench_files_dict.items():

        df = pd.read_csv(filepath, lines=True) # Input is jsonl, not normal json

        # 3. Translate
        result_df = translate_dataframe(df, "filtered_resps", TARGET_LANGUAGES)
        print(f"Translated df info: {result_df.info}")
        result_df.to_csv(OUT_PATH, index=False)
