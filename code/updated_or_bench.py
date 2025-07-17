import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import sentencepiece

# Global cache for loaded models
_model_cache = {}

OR_BENCH_PATH = "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/data/sampled_or_bench_200_prompts.csv"
TARGET_LANGUAGES = ["zh", "ja"]  ##  "ko"

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
}


LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}


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


def translate_dataframe_optimized(sampled_df, prompt_column, target_langs):
    """
    Optimized translation that loads each model only once and processes all prompts
    for that language before moving to the next language.

    Args:
        sampled_df (pd.DataFrame): Input DataFrame with prompts.
        prompt_column (str): Name of the column containing prompts.
        target_langs (List[str]): List of ISO codes to translate into.

    Returns:
        pd.DataFrame: DataFrame with translated samples, ids, and language codes.
    """
    print("=== Starting optimized translate_dataframe ===")
    translations = []

    # Extract all prompts and their indices
    prompts = sampled_df[prompt_column].tolist()
    indices = sampled_df.index.tolist()

    # Process one language at a time
    for lang in target_langs:
        print(f"\n--- Processing language: {lang} ---")

        try:
            # Load model once for this language
            model, tokenizer = load_model_for_lang(lang)

            # Translate all prompts for this language
            translated_texts = translate_batch(
                prompts,
                source_lang="en",
                target_lang=lang,
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


def clear_model_cache():
    """Clear the model cache to free up memory."""
    global _model_cache
    _model_cache.clear()
    print("Model cache cleared")


if __name__ == "__main__":

    # read in dataframe of sampled OR-bench.
    df = pd.read_csv(OR_BENCH_PATH)
    print(f"Loaded df: {df.info}")

result_df = translate_dataframe_optimized(df, "prompt", TARGET_LANGUAGES)
