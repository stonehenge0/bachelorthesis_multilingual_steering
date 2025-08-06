# Translation script for OR-Bench using X-ALMA model.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import sentencepiece

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


def load_model_for_lang(
    lang_code,
):  ## was passiert hier wenn ich zweimal das gleiche model lade?
    """Load model and tokenizer based on language group."""
    print("=== Starting Load model ===")
    print(f"lang_code for model load: {lang_code}")
    group_id = LANG2GROUP.get(lang_code)

    if group_id is None:
        raise ValueError(f"Language '{lang_code}' not supported.")

    model_name = f"haoranxu/X-ALMA-13B-Group{group_id}"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    print(f"model:{model_name}\n ===load_model has run sucessfully===")
    return model, tokenizer


def translate(text, source_lang, target_lang):
    """Translate text from source_lang to target_lang using X-ALMA model."""
    model, tokenizer = load_model_for_lang(target_lang)

    # get full language names
    src_name = ISO_TO_NAME.get(source_lang, source_lang)
    tgt_name = ISO_TO_NAME.get(target_lang, target_lang)

    # format our prompt
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

    return translation


def translate_dataframe(sampled_df, prompt_column, target_langs):
    """
    Translates each prompt in the DataFrame into multiple target languages.

    Args:
        sampled_df (pd.DataFrame): Input DataFrame with prompts.
        prompt_column (str): Name of the column containing prompts.
        target_langs (List[str]): List of ISO codes to translate into.

    Returns:
        pd.DataFrame: DataFrame with translated samples, ids, and language codes.
    """
    print("=== Starting translate_dataframe ===")
    translations = []

    for i, row in sampled_df.iterrows():
        print(f"one iteration: i = {i}, row = {row}")

        original_text = row[prompt_column]
        print(f"Original text: {original_text}")

        for lang in target_langs:
            try:
                translated_text = translate(
                    original_text,
                    source_lang="en",
                    target_lang=lang,  # (we assume English is always source lang.)
                )
                translations.append(
                    {
                        "id": f"{i}_{lang}",
                        "lang": lang,
                        "text": translated_text,
                    }
                )
            except Exception as e:
                print(f"Error translating row {i} to {lang}: {e}")
                continue

    print(f"number of translations collected: {len(translations)}")
    print(f"First couple Translations: {translations[:4]}")

    print("=== Finished translate_dataframe ===")
    return pd.DataFrame(
        translations
    )  ## remember to also return everything English and ideally just append.


# Example usage: Translate Chinese text to English
if __name__ == "__main__":

    # read in dataframe of sampled OR-bench.
    df = pd.read_csv(OR_BENCH_PATH)
    print(f"Loaded df: {df.info}")

    # translate into each target language.
    translated_df = translate_dataframe(
        df, prompt_column="prompt", target_langs=TARGET_LANGUAGES
    )
    print(f"Translated df info: {translated_df.info}")

    # save the translated DataFrame to a CSV file.
    translated_df.to_csv("translated_or_bench.csv", index=False)
