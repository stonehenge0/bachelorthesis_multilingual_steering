"""We translate the over-refusal benchmark XSTest to more languages for multilingual evaluation."""

## remember to add to original english things back too!
## import these into your venv!
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import json

# globals
MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # full 12.9B model: CohereLabs/aya-101
DEVICE= "cpu"
path_to_xstest = "data/xstest.jsonl"
test_text = ["Das ist ein Test", "Katzen sind schon cute"]

target_langs = ["zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]

def translate(texts: list, target_lang: str) -> list:
    """Translate a list of prompts to a target language."""

    # build translation pipe specifically for that language.
    lang_pattern_pipe = "translation_en_to_" + target_lang
    lang_specific_pipe = pipeline(lang_pattern_pipe, model=MODEL, device = DEVICE, batch_size = BATCH_SIZE)

    translated_texts = lang_specific_pipe(
        texts
    )  # not sure if this can handle a list, but let's try...

    return translated_texts
