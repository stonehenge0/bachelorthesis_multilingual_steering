import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import sentencepiece

OR_BENCH_PATH = "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/data/sampled_or_bench_200_prompts.csv"
df = pd.read_csv(OR_BENCH_PATH)
print(f"Loaded df: {df.info}")

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
    "pt": "Portuguese"
}


LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}

def check(x, name_of_x =False):
    """helper function for checking shapes and types during debugging."""
    print("====CHECKING======")
    if name_of_x:
        print(f"name:{name_of_x}")
    print(f"type: {type(x)}")
    if x.shape:
        print(f"shape: {x.shape}")
        

def load_model_for_lang(lang_code): ## was passiert hier wenn ich zweimal das gleiche model lade? 
    """Load model and tokenizer based on language group."""
    print("=== Starting Load model ===")
    print(f"lang_code passed in: {lang_code}")
    group_id = LANG2GROUP.get(lang_code)
    print(f"group_id inside load_model:{group_id}")
    
    if group_id is None:
        raise ValueError(f"Language '{lang_code}' not supported.")

    model_name = f"haoranxu/X-ALMA-13B-Group{group_id}"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    print(f"model:{model} \n tokenizer: {tokenizer}\n ===load_model has run sucessfully===")
    return model, tokenizer

def translate(text, source_lang, target_lang):
    """Translate text from source_lang to target_lang using X-ALMA model."""
    print("=== Starting translate function ===")
    model, tokenizer = load_model_for_lang(target_lang) ## here is the issue...

    # get full language names
    src_name = ISO_TO_NAME.get(source_lang, source_lang)
    tgt_name = ISO_TO_NAME.get(target_lang, target_lang)
    print(f"source lang id and name: {source_lang}  {src_name}")
    print(f"target lang id and name: {target_lang}  {tgt_name}")


    # format our prompt
    prompt = f"Translate this from {src_name} to {tgt_name}:\n{src_name}: {text}\n{tgt_name}:"
    print(f"Prompt-text: {prompt}")
    chat_prompt = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
    print(f"prompt with chat template after apply chat template: {prompt}")

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=256, truncation=True).input_ids.cuda()
    print(f"input ids: {input_ids}")
    
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
    print(f"outputs after torch.no grad forward: {outputs}")
    translation = outputs[0].strip()
    print(f"returned translation object {translation}")
    return translation


# Example usage: Translate Chinese text to English
if __name__ == "__main__":
    input_text = "Das ist ein Test!"
    result = translate(input_text, source_lang="en", target_lang="zh")
    print(result)
