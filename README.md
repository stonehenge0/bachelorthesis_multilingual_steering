# Multilingual Refusal Steering
![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-red?logo=pytorch&logoColor=white) ![Status](https://img.shields.io/badge/status-final-brightgreen) ![Code Style](https://img.shields.io/badge/code%20style-black-black) 

This repository contains code and experiments for improving **LLM safety across languages** using steering vectors.  

Current state-of-the-art safety methods often perform well in English but fail to generalize to other languages. For example if you ask a model to build a bomb in English it will refuse to answer. If you ask the exact same question in Bengali however, it might answer you. Unsafety rates are up to 8 times higher for low-resource languages than for English. This is a problem!

Our project explores a lightweight and scalable approach: deriving **refusal steering vectors** from contrastive prompts (harmful vs. harmless) and applying them during inference to encourage safer model behavior across multiple languages.  

Our goals are to:  
- Derive steering vectors that reliably increase refusal to harmful prompts.  
- Test whether vectors derived from English generalize to other languages.  
- Explore language-specific and multilingual steering vectors.  
- Measure tradeoffs between improved safety, over-refusal, and general model performance.  

<details>
<summary>Read the Abstract</summary>

The increasing adoption of Large Language Models has made them accessible to billions of speakers across hundreds of languages. Despite this, their safety mechanisms remain predominantly designed and tested for English. A cornerstone of these mechanisms is the ability to refuse answering harmful prompts. Recent research has shown that while this refusal functions well in English, LLMs are significantly more vulnerable to malicious queries when prompted in other languages, especially low-resource ones. This is a major risk to equitable and safe deployment across the globe as it is currently unmitigated.

Although previous methods have shown some success in tackling this problem, their solutions often require computationally expensive and data-intense fine-tuning or language-specific interventions. What remains underexplored is a method that is lightweight and can scale to many languages, including lower-resource ones. For these languages, approaches requiring large amounts of high-quality data are often impossible to implement due to limited data available.

This thesis addresses this gap and analyzes the potential of steering vectors derived from the refusal subspace of a model to tackle this problem: We evaluate their ability to improve safety across languages as well as their trade-offs with general language understanding and over-refusal. We show that steering using vectors derived exclusively from English data successfully reduces the rate of answers to malicious prompts by 40 percent across almost all languages of different resource levels. Our implementation does not require language-specific data to increase model safety, circumventing a major bottleneck in scaling alignment.

Moreover, we are the first to extend the analysis of over-refusal of benign prompts to a multilingual setting. Models refusing benign prompts is a frequently underexamined trade-off in safety research, and we provide valuable contributions to the field by benchmarking it for both the baseline case and steered models.

</details>

---

## Methodology
![Methodology Overview](images/methodology.png)


1. **Steering Vector Extraction**  
   - We follow the approach of *Contrastive Activation Addition*.  
   - Harmful and harmless prompts are passed through the model.  
   - A **difference-in-means vector** is computed for each layer.  
   - Vectors are filtered using KL divergence and refusal success criteria.  

2. **Applying Steering Vectors**  
   - The selected vector is added to the forward pass across all token positions in a chosen layer.  
   - Steering strength is controlled by a scalar hyperparameter.  

3. **Evaluation**  
   - **Safety:** refusal to harmful prompts → [MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail).  
   - **Over-refusal:** false refusals to harmless prompts → [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench).  
   - **General ability:** accuracy on diverse tasks → [Global MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite).  

4. **Languages Covered**  
   - Core: English, Chinese, Italian, Arabic, Korean.  
   - Additional: Vietnamese, Thai, Bengali, Swahili, Javanese.  

---

## Technical Setup

- **Base Model:** Llama-3.1-8B-Instruct  
- **Evaluation Harness:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with custom tasks for MultiJail and OR-Bench  
- **Compute:** Single NVIDIA A100 GPU 40GB
- **Translation Model:** [X-ALMA](https://huggingface.co/x-alma), used for preparing multilingual datasets  

---

## Setup

```bash
git clone https://github.com/stonehenge0/safety_steering.git
cd safety_steering
bash setup.sh
```

The script will prompt you for a HuggingFace token (required to access gated models). It will then set up a Conda environment, install the required packages. For evaluation we use a modified_lm_eval_harness. It is exactly the same as the [original](https://github.com/EleutherAI/lm-evaluation-harness), but with two added tasks (MultiJail and OR-Bench).

> The script is primarily inteded for setup on an HPC cluster, but you can use environment.yaml to set up wherever you like. 
 
 ## Run main script ## Command Line Arguments | Flag | Type | Required | Default | Description | |------|------|----------|---------|-------------| | --model_path | string | Yes | - | Path to the model (e.g., "meta-llama/Llama-3.1-8B-Instruct") | | --steering_vector_path | string | Yes | - | Path to the steering vector .pt file | | --steering_layer | integer | Yes | - | Layer to apply steering to (e.g., 11) | | --steering_strengths | float(s) | Yes | - | Steering strengths to use, separated by spaces (e.g., 0.33 0.66 1.0) | | --device | string | No | cuda:0 | Device to use for computation (e.g., "cuda:0", "cpu") | | --debug | flag | No | False | Run on a subsample of datasets and tasks for testing | 
 
 
### Example Usage
```bash
python /code/lm_eval_steered_and_baseline_tasks.py \
  --model_path "meta-llama/Llama-3.1-8B-Instruct" \
  --steering_vector_path "/path/to/steering-vector.pt" \
  --steering_layer 11 \
  --steering_strengths 0.33 1.0 \
  --device "cuda:0" \
```

## Curious?
- **Want to read the full thesis?** There you go. [Link to thesis here]  
- **What else does you Lab do?** More AI Safety, Plagiarism, AI Agents,...
-**How can I contact you?** ```em dot stein29 at gmail dot com```
