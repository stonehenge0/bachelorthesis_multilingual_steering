external/refusal_direction. Andi Arditi et al. original  [Git](https://github.com/andyrdt/refusal_direction)

multijail 

OR bench

MMLU

lm eval harness

translation model (currently X-Alma)

### Notes on what's in this repo: 
- **sbash scripts** for running on HPC. They have to be on the main level to work, so no folder deeper or something. All write output to the slurm_files folder.

- **translations_for_inspection** contains some answers on mostly MultiJail and OR-Bench in a good readably format, because the way lm-eval-harness returns things is pretty ugly. There are the results of two different runs with different temperatures in there (t=0.7 and the later one with t=0.0, greedy). The runs that are named _long_translation are the 0.0t runs and were generated because the Aya model lowkey cut off some of our data for the t0.7 run and didn't have time to run another one. For the t0 runs the "index" is the steer strength with 0=baseline, 1=0.33 and so on.


### Archive Notes
- Use the larger Qwen Model for evaluation (3.5 instead of 2.5). Test that this aligns with your own judgement!
- Do LLM judge for OR-Bench too, instead of string, I have little faith in that.
- Don't rely on Aya101 for good translations. X-Alma is good, but has no coverage for low-resource.
- Write steering independent from LM-eval harness.