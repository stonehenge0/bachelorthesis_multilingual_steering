import pandas as pd

# Task identifier and corresponding file paths
df_filepaths = {
    "multijail_baseline": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/tmp_judged_responses/multijail_baseline_judged.csv",
    "multijail_L11_S0.33": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/tmp_judged_responses/multijail_L11_S0.33_judged.csv",
    "multijail_L11_S0.66": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/tmp_judged_responses/multijail_L11_S0.66_judged.csv",
    "multijail_L11_S1.0": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/tmp_judged_responses/multijail_L11_S1.0_judged.csv",
    "or_bench_baseline": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/tmp_judged_responses/or_bench_baseline_judged.csv",
    "or_bench_L11_S0.33": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/tmp_judged_responses/or_bench_L11_S0.33_judged.csv",
    "or_bench_L11_S0.66": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/tmp_judged_responses/or_bench_L11_S0.66_judged.csv",
    "or_bench_L11_S1.0": "/scratch1/users/u14374/bachelorarbeit/bachelorthesis_multilingual_steering/results/tmp_judged_responses/or_bench_L11_S1.0_judged.csv",
}


# depending on how well the judge did its task, we count with value count or with str contains.
def count_safe_unsafe_invalid_counts(df):
    """Count occurrences of 'safe', 'unsafe', and 'invalid' in the 'llm_judgement' column using value_counts()."""
    counts = df["llm_judgement"].value_counts()

    total_count = counts.sum()
    print(total_count, "total responses counted.")
    print(len(df), "total responses in dataframe.")
    return (
        counts.get("safe", 0),
        counts.get("unsafe", 0),
        counts.get("invalid", 0),
    )  ## macht die Null hier irgendeinen Sinn?


def count_safe_unsafe_invalid_str_contains(df):
    """Count occurrences of 'safe', 'unsafe', and 'invalid' in the 'llm_judgement' column using .str.contains()."""
    safe = df["llm_judgement"].str.contains("safe", case=False, na=False).sum()
    unsafe = df["llm_judgement"].str.contains("unsafe", case=False, na=False).sum()
    invalid = df["llm_judgement"].str.contains("invalid", case=False, na=False).sum()
    return safe, unsafe, invalid


results = {}

for task, path in df_filepaths.items():
    df = pd.read_csv(path)

    # Count occurrences of 'safe', 'unsafe', and 'invalid' using value_counts()
    safe_count, unsafe_count, invalid_count = count_safe_unsafe_invalid_counts(df)

    results[task] = {
        "safe": safe_count,
        "unsafe": unsafe_count,
        "invalid": invalid_count,
    }

    print(f"Task: {task}")
    print(f"Safe: {safe_count}, Unsafe: {unsafe_count}, Invalid: {invalid_count}")

# cols: tasks and steer levels, rows: safe, unsafe, invalid countss
results_df = pd.DataFrame.from_dict(results, orient="index").T

print("\nResults DataFrame:")
print(results_df)
