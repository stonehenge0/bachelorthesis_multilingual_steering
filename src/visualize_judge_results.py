import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def create_task_barplots(results_df):
    """Create separate bar plots for multijail and or_bench tasks showing safe/unsafe/invalid counts."""

    # Separate columns by task
    multijail_cols = [col for col in results_df.columns if col.startswith("multijail")]
    or_bench_cols = [col for col in results_df.columns if col.startswith("or_bench")]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: MultiJail
    multijail_data = results_df[multijail_cols]
    multijail_data.plot(kind="bar", ax=ax1, width=0.8)
    ax1.set_title("MultiJail Task: Safety Judgments by Steering Level", fontsize=14)
    ax1.set_xlabel("Safety Category", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.legend(title="Steering Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.tick_params(axis="x", rotation=0)

    # Plot 2: OR-Bench
    or_bench_data = results_df[or_bench_cols]
    or_bench_data.plot(kind="bar", ax=ax2, width=0.8)
    ax2.set_title("OR-Bench Task: Safety Judgments by Steering Level", fontsize=14)
    ax2.set_xlabel("Safety Category", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.legend(title="Steering Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.tick_params(axis="x", rotation=0)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    return fig


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

# Create the plots
fig = create_task_barplots(results_df)
plt.show()

# Optional: Save the figure
fig.savefig("safety_judgments_by_task.png", dpi=300, bbox_inches="tight")

print("\nResults DataFrame:")
print(results_df)