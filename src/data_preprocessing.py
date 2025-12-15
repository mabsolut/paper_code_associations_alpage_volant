from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_and_clean_data(
    path="data/pinpoints.csv",
):
    """
    Data loading and preprocessing for spatial interaction analysis.

    This script imports raw pinpoint vegetation data, applies quality filters, and standardizes species names.
    It removes undetermined entries, duplicates, species with incomplete names and subspecies.
    Additionally, pinpoints are recoded to account for subplot positioning, ensuring a consistent spatial reference across quadrats.
    """
    df = pd.read_csv(path)
    # Remove errors in the data
    df = df.drop(df.index[[5470, 30001]])
    df = df[df["Species"] != "aaa +++ En attente de détermination"]
    df.drop_duplicates(inplace=True)
    # Remove individuals only identify at the gender level
    df = df[df["Species"].str.split().str.len() >= 2]
    df["Species"] = df["Species"].apply(
        lambda x: f"{x.split()[0][:3]} {x.split()[1][:4]}"
    )

    # Recode pinpoints (pinpoints are initially divided in 4 Subplots2 of 25 pinpoints)
    df.loc[df["Subplot2"] == "LR", "pinpoint"] += 25
    df.loc[df["Subplot2"] == "UL", "pinpoint"] += 50
    df.loc[df["Subplot2"] == "UR", "pinpoint"] += 75

    return df


def compute_origin(df):
    """
    Classify species origin based on their presence in the alpine community.

    This function assigns each species a color label based on its distribution:
    - Alpine specialist (present only in alpine community): blue
    - Colonizer (not present in alpine community): orange
    - Non-specialist (present in both): grey

    The output is a dictionary mapping species to their corresponding origin color.
    """
    origin = {}
    for sp in df["Species"].unique():
        in_alpine = sp in df[df["Site_Treatment"] == "G_CP"]["Species"].values
        in_subalpine = sp in df[df["Site_Treatment"] == "L_CP"]["Species"].values
        origin[sp] = (
            "grey"
            if in_alpine and in_subalpine
            else ("#0072b2ff" if in_alpine else "#e69f00fe")
        )
    return origin


def compute_DCi(df_group):
    """
    Calculate the Dominance Candidate index (DCi) for species within a treatment group.

    Returns a dictionary mapping each species to its DCi value.
    """
    abund = defaultdict(list)
    DCi = {}
    replicates = list(df_group.groupby(["Year", "Subplot", "Replicate"]))
    n_reps = len(replicates)
    for _, rep_df in replicates:
        rel_abund = rep_df["Species"].value_counts(normalize=True)
        for sp, val in rel_abund.items():
            abund[sp].append(val)
    for sp, values in abund.items():
        freq = sum(sp in rep_df["Species"].values for _, rep_df in replicates)
        DCi[sp] = (np.mean(values) + freq / n_reps) / 2
    return DCi


def abund_freq_vs_DCi(df, treatment, origin):
    """
    Visualize what drives the DCi: relative abundance and frequency.
    """
    df = df[df["Site_Treatment"] == treatment]
    df = df[(df["Subplot"] == "B") | (df["Subplot"] == "0")]
    abund = defaultdict(list)
    freqs, abunds = [], []
    DCi = {}
    replicates = list(df.groupby(["Year", "Subplot", "Replicate"]))
    n_reps = len(replicates)
    for _, rep_df in replicates:
        rel_abund = rep_df["Species"].value_counts(normalize=True)
        for sp, val in rel_abund.items():
            abund[sp].append(val)
    for sp, values in abund.items():
        freq = sum(sp in rep_df["Species"].values for _, rep_df in replicates)
        freqs.append(freq / n_reps)
        abunds.append(np.mean(values))
        DCi[sp] = (np.mean(values) + freq / n_reps) / 2

    abunds, freqs = np.array(abunds), np.array(freqs)
    species = np.array(list(DCi.keys()))
    DCi_values = np.array([DCi[sp] for sp in species])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.array(DCi_values).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, freqs)
    slope, intercept, r2 = (
        model.coef_[0],
        model.intercept_,
        model.score(x, freqs),
    )
    x_line = np.linspace(min(DCi_values), max(DCi_values), 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    axes[0].plot(DCi_values, freqs, "o", label="Observed data")
    axes[0].plot(
        x_line,
        y_line,
        "-",
        label=f"Linear fit: y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r2:.3f}",
    )
    axes[0].set_title("Frequency vs DCi")
    axes[0].set_xlabel("DCi")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    model2 = LinearRegression()
    model2.fit(x, abunds)
    slope, intercept, r2 = (
        model2.coef_[0],
        model2.intercept_,
        model2.score(x, abunds),
    )
    x_line = np.linspace(min(DCi_values), max(DCi_values), 200).reshape(-1, 1)
    y_line = model2.predict(x_line)
    axes[1].plot(DCi_values, abunds, "o", label="Observed data")
    axes[1].plot(
        x_line,
        y_line,
        "-",
        label=f"Linear fit: y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r2:.3f}",
    )
    axes[1].set_title("Abundance vs DCi")
    axes[1].set_xlabel("DCi")
    axes[1].set_ylabel("Abundance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"save_result/Dominance/impact_abund_freq_DCi_{treatment}.svg",
        format="svg",
        dpi=300,
    )
