from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_clean_data(
    path="data/pinpoints.csv",
):
    """
    Data loading and preprocessing for spatial interaction analysis.

    This script imports raw pinpoint vegetation data, applies quality filters, and standardizes species names.
    It removes undetermined entries, duplicates, and species with incomplete names. Species names are then truncated
    to standardized short labels and species defind only at the species level (not at the subspecies level).
    Additionally, pinpoints are recoded to account for subplot positioning, ensuring a consistent spatial reference
    across quadrats.
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
    - Colonizer (not present in alpine community): yellow
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
            else ("#52cfebff" if in_alpine else "#feff63ff")
        )
    return origin


def compute_DCi(df_group):
    """
    Calculate the Dominance Candidate index (DCi) for species within a treatment group.

    The DCi quantifies a species' dominance by averaging its mean relative abundance
    and its frequency of occurrence across replicate plots. Relative abundance is
    computed per replicate, and frequency reflects the proportion of replicates in
    which the species occurs.

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

def impact_abund_freq_DCi(df, treatment):
    """
    Visualize by what is driven the DCi.
    """
    df = df[df["Site_Treatment"] == treatment]
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

    # Identifie the dominant species
    threshold = np.percentile(DCi_values, 90)
    dominant_indices = np.where(DCi_values >= threshold)[0]

    abund_bins = np.histogram_bin_edges(abunds, bins=20)
    freq_bins = np.histogram_bin_edges(freqs, bins=20)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(abunds, bins=20, color="steelblue", edgecolor="black")
    axes[0].hist(
        abunds[dominant_indices],
        bins=abund_bins,
        color="red",
        edgecolor="black",
        alpha=0.7,
        label="Dominant species",
    )
    axes[0].set_title("Mean relative abundance")
    axes[0].set_xlabel("Abundance")
    axes[0].set_ylabel("Number of species")
    axes[0].legend()

    axes[1].hist(freqs, bins=20, color="darkorange", edgecolor="black")
    axes[1].hist(
        freqs[dominant_indices],
        bins=freq_bins,
        color="red",
        edgecolor="black",
        alpha=0.7,
        label="Dominant species",
    )
    axes[1].set_title("Frequency of occurence")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Number of species")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"save_result/impact_abund_freq_DCi_{treatment}.svg",
        format="svg",
        dpi=300,
    )
