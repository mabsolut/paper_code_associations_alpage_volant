import json
import multiprocessing as mp
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from association_detection import detect_species_associations


def simulate_alpha_error(
    df, treatment, nbplots: int = 30, n_iter: int = 1000, n_cores: int = 12
):
    """
    Performs simulation-based estimation of alpha error under the null hypothesis,
    by randomizing species–pinpoint assignments within plots.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing columns like 'Species', 'pinpoint', etc.
    treatment : str
        Site_Treatment to analyze (e.g., 'G_CP', 'L_TP', etc.).
    nbplots : int
        Number of plots to include in each simulation.
    n_iter : int
        Number of random simulations to perform.
    n_cores : int
        Number of processes to use for parallel computation.
    """
    subdf = df[df["Site_Treatment"] == treatment]
    grouped = list(subdf.groupby(["Year", "Site_Treatment", "Subplot", "Replicate"]))
    all_subdfs = []

    for _ in range(n_iter):
        random.shuffle(grouped)
        simdf = []
        for _, subsimdf in grouped[:nbplots]:
            total_pinpoint = subsimdf["pinpoint"].nunique()
            abundances = dict(Counter(subsimdf["Species"]))
            randomized_rows = []

            for species, abundance in abundances.items():
                tirage = np.random.choice(
                    range(total_pinpoint), abundance, replace=False
                )
                rows = (
                    subsimdf[subsimdf["Species"] == species]
                    .copy()
                    .reset_index(drop=True)
                )
                for i, pinpoint in enumerate(tirage):
                    row = rows.iloc[i].copy()
                    row["pinpoint"] = pinpoint
                    randomized_rows.append(row)

            randomized_df = pd.DataFrame(randomized_rows)
            simdf.append(randomized_df)

        all_subdfs.append(pd.concat(simdf, ignore_index=True))

    with mp.Pool(n_cores) as pool:
        res_list = pool.map(detect_species_associations, all_subdfs)

    # Save
    save_path = f"save_json/json_alpha_error_{treatment}.json"
    with open(save_path, "w") as f:
        json.dump(res_list, f)
    return res_list


def report_alpha_error(res_list, treatment):
    """
    Loads alpha error simulation results and prints average number of false positive links.

    Parameters
    ----------
    treatment : str
        Site_Treatment identifier (e.g., 'G_CP', 'L_TP').
    """
    print(
        "Alpha error of positive links",
        treatment,
        ":",
        np.mean([len(res_dict["over"]) for res_dict in res_list]),
    )

    print(
        "Alpha error of negative links",
        treatment,
        ":",
        np.mean([len(res_dict["under"]) for res_dict in res_list]),
    )


def compute_impact_plots_simulation(df, treatment):
    """
    Estimates the accumulation of species associations (links) as a function of the number of sampling plots,
    under a null model where pinpoint-level spatial structure is randomized while maintaining species abundances.

    This simulates the expected increase in co-occurrences due to sample size alone.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed input dataset including columns: 'Year', 'Site_Treatment', 'Subplot', 'Replicate',
        'Species', and 'pinpoint'.
    treatment : str
        Name of the treatment to consider (e.g., 'G_CP', 'L_TP').

    Returns
    -------
    result : dict
        Dictionary where keys are the number of used plots, and values are lists of
        detected associations (output of detect_species_associations).
    """
    subdf = df[df["Site_Treatment"] == treatment]

    result = defaultdict(list)

    grouped = list(subdf.groupby(["Year", "Site_Treatment", "Subplot", "Replicate"]))
    nb_plots = len(grouped)
    all_subdfs = []
    ns = []

    for n in range(1, nb_plots + 1):
        random.shuffle(grouped)
        for i in range(0, len(grouped) - n + 1, n):
            selected_groups = [group[1] for group in grouped[i : i + n]]
            simdf = []
            for subsimdf in selected_groups:
                total_pinpoint = subsimdf["pinpoint"].nunique()
                abondances = dict(Counter(subsimdf["Species"]))
                randomized_rows = []
                for specie, abondance in abondances.items():
                    tirage = np.random.choice(
                        range(total_pinpoint), abondance, replace=False
                    )
                    matching_rows = (
                        subsimdf[subsimdf["Species"] == specie]
                        .copy()
                        .reset_index(drop=True)
                    )
                    for i, pinpoint in enumerate(tirage):
                        row = matching_rows.iloc[i].copy()
                        row["pinpoint"] = pinpoint
                        randomized_rows.append(row)
                randomized_df = pd.DataFrame(randomized_rows)
                simdf.append(randomized_df)
            all_subdfs.append(pd.concat(simdf, ignore_index=True))
            ns.append(n)

    with mp.Pool(12) as pool:
        res_list = pool.map(detect_species_associations, all_subdfs)

    for n, res_dict in zip(ns, res_list):
        result[n].append(res_dict)

    with open(
        f"save_json/json_impact_plots_simulation_{treatment}.json",
        "w",
    ) as f:
        json.dump(result, f)

    return result
