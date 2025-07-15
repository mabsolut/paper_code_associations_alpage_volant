import json
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import pandas as pd
from gaste_test import get_pval_comb
from scipy.stats import hypergeom
from tqdm import tqdm

np.set_printoptions(legacy="1.21")

from association_detection import detect_species_associations


def simulate_alpha_error(df, treatment, n_iter: int = 2000, jobs=-1):
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
    subdf = subdf[(subdf["Subplot"] == "B") | (subdf["Subplot"] == "0")]
    grouped = list(subdf.groupby(["Year", "Site_Treatment", "Subplot", "Replicate"]))
    all_subdfs = [subdf.copy()] * n_iter

    # for _ in range(n_iter):
    #     simdf = []
    #     for _, subsimdf in grouped:
    #         total_pinpoint = subsimdf["pinpoint"].nunique()
    #         abundances = dict(Counter(subsimdf["Species"]))
    #         randomized_rows = []

    #         for species, abundance in abundances.items():
    #             tirage = np.random.choice(
    #                 range(total_pinpoint), abundance, replace=False
    #             )
    #             rows = (
    #                 subsimdf[subsimdf["Species"] == species]
    #                 .copy()
    #                 .reset_index(drop=True)
    #             )
    #             for i, pinpoint in enumerate(tirage):
    #                 row = rows.iloc[i].copy()
    #                 row["pinpoint"] = pinpoint
    #                 randomized_rows.append(row)

    #         randomized_df = pd.DataFrame(randomized_rows)
    #         simdf.append(randomized_df)

    #     all_subdfs.append(pd.concat(simdf, ignore_index=True))

    if jobs == -1:
        jobs = mp.cpu_count()
    with mp.Pool(jobs) as pool:
        res_list = pool.map(
            partial(detect_species_associations, random_state=True),
            tqdm([e for e in all_subdfs], total=n_iter),
        )

    # Save
    save_path = f"save_json/json_alpha_error_{treatment}_2.json"
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
    # print(Counter([link for res_dict in res_list for link in res_dict["over"].keys()]))
    print(
        "Alpha error of negative links",
        treatment,
        ":",
        np.mean([len(res_dict["under"]) for res_dict in res_list]),
    )
    # print(Counter([link for res_dict in res_list for link in res_dict["under"].keys()]))
