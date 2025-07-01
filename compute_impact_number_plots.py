import json
import multiprocessing as mp
import random
from collections import defaultdict

import pandas as pd
from association_detection import detect_species_associations


def compute_impact_number_plot(df, treatment):
    """
    Computes how the number of detected species associations (links) increases
    with the number of observed sampling plots, for a given treatment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset after preprocessing. Must include columns:
        'Year', 'Site_Treatment', 'Subplot', 'Replicate', 'Species', 'pinpoint'.
    treatment : str
        Treatment group to analyze (e.g., 'G_CP', 'L_TP').

    Returns
    -------
    result : dict
        Dictionary where keys are number of used plots (int) and values are lists
        of association dictionaries, as returned by `detect_species_associations`.
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
            subdf = pd.concat(selected_groups)
            all_subdfs.append(subdf)
            ns.append(n)

    with mp.Pool(12) as pool:
        res_list = pool.map(detect_species_associations, all_subdfs)

    for n, res_dict in zip(ns, res_list):
        result[n].append(res_dict)

    with open(
        f"/home/alpage_volant/alpage_volant_interaction/save_json/json_impact_number_plot_{treatment}.json",
        "w",
    ) as f:
        json.dump(result, f)

    return result
