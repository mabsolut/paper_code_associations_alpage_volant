import json
import math
import multiprocessing as mp
import random
from collections import defaultdict
from itertools import combinations, islice

import pandas as pd
from association_detection import detect_species_associations


def compute_impact_number_plot(df, treatment, jobs=-1):
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

    if jobs == -1:
        jobs = mp.cpu_count()
    with mp.Pool(jobs) as pool:
        res_list = pool.map(detect_species_associations, all_subdfs)

    for n, res_dict in zip(ns, res_list):
        result[n].append(res_dict)

    with open(
        f"save_json/json_impact_number_plot_{treatment}.json",
        "w",
    ) as f:
        json.dump(result, f)

    return result


def compute_impact_number_plot_allcomb(df, treatment, max_comb_per_n=10, jobs=-1):
    """
    Computes all combinations of plots for each number n, and detects associations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after preprocessing.
    treatment : str
        Treatment group to analyze.
    max_comb_per_n : int
        Maximum number of combinations per n (to avoid explosion of memory/time).

    Returns
    -------
    result : dict
        Dictionary: key = number of plots (n), value = list of result dicts.
    """
    subdf = df[df["Site_Treatment"] == treatment]
    grouped = list(subdf.groupby(["Year", "Site_Treatment", "Subplot", "Replicate"]))
    nb_plots = len(grouped)

    result = defaultdict(list)
    all_subdfs = []
    ns = []

    for n in range(1, 31):
        comb_iter = combinations(grouped, n)
        if math.comb(nb_plots, n) <= max_comb_per_n:
            selected_combs = list(comb_iter)
        else:
            selected_combs = list(islice(comb_iter, max_comb_per_n))
        for comb in selected_combs:
            selected_groups = [group[1] for group in comb]
            subdf_comb = pd.concat(selected_groups)
            all_subdfs.append(subdf_comb)
            ns.append(n)

    if jobs == -1:
        jobs = mp.cpu_count()
    with mp.Pool(jobs) as pool:
        res_list = pool.map(detect_species_associations, all_subdfs)

    for n, res_dict in zip(ns, res_list):
        result[n].append(res_dict)

    with open(
        f"save_json/json_impact_number_plot_{treatment}_allcomb.json",
        "w",
    ) as f:
        json.dump(result, f)

    return result

