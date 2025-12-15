import json
import multiprocessing as mp
import random
from collections import defaultdict
import numpy as np

import pandas as pd
from association_detection import detect_species_associations


def effect_size(df, treatment, jobs=-1):
    """
    Computes how the number of detected species associations (links) increases with the number of observed sampling communities, for a given treatment.
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
        f"save_json/json_effect_size_{treatment}.json",
        "w",
    ) as f:
        json.dump(result, f)

    return result


def species_pseudo_degree(df, treatment, result):
    """
    Compute the accumulation of network links involving individual species (species 
    degrees) across increasing numbers of sampled communities.

    Parameters
    ----------
    result_alpine : dict
        Nested dictionary with link data for the Alpine treatment.
    result_warmed : dict
        Nested dictionary with link data for the Warmed treatment.
    list_sp : list of str
        List of species identifiers to analyze and compare individually.
    """
    slopes, sp_list = [], []

    for sp in df[df["Site_Treatment"] == treatment]["Species"].unique():
        x, y = [], []
        for n in result:
            for res_dict in result[str(n)]:
                if len(res_dict["under"]) + len(res_dict["over"]) > 0:
                    x.append(int(n))
                    count = sum(
                        sp in pair.split("|") for pair in res_dict["under"]
                    ) + sum(sp in pair.split("|") for pair in res_dict["over"])
                    y.append(count)
        slope = np.dot(x, y) / np.dot(x, x)
        if slope != 0:
            slopes.append(slope)
            sp_list.append(sp)

    degrees = {}
    for sp, degree in zip(sp_list, slopes):
        degrees[sp] = degree * 100
    return degrees