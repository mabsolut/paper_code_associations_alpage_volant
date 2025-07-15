import copy
from collections import defaultdict
from itertools import combinations

import numpy as np
from gaste_test import get_pval_comb
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def detect_species_associations(
    df, plot_impact: bool = False, plot_effect: bool = False, random_state: bool = False
):
    """
    Detect significant species associations using hypergeometric tests
    by combining replicate-level results with GASTE and controlling FDR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['Year', 'Site_Treatment', 'Subplot', 'Replicate', 'Species', 'pinpoint']
        Each row represents one species detected on a pinpoint.

    Returns
    -------
    result : dict
        {
        'under': dict of significantly negative associations,
        'over': dict of significantly positive associations,
        "importance_under": dict of the importance of all the plots involve in all the significant
        negative associations
        }
    """
    pvals_under, pvals_over = defaultdict(dict), defaultdict(dict)
    params_all = defaultdict(list)
    min_under, min_over = defaultdict(list), defaultdict(list)
    params_min_under, params_min_over = defaultdict(list), defaultdict(list)
    threshold_compute_explicite = 1e5
    disable_tqdm = False
    if random_state:
        np.random.seed()
        disable_tqdm = True

    for plot, subdf in df.groupby(["Year", "Site_Treatment", "Subplot", "Replicate"]):
        plot = f"{int(plot[0])}_{plot[1]}_{plot[2]}_{int(plot[3])}"
        spp = sorted(subdf["Species"].unique())
        pinpoints = subdf.groupby("pinpoint")["Species"].apply(set)
        N = len(pinpoints)

        # Co_occurences compte
        co_occs = {f"{sp1}|{sp2}": 0 for sp1, sp2 in combinations(spp, 2)}
        freq = dict.fromkeys(spp, 0)

        for sp1, sp2 in combinations(spp, 2):
            for p in pinpoints:
                if sp1 in p and sp2 in p:
                    co_occs[f"{sp1}|{sp2}"] += 1

        for species in pinpoints:
            for sp in species:
                freq[sp] += 1

        for pair, k in co_occs.items():
            sp1, sp2 = pair.split("|")
            n1, n2 = freq[sp1], freq[sp2]
            dist = hypergeom(N, n1, n2)
            if random_state == True:
                k = len(
                    set(np.random.choice(range(N), n1, replace=False))
                    & set(np.random.choice(range(N), n2, replace=False))
                )

            # observed p-values
            pvals_under[pair][plot] = dist.cdf(k)
            pvals_over[pair][plot] = dist.sf(k - 1)
            params_all[pair].append((N, n1, n2, k))

            # minimum bornes
            min_k_under = max(0, n1 + n2 - N)
            min_k_over = min(n1, n2) - 1
            min_under[pair].append(dist.cdf(min_k_under))
            min_over[pair].append(dist.sf(min_k_over))
            params_min_under[pair].append((N, n1, n2, min_k_under))
            params_min_over[pair].append((N, n1, n2, min_k_over))

    # prefiltering with the minimum p-values combined
    min_comb_under = {
        pair: get_pval_comb(
            params_min_under[pair],
            min_under[pair],
            "under",
            moment=2,
            tau=1,
            threshold_compute_explicite=threshold_compute_explicite,
        )
        for pair in tqdm(
            min_under,
            desc="Combining minimum achievable p-values under (prefiltering)",
            disable=disable_tqdm,
        )
    }
    min_comb_over = {
        pair: get_pval_comb(
            params_min_over[pair],
            min_over[pair],
            "over",
            moment=2,
            tau=1,
            threshold_compute_explicite=threshold_compute_explicite,
        )
        for pair in tqdm(
            min_over,
            desc="Combining minimum achievable p-values over (prefiltering)",
            disable=disable_tqdm,
        )
    }

    # FDR filtering
    def fdr_select(pdict):
        pvals = list(pdict.values())
        rej, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")
        return {k: pvals_adj[i] for i, (k, r) in enumerate(zip(pdict.keys(), rej)) if r}

    # FDR for the prefilter
    keep_under = fdr_select(min_comb_under)
    keep_over = fdr_select(min_comb_over)

    # Final p-values combinaison
    comb_under, comb_over = {}, {}

    for pair in tqdm(
        keep_under.keys(), desc="Combining p-values under", disable=disable_tqdm
    ):
        comb_under[pair] = get_pval_comb(
            params_all[pair],
            list(pvals_under[pair].values()),
            "under",
            moment=2,
            tau=1,
            threshold_compute_explicite=threshold_compute_explicite,
        )

    for pair in tqdm(
        keep_over.keys(), desc="Combining p-values over", disable=disable_tqdm
    ):
        comb_over[pair] = get_pval_comb(
            params_all[pair],
            list(pvals_over[pair].values()),
            "over",
            moment=2,
            tau=1,
            threshold_compute_explicite=threshold_compute_explicite,
        )

    # FDR
    if random_state:
        final_under = fdr_select(comb_under)
        final_over = fdr_select(comb_over)
        d_under = dict()
        for pair, adjusted_fdr_bh_pval in final_under.items():
            d_under[pair] = {
                "comb_pval": comb_under[pair],
                "adjusted_fdr_bh_pval": adjusted_fdr_bh_pval,
                "params": params_all[pair],
                "pvals": pvals_under[pair],
            }
        d_over = dict()
        for pair, adjusted_fdr_bh_pval in final_over.items():
            d_over[pair] = {
                "comb_pval": comb_over[pair],
                "adjusted_fdr_bh_pval": adjusted_fdr_bh_pval,
                "params": params_all[pair],
                "pvals": pvals_over[pair],
            }
        result = {"under": d_under, "over": d_over}
    else:
        result = {"under": fdr_select(comb_under), "over": fdr_select(comb_over)}

    if plot_impact == True:
        importance_under = defaultdict(dict)
        for pair in fdr_select(comb_under).keys():
            pval = comb_under[pair]
            for plot in pvals_under[pair].keys():
                new_dict = copy.deepcopy(pvals_under[pair])
                del new_dict[plot]
                new_pval = get_pval_comb(
                    params_all[pair],
                    list(new_dict.values()),
                    "under",
                    moment=2,
                    tau=1,
                    threshold_compute_explicite=threshold_compute_explicite,
                )
                importance_under[pair][plot] = new_pval - pval
        result["importance_under"] = importance_under

    if plot_effect == True:
        effect_under = defaultdict(dict)
        for pair in fdr_select(comb_under).keys():
            pval = comb_under[pair]
            for plot in pvals_under[pair].keys():
                new_dict = copy.deepcopy(pvals_under[pair])
                del new_dict[plot]
                new_pval = get_pval_comb(
                    params_all[pair],
                    list(new_dict.values()),
                    "under",
                    moment=2,
                    tau=1,
                    threshold_compute_explicite=threshold_compute_explicite,
                )
                new_comb_under = copy.deepcopy(comb_under)
                new_comb_under[pair] = new_pval
                effect_under[pair][plot] = (
                    1
                    if len(fdr_select(new_comb_under)) < len(fdr_select(comb_under))
                    else 0
                )
        result["effect_under"] = effect_under

        effect_over = defaultdict(dict)
        for pair in fdr_select(comb_over).keys():
            for plot in pvals_over[pair].keys():
                new_dict = copy.deepcopy(pvals_over[pair])
                del new_dict[plot]
                new_pval = get_pval_comb(
                    params_all[pair],
                    list(new_dict.values()),
                    "over",
                    moment=2,
                    tau=1,
                    threshold_compute_explicite=threshold_compute_explicite,
                )
                new_comb_over = copy.deepcopy(comb_over)
                new_comb_over[pair] = new_pval
                new_keep_over = fdr_select(new_comb_over)
                effect_over[pair][plot] = (
                    1
                    if len(fdr_select(new_comb_over)) < len(fdr_select(comb_over))
                    else 0
                )
        result["effect_over"] = effect_over

    return result
