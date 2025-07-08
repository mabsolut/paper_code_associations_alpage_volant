from itertools import combinations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import prince
from data_preprocessing import compute_DCi
from matplotlib.colors import to_rgba
from scipy.spatial import ConvexHull


def weighted_permutation_test(x, w_x, y, w_y, n_permutations=10000):
    """
    Conducts a weighted permutation test to assess the significance of the difference
    in weighted means between two independent groups.

    Parameters
    ----------
    x : array-like
        Numeric values for group 1.
    w_x : array-like
        Corresponding weights for group 1.
    y : array-like
        Numeric values for group 2.
    w_y : array-like
        Corresponding weights for group 2.
    n_permutations : int, optional (default=10000)
        Number of permutations to perform for estimating the null distribution.

    Returns
    -------
    pval : float
        Two-sided p-value estimating the probability that the observed difference
        in weighted means could arise under the null hypothesis of no group effect.

    Method
    ------
    The test computes the observed difference in weighted averages between the two groups.
    It then generates a null distribution by repeatedly permuting the combined dataset,
    recalculating the weighted mean difference for each permutation.
    The p-value is estimated as the proportion of permuted differences at least as extreme
    as the observed difference.
    """
    obs_diff = np.average(x, weights=w_x) - np.average(y, weights=w_y)

    # Permutation test
    combined = np.concatenate([x, y])
    weights = np.concatenate([w_x, w_y])
    count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(len(combined))
        x_perm = combined[perm[: len(x)]]
        y_perm = combined[perm[len(x) :]]
        wx_perm = weights[perm[: len(x)]]
        wy_perm = weights[perm[len(x) :]]
        diff = np.average(x_perm, weights=wx_perm) - np.average(y_perm, weights=wy_perm)
        if abs(diff) >= abs(obs_diff):
            count += 1
    pval = count / n_permutations
    return max(pval, 1 / n_permutations)


def r2_weighted_permutation_test(x, w_x, y, w_y):
    """
    Computes a weighted pseudo-R² value quantifying the proportion of variance
    explained by grouping in weighted data.

    Parameters
    ----------
    x : array-like
        Numeric values for group 1.
    w_x : array-like
        Corresponding weights for group 1.
    y : array-like
        Numeric values for group 2.
    w_y : array-like
        Corresponding weights for group 2.

    Returns
    -------
    r2_weighted : float
        Weighted pseudo-R² value representing the fraction of weighted variance
        explained by differences between the two groups.

    Method
    ------
    The weighted pseudo-R² is calculated as the ratio of between-group weighted
    sum of squares to the total weighted sum of squares, where weighting accounts
    for heteroscedasticity or varying observation importance.
    """
    all_proj = np.concatenate([x, y])
    all_weights = np.concatenate([w_x, w_y])
    ss_total = np.sum(
        all_weights * (all_proj - np.average(all_proj, weights=all_weights)) ** 2
    )
    ss_between = (
        np.sum(w_x)
        * (np.average(x, weights=w_x) - np.average(all_proj, weights=all_weights)) ** 2
        + np.sum(w_y)
        * (np.average(y, weights=w_y) - np.average(all_proj, weights=all_weights)) ** 2
    )
    return ss_between / ss_total


def replicate_by_weight(values, weights, total_count=100):
    """
    Generates a replicated array of values proportional to their relative weights,
    facilitating visualization of weighted distributions, for example using boxplots.

    Parameters
    ----------
    values : array-like
        Numeric data points to be replicated.
    weights : array-like
        Relative weights corresponding to each value. Should sum to 1 or be
        normalized accordingly.
    total_count : int, optional (default=100)
        Total number of replicated elements to generate, controlling the scale
        of the output array.

    Returns
    -------
    replicated : list
        List of values replicated according to their weighted proportions,
        suitable for use in standard plotting functions that do not support weights.
    """
    replicated = []
    for v, w in zip(values, weights):
        replicated.extend([v] * int(round(w * total_count)))
    return replicated


def CA_creation(df, degrees, y):
    """
    Performs Correspondence Analysis (CA) on species abundance data filtered by year(s),
    visualizes treatment group convex hulls, species projections, and statistical
    comparisons of species' DCi and network degree values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns ['Site_Treatment', 'Year', 'Subplot', 'Replicate', 'Species'].
    degrees : dict of dicts
        Nested dictionary of species degrees per treatment, e.g. degrees['G_CP'][species].
    y : list or array-like
        Year(s) to filter the data on.
    """
    # === Create abundance table (plots x species) ===
    df_CA = df[df["Site_Treatment"] != "G_TP"]
    df_CA = df_CA[df_CA["Year"].isin(y)]
    abundance_table = (
        df_CA.groupby(["Year", "Site_Treatment", "Subplot", "Replicate"])["Species"]
        .value_counts()
        .unstack(fill_value=0)
    )
    abundance_table.index = abundance_table.index.map(
        lambda x: " | ".join(str(i) for i in x)
    )

    # === Run Correspondence Analysis (CA) ===
    ca = prince.CA(
        n_components=2, n_iter=10, copy=True, check_input=True, engine="scipy"
    )
    ca = ca.fit(abundance_table)
    row_coords = ca.row_coordinates(abundance_table)
    col_coords = ca.column_coordinates(abundance_table)
    # Extract x-axis values (1st CA dimension)
    coord = col_coords.iloc[:, 0]
    percent_axis1 = ca.eigenvalues_[0] / sum(ca.eigenvalues_) * 100
    if len(y) == 0:
        print(f"\n\n=== Community Assemblage (CA) – Year {y[0]} ===")
    else:
        print(f"\n\n=== Community Assemblage (CA) – All years ===")
    print(f"\nAxe 1 explains {percent_axis1:.2f}% of inertia")

    # === Define treatment groups and colors ===
    group_colors = {"G_CP": "#52cfebff", "L_TP": "#ff544bff", "L_CP": "#feff63ff"}
    plot_treatment = {grp: [] for grp in group_colors.keys()}

    # Group samples by treatment for hull computation
    for i, plot in enumerate(row_coords.index):
        treatment = plot.split(" | ")[1]
        plot_treatment[treatment].append(row_coords.iloc[i].values)

    # === Plot treatment convex hulls on CA plot ===
    _, ax = plt.subplots(figsize=(12, 12))
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)

    for treatment, plot in plot_treatment.items():
        plot = np.array(plot)
        hull = ConvexHull(plot)
        color = group_colors[treatment]
        darker = tuple(c * 0.7 for c in mcolors.to_rgba(color)[:3]) + (1.0,)
        ax.fill(plot[hull.vertices, 0], plot[hull.vertices, 1], color=color, alpha=0.6)
        ax.plot(plot[hull.vertices, 0], plot[hull.vertices, 1], color=darker, lw=2)
        ax.plot(
            [plot[hull.vertices[-1], 0], plot[hull.vertices[0], 0]],
            [plot[hull.vertices[-1], 1], plot[hull.vertices[0], 1]],
            color=darker,
            lw=2,
        )

    # === Plot species colored and sized by its presence in networks on CA plot ===
    sp_colors, sp_sizes = {}, {}
    for sp in df_CA["Species"].unique():
        degree_alpine = degrees["G_CP"].get(sp, 0)
        degree_warmed = degrees["L_TP"].get(sp, 0)

        if degree_alpine != 0 and degree_warmed != 0:
            sp_colors[sp] = "#6e6e6eff"  # present in both
        elif degree_alpine != 0:
            sp_colors[sp] = "#BF00FF"  # alpine only
        elif degree_warmed != 0:
            sp_colors[sp] = "#fbaf00"  # warmed only

        # Average if present in both, otherwise keep single value
        sp_sizes[sp] = (
            (degree_alpine + degree_warmed) / 2
            if degree_alpine and degree_warmed
            else degree_alpine + degree_warmed
        )

    for i, sp in enumerate(col_coords.index):
        ax.scatter(
            col_coords.iloc[i, 0],
            col_coords.iloc[i, 1],
            color=sp_colors.get(sp),
            s=sp_sizes.get(sp, 0) * 50,
            zorder=5,
        )
        ax.text(
            col_coords.iloc[i, 0],
            col_coords.iloc[i, 1],
            str(sp),
            fontsize=7,
            color="black",
            zorder=6,
        )

    # === Compute DCi and extract values per plot ===
    DCi_vals = [
        compute_DCi(df_CA[df_CA["Site_Treatment"] == k])
        for k in ["G_CP", "L_TP", "L_CP"]
    ]
    coord_DCi = [[coord[sp] for sp in DCi if sp in coord] for DCi in DCi_vals]
    weights_DCi = [list(DCi.values()) for DCi in DCi_vals]
    boxplots_DCi = [replicate_by_weight(x, w) for x, w in zip(coord_DCi, weights_DCi)]

    # === Extract degree values per plot ===
    coord_degrees = [
        [coord[sp] for sp in degree if sp in coord] for degree in degrees.values()
    ]
    weights_degrees = [list(degree.values()) for degree in degrees.values()]

    # === Plot DCi boxplots above CA species projection ===
    labels_box = ["Alpine", "Alpine warmed", "Subalpine"]
    colors_box = ["#52cfebff", "#ff544bff", "#feff63ff"]
    y_pos_DCi = [5.0, 5.6, 6.2]
    for i, (group, y_pos, color) in enumerate(zip(boxplots_DCi, y_pos_DCi, colors_box)):
        ax.boxplot(
            group,
            positions=[y_pos],
            vert=False,
            patch_artist=True,
            widths=0.4,
            boxprops=dict(facecolor=to_rgba(color, alpha=0.6)),
            medianprops=dict(color="black"),
            flierprops=dict(markerfacecolor=color, marker="o", alpha=0.4),
        )

    # === Plot degrees boxplots above CA species projection ===
    boxplots_degrees = [
        [
            coord[sp]
            for sp, w in degrees[k].items()
            if sp in coord.index
            for _ in range(int(w))
        ]
        for k in ["G_CP", "L_TP", "L_CP"]
    ]
    y_pos_deg = [7.6, 8.2, 8.8]
    for i, (group, y_pos, color) in enumerate(
        zip(boxplots_degrees, y_pos_deg, colors_box)
    ):
        ax.boxplot(
            group,
            positions=[y_pos],
            vert=False,
            patch_artist=True,
            widths=0.4,
            boxprops=dict(facecolor=to_rgba(color, alpha=0.6)),
            medianprops=dict(color="black"),
            flierprops=dict(markerfacecolor=color, marker="o", alpha=0.4),
        )

    # === Statistical comparisons and pseudo-R² computation ===
    print("\n--- DCi comparisons ---")
    for i, j in combinations(range(3), 2):
        pval = weighted_permutation_test(
            coord_DCi[i], weights_DCi[i], coord_DCi[j], weights_DCi[j]
        )
        pseudo_r2 = r2_weighted_permutation_test(
            coord_DCi[i], weights_DCi[i], coord_DCi[j], weights_DCi[j]
        )
        print(
            f"{labels_box[i]} vs {labels_box[j]}: p = {pval:.1e}, pseudo-R² = {pseudo_r2:.2f}"
        )

    print("\n--- Degree comparisons ---")
    for i, j in combinations(range(3), 2):
        pval = weighted_permutation_test(
            coord_degrees[i], weights_degrees[i], coord_degrees[j], weights_degrees[j]
        )
        pseudo_r2 = r2_weighted_permutation_test(
            coord_degrees[i], weights_degrees[i], coord_degrees[j], weights_degrees[j]
        )
        print(
            f"{labels_box[i]} vs {labels_box[j]}: p = {pval:.1e}, pseudo R² = {pseudo_r2:.2f}"
        )

    # === Final layout ===
    ax.invert_yaxis()
    ax.set_xlabel("CA Axis 1 (78%): alpine-subalpine gradient")
    ax.set_ylabel("CA Axis 2")
    ax.set_title("CA ordination of species abundances")
    plt.tight_layout()
    if len(y) == 3:
        plt.savefig(
            "save_result/CA/CA.svg",
            format="svg",
            dpi=300,
        )
    else:
        plt.savefig(
            f"save_result/CA/CA_{y[0]}.svg",
            format="svg",
            dpi=300,
        )
