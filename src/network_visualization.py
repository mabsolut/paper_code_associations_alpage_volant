import json
import multiprocessing as mp

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

from association_detection import detect_species_associations
from data_preprocessing import compute_DCi


def graph_creation(df, origin, jobs=-1):
    """
    Visualisation of species spatial association networks across treatments.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing species observations with columns 'Site_Treatment', 'Subplot', and 'Species'.
    origin : dict
        Dictionary mapping species to colors.

    Returns
    -------
    degrees : dict
        Species degree (number of connections) per treatment.
    DCis : dict
        Dominance Candidate Index per treatment.
    """
    treatments = ["G_CP", "L_TP", "L_CP"]
    subdf = df[df["Site_Treatment"].isin(treatments)]
    # Keep only one subplot by treatment (alpine, alpine warmed and subalpine) to be able to compare them after
    subdf = subdf[(subdf["Subplot"] == "B") | (subdf["Subplot"] == "0")]

    # === Detect spatial associations ===
    subdf_grouped = subdf.groupby(["Site_Treatment"])
    if jobs == -1:
        jobs = mp.cpu_count()
    with mp.Pool(jobs) as pool:
        res_list = pool.map(
            detect_species_associations,
            [group for _, group in subdf_grouped],
        )
    result = {
        treatment: {
            "Site_Treatment": treatment,
            "res_dict": res_dict,
        }
        for treatment, res_dict in zip(subdf_grouped.indices.keys(), res_list)
    }

    # === Create reference layout for alpine and alpine warmed networks ===
    ref_graph = nx.Graph()
    for t in ["G_CP", "L_TP"]:
        for edge_type in ["under", "over"]:
            for link in result[t]["res_dict"][edge_type].keys():
                sp1, sp2 = link.split("|")
                ref_graph.add_edge(sp1, sp2, weight=1)
    pos_G = nx.kamada_kawai_layout(ref_graph)

    degrees, DCis = {}, {}
    print("=== Network summary ===")
    for t in treatments:
        _, ax = plt.subplots(figsize=(8, 6))
        subdf_t = subdf[subdf["Site_Treatment"] == t]
        DCi = compute_DCi(subdf_t)
        DCis[t] = DCi
        res = result[t]["res_dict"].copy()

        if t == "L_CP":
            edge_sets, edge_colors = [res["under"], res["over"]], ["red", "blue"]
        else:
            ref_key = "L_TP" if t == "G_CP" else "G_CP"
            res_ref = result[ref_key]["res_dict"].copy()

            # selection of links common to the alpine and alpine warmed networks
            shared_under = {
                k: v for k, v in res_ref["under"].items() if k in res["under"].keys()
            }
            # selection of links unique to the alpine or alpine warmed network
            unique_under = {
                k: v for k, v in res["under"].items() if k not in shared_under.keys()
            }

            edge_sets = [unique_under, res["over"], shared_under]
            edge_colors = (
                ["#BF00FF", "blue", "#6e6e6eff"]
                if t == "G_CP"
                else ["#fbaf00", "blue", "#6e6e6eff"]
            )

        # === Build graph ===
        g = nx.Graph()
        for color, edges in zip(edge_colors, edge_sets):
            g.add_weighted_edges_from(
                [
                    (
                        pair.split("|")[0],
                        pair.split("|")[1],
                        -np.log10(pval) / 2,
                    )
                    for pair, pval in edges.items()
                ],
                weight="pval",
                color=color,
            )

        node_sizes = [DCi.get(node, 0) * 300 for node in g.nodes()]

        if t == "L_CP":
            node_colors = [origin.get(node) for node in g.nodes()]
            pos = nx.spring_layout(g, seed=42)
        else:
            nodes_ref = {
                node for edge in res_ref["under"].keys() for node in edge.split("|")
            }
            node_colors = [
                (
                    "#6e6e6eff"
                    if node in nodes_ref
                    else "#BF00FF" if t == "G_CP" else "#fbaf00"
                )
                for node in g.nodes()
            ]
            pos = pos_G

        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(
            g,
            pos,
            edge_color=nx.get_edge_attributes(g, "color").values(),
            width=list(nx.get_edge_attributes(g, "pval").values()),
        )
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=6, font_color="black")
        ax.set_title(t)
        ax.axis("off")

        # === Plot absent species ===
        absent_sp = sorted(
            set(subdf_t["Species"].unique()) - set(g.nodes()),
            key=lambda sp: DCi.get(sp, 0),
            reverse=True,
        )
        for k, sp in enumerate(absent_sp):
            x = -1 + 0.08 * (k % 20)
            y = -1 - 0.08 * (k // 20)
            color = origin.get(sp) if t == "L_CP" else "black"
            ax.scatter(x, y, s=DCi.get(sp) * 300, color=color)

        order = len(g.nodes())
        size = len(g.edges())
        print(
            t,
            "- Order = ",
            order,
            "; Size = ",
            size,
            "; Connectance = ",
            size / (order * (order - 1) / 2),
        )
        degrees[t] = dict(g.degree())

        plt.savefig(
            f"save_result/graphs/graphs_{t}.svg",
            format="SVG",
            dpi=300,
        )
    return degrees, DCis


def graph_creation_common(df, degrees):
    """
    Visualizes spatial association links shared between the alpine control (G_CP)
    and alpine warmed (L_TP) communities, comparing interaction strength and
    detecting significant shifts represented in a combined network.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'Site_Treatment' and species observations.
    degrees : dict
        Dictionary of species degrees per treatment from previous analysis.

    Returns
    -------
    None
    """
    # Compute associations from pooled data
    pooled_df = df[df["Site_Treatment"].isin(["G_CP", "L_TP"])]
    res = detect_species_associations(pooled_df, plot_impact=True)
    with open("save_json/json_common.json", "w") as f:
        json.dump(res, f)

    # Compare negative association strengths between alpine and alpine warmed communities
    importance = res["importance_under"]

    stronger_a, stronger_w, unchanged = {}, {}, {}
    only_a, only_w = {}, {}

    for pair, pourcents in importance.items():
        vals_a = [
            p
            for plot, p in pourcents.items()
            if "_".join(plot.split("_")[1:3]) == "G_CP"
        ]
        vals_w = [
            p
            for plot, p in pourcents.items()
            if "_".join(plot.split("_")[1:3]) == "L_TP"
        ]
        if vals_a and vals_w:
            _, pval_less = mannwhitneyu(vals_a, vals_w, alternative="less")
            _, pval_greater = mannwhitneyu(vals_a, vals_w, alternative="greater")
            if pval_less <= 0.05:
                stronger_w[pair] = res["under"][pair]
            elif pval_greater <= 0.05:
                stronger_a[pair] = res["under"][pair]
            else:
                unchanged[pair] = res["under"][pair]
        elif vals_a:
            only_a[pair] = res["under"][pair]
        else:
            only_w[pair] = res["under"][pair]

    # Network building
    g = nx.Graph()

    color_map = {
        "unchanged": "#e6e6e6",
        "stronger_a": "#BF00FF",
        "stronger_w": "#fbaf00",
        "only_w": "yellow",
        "only_a": "pink",
    }

    for group, color in zip(
        [unchanged, stronger_a, stronger_w, only_w, only_a],
        color_map.values(),
    ):
        g.add_weighted_edges_from(
            [
                (
                    pair.split("|")[0],
                    pair.split("|")[1],
                    -np.log10(pval) / 2,
                )
                for pair, pval in group.items()
            ],
            weight="pval",
            color=color,
        )

    # Compute node properties
    DCi = compute_DCi(pooled_df)
    node_sizes, node_colors = [], []
    for node in g.nodes():
        if node in degrees["G_CP"].keys() and node in degrees["L_TP"].keys():
            node_colors.append("grey")
        elif node in degrees["G_CP"].keys():
            node_colors.append("#BF00FF")
        elif node in degrees["L_TP"].keys():
            node_colors.append("#fbaf00")
        else:
            node_colors.append("black")
        node_sizes.append(DCi.get(node) * 300)
    pos = nx.spring_layout(g, seed=42, k=0.4, iterations=200)

    # Plot of the network
    _, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(
        g, pos, node_color=node_colors, node_size=node_sizes, linewidths=1
    )
    edges_unchanged = [e for e in g.edges(data=True) if e[2]["color"] == "#e6e6e6"]
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=[(u, v) for u, v, _ in edges_unchanged],
        edge_color=[d["color"] for _, _, d in edges_unchanged],
        width=[d["pval"] for _, _, d in edges_unchanged],
    )
    edges_other = [e for e in g.edges(data=True) if e[2]["color"] != "#e6e6e6"]
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=[(u, v) for u, v, _ in edges_other],
        edge_color=[d["color"] for _, _, d in edges_other],
        width=[d["pval"] for _, _, d in edges_other],
    )
    nx.draw_networkx_labels(g, pos, font_size=6, font_color="black", ax=ax)

    # Adding of a pie chart summarizing changes
    labels = ["Decreased with warming", "Increased with warming"]
    sizes = [len(stronger_a), len(stronger_w)]
    colors_pie = [color_map["stronger_a"], color_map["stronger_w"]]
    inset_ax = ax.inset_axes([0.75, 0.05, 0.2, 0.2])
    inset_ax.pie(
        sizes,
        labels=labels,
        colors=colors_pie,
        startangle=90,
        autopct="%1.1f%%",
        textprops={"fontsize": 6},
    )
    inset_ax.set_title("Link bias", fontsize=6)

    # Legend
    legend_elements = [
        Line2D(
            [0], [0], color="#fbaf00", label="Increased with warming", linestyle="-"
        ),
        Line2D(
            [0], [0], color="#BF00FF", label="Decreased with warming", linestyle="-"
        ),
        Line2D([0], [0], color="yellow", label="Appeared with warming", linestyle="-"),
        Line2D([0], [0], color="pink", label="Disappeared with warming", linestyle="-"),
        Line2D(
            [0],
            [0],
            color="#e6e6e6",
            label="Did not change with warming",
            linestyle="-",
        ),
    ]
    plt.legend(handles=legend_elements, loc=(1, 1))

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"save_result/graphs/graphs_common.svg",
        format="SVG",
        dpi=300,
    )
