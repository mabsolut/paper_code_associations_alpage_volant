import multiprocessing as mp

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from association_detection import detect_species_associations, detect_species_associations_noprefilter
from data_preprocessing import compute_DCi


def graph_creation(df, origin, jobs=-1, prefilter = True):
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
    if prefilter == True:
        with mp.Pool(jobs) as pool:
            res_list = pool.map(
                detect_species_associations,
                [group for _, group in subdf_grouped],
            )
    else : 
        with mp.Pool(jobs) as pool:
            res_list = pool.map(
                detect_species_associations_noprefilter,
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
    if prefilter:
        print("=== Network summary ===")
    else:
        print("=== Network summary without prefilter ===")
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
                ["#009e73ff", "blue", "#6e6e6eff"]
                if t == "G_CP"
                else ["#cc79a7ff", "blue", "#6e6e6eff"]
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
                    else "#009e73ff" if t == "G_CP" else "#cc79a7ff"
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
            color = "black"
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

        if prefilter == True:
            plt.savefig(
                f"save_result/graphs/graphs_{t}.svg",
                format="SVG",
                dpi=300,
            )
        else:
            plt.savefig(
                f"save_result/graphs/graphs_{t}_noprefilter.svg",
                format="SVG",
                dpi=300,
            )
    return degrees, DCis
