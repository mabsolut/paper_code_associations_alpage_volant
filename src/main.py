import json

from CA_creation import CA_creation
from effect_size import effect_size, species_pseudo_degree
from data_preprocessing import (
    compute_origin,
    load_and_clean_data,
    abund_freq_vs_DCi,
)
from DCi_vs_degree import degree_vs_DCi
from degree_shift import degree_shift
from dominance_shift import dominance_shift
from effect_size_links import (
    difference_impact_plots_links,
    difference_impact_plots_positive_links,
    impact_plots_links,
)
from effect_size_nodes import difference_impact_plots_nodes, impact_plots_nodes
from network_visualization import graph_creation

precompute = True
# === Load and prepare raw data ===
df = load_and_clean_data()
# Determine for each species if it is alpine specialist, colonizer or non-specialist
origin = compute_origin(df)

# === Dominance structure shifts ===
# Figure 2a: After five years of warming
dominance_shift(df, origin, 2021)
# After 6 months and 2 years of warming, SuppFig. 2
dominance_shift(df, origin, 2017)
dominance_shift(df, origin, 2018)

# === Network visualization of species associations ===
# Figure 3a, 3c, and SuppFig. 4a for alpine, alpine warmed and subalpine communities
degrees, DCis = graph_creation(df, origin)

# === Figure 3b, 3d, SuppFig. 4b: correlation between DCi and node degree ===
degree_vs_DCi(degrees, DCis, origin)

# === Figure 2b: Degree shift with warming ===
degree_shift(df, degrees, origin)

# === Figure 4: Construction of the Correspondance analysis (CA) and associated boxplots ===
CA_creation(df, degrees)

# === What drives DCi: SuppFig. 1 ===
abund_freq_vs_DCi(df, "G_CP", origin)
abund_freq_vs_DCi(df, "L_TP", origin)

# === Effect of the prefilter on our conclusions ===
# Network visualization of species associations without prefilter: SuppFig. 3a and 3c
degrees_noprefilter, DCis_noprefilter = graph_creation(df, origin, prefilter = False)
# SuppFig. 3b, 3d: correlation between DCi and node degree without pre-filter
degree_vs_DCi(degrees_noprefilter, DCis_noprefilter, origin, prefilter = False)

# === Evaluate the effect of sample size (number of communities used) on network properties ===
if not precompute:
    # For alpine warmed data
    impact_plots_warmed = effect_size(df, "L_TP")
    # For alpine data
    impact_plots_alpine = effect_size(df, "G_CP")
else:
    # Long step, you can also use the stored json
    with open(
        "save_json/json_effect_size_L_TP.json",
        "r",
        encoding="utf-8",
    ) as f:
        impact_plots_warmed = json.load(f)
    with open(
        "save_json/json_effect_size_G_CP.json",
        "r",
        encoding="utf-8",
    ) as f:
        impact_plots_alpine = json.load(f)

# === Link and node accumulation with the sample size analysis ===
# SuppFig. 5a: Number of links vs. number of communities used, for alpine warmed data
impact_plots_links("warmed", impact_plots_warmed)
# SuppFig. 5b: Number of nodes vs. number of communities used, for alpine warmed data
impact_plots_nodes("warmed", impact_plots_warmed)

# SuppFig. 6: Statistical comparison of link accumulation between alpine warmed and alpine networks
difference_impact_plots_links(impact_plots_alpine, impact_plots_warmed)
# SuppFig. 7: Statistical comparison of positive link accumulation slopes between alpine warmed and alpine networks
difference_impact_plots_positive_links(impact_plots_alpine, impact_plots_warmed)
# SuppFig. 8: Statistical comparison of node accumulation between alpine warmed and alpine networks
difference_impact_plots_nodes(impact_plots_alpine, impact_plots_warmed)

# SuppFig. 9: The strong shift in the identity of the most structuring species under warming is robust to effect size.
pseudo_degrees = dict()
pseudo_degrees["G_CP"] = species_pseudo_degree(df, "G_CP", impact_plots_alpine)
pseudo_degrees["L_TP"] = species_pseudo_degree(df, "L_TP", impact_plots_warmed)
degree_shift(df, pseudo_degrees, origin, pseudo_degrees=True)