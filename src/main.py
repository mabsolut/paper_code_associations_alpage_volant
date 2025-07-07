import json

from CA_creation import CA_creation
from compute_impact_number_plots import compute_impact_number_plot
from data_preprocessing import compute_origin, load_and_clean_data
from data_simulation import (
    compute_impact_plots_simulation,
    report_alpha_error,
    simulate_alpha_error,
)
from DCi_degree_analysis import degree_vs_DCi
from dominance_shift import dominance_shift
from impact_plots_links import (
    difference_impact_plots_links,
    difference_impact_plots_positive_links,
    difference_impact_plots_links_sp,
    impact_plots_links,
)
from impact_plots_nodes import difference_impact_plots_nodes, impact_plots_nodes
from network_visualization import graph_creation, graph_creation_common

# === Load and prepare raw data ===
df = load_and_clean_data()
# Determine the origin (treatment or elevation) of each species
origin = compute_origin(df)

# === Temporal analysis of dominance structure shifts ===
# Figure 2: After five years of warming
dominance_shift(df, origin, 2021)

# Supplemental figures X: After 6 months and 2 years of warming
dominance_shift(df, origin, 2017)
dominance_shift(df, origin, 2018)

# === Network visualization of species associations ===
# Figure 3a, 3c, and SupFigure Xa for alpine, alpine warmed and subalpine communities
degrees, DCis = graph_creation(df, origin)

# === Estimate alpha error rate by simulating randomized data ===
# Applied separately for each climate treatment (very long step)
simulate_alpha_alpine = simulate_alpha_error(df, "G_CP")  # Alpine
simulate_alpha_warmed = simulate_alpha_error(df, "L_TP", n_iter=120)  # Alpine warmed
simulate_alpha_subalpine = simulate_alpha_error(df, "L_CP")  # Subalpine
# Long step, you can also use the stored json
with open(
    "save_json/json_alpha_error_G_CP.json",
    "r",
    encoding="utf-8",
) as f:
    simulate_alpha_alpine = json.load(f)
with open(
    "save_json/json_alpha_error_L_TP.json",
    "r",
    encoding="utf-8",
) as f:
    simulate_alpha_warmed = json.load(f)
with open(
    "save_json/json_alpha_error_L_CP.json",
    "r",
    encoding="utf-8",
) as f:
    simulate_alpha_subalpine = json.load(f)
report_alpha_error(simulate_alpha_alpine, "Alpine")
report_alpha_error(simulate_alpha_warmed, "Warmed")
report_alpha_error(simulate_alpha_subalpine, "Subalpine")


# === Figure 3b, 3d, SupFigure Xb: correlation between DCi and node degree ===
degree_vs_DCi(degrees, DCis, origin)

# === Visualization of the network built from alpine and all alpine warmed plots ===
graph_creation_common(df, degrees)

# === Construction of Correspondance analysis (CA) and associated boxplots ===
# Figure 4: construction with all data
CA_creation(df, degrees, [2017, 2018, 2021])

# Supplemental figure X: CA for individual years
CA_creation(df, degrees, [2017])
CA_creation(df, degrees, [2018])
CA_creation(df, degrees, [2021])

# === Evaluate the effect of sample size (number of plots) on network properties ===
# For both empirical and simulated data (alpine and alpine warmed)
impact_plots_alpine = compute_impact_number_plot(df, "G_CP")
impact_plots_warmed = compute_impact_number_plot(df, "L_TP")
impact_plots_warmed_simulation = compute_impact_plots_simulation(df, "L_TP")

# Long step, you can also use the stored json
with open(
    "save_json/json_impact_number_plot_G_CP.json",
    "r",
    encoding="utf-8",
) as f:
    impact_plots_alpine = json.load(f)
with open(
    "save_json/json_impact_number_plot_L_TP.json",
    "r",
    encoding="utf-8",
) as f:
    impact_plots_warmed = json.load(f)
with open(
    "save_json/json_impact_plots_simulation_L_TP.json",
    "r",
    encoding="utf-8",
) as f:
    impact_plots_warmed_simulation = json.load(f)

# === Link accumulation analysis ===
# Plots the number of links vs. number of plots, for both empirical and simulated alpine warmed data
impact_plots_links("warmed", impact_plots_warmed)
impact_plots_links("warmed_simulation", impact_plots_warmed_simulation)

# Statistical comparison of link accumulation slopes between alpine warmed and alpine networks
difference_impact_plots_links(impact_plots_alpine, impact_plots_warmed)

# Statistical comparison of positive link accumulation slopes between alpine warmed and alpine networks
difference_impact_plots_positive_links(impact_plots_alpine, impact_plots_warmed)

# Statistical comparison of link accumulation slopes between alpine warmed and alpine networks for all structuring species
structuring_sp = [
    "Tri alpi",
    "Car semp",
    "Poa alpi",
    "Pot aure",
    "Alo gera",
    "Fes nigr",
    "Pla alpi",
    "Hel vers",
    "Nar stri",
    "Agr capi",
    "Pla mari",
]
difference_impact_plots_links_sp(
    impact_plots_alpine, impact_plots_warmed, structuring_sp
)

# === Node accumulation analysis ===
# Same as above but for number of nodes in the network
impact_plots_nodes("warmed", impact_plots_warmed)
impact_plots_nodes("warmed_simulation", impact_plots_warmed_simulation)
difference_impact_plots_nodes(impact_plots_alpine, impact_plots_warmed)
