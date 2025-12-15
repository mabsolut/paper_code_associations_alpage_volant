import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


def degree_shift(df, degrees, origin, pseudo_degrees=False):
    """
    Analysis of degree shifts under experimental warming.
    """
    df_y = df[df["Site_Treatment"].isin(["G_CP", "L_TP"]) & df["Subplot"].isin(["B", "0"])]
    degree_a = degrees["G_CP"]
    degree_w = degrees["L_TP"]

    vals_a, vals_w, sp_labels = [], [], []
    vals_a_quantile, vals_w_quantile = [], []

    for sp in sorted(set(df_y["Species"].unique())):
        vals_a.append(degree_a.get(sp, 0))
        vals_w.append(degree_w.get(sp, 0))
        sp_labels.append(sp)
    for sp in sorted(set(df_y[df_y["Site_Treatment"] == "G_CP"]["Species"].unique())):
        vals_a_quantile.append(degree_a.get(sp, 0))
    for sp in sorted(set(df_y[df_y["Site_Treatment"] == "L_TP"]["Species"].unique())):
        vals_w_quantile.append(degree_w.get(sp, 0))

    # === Confidence envelopes ===
    delta = np.array(vals_w) - np.array(vals_a)
    lower, upper = np.quantile(delta, [0.1, 0.9])
    lims = [min(vals_a + vals_w), max(vals_a + vals_w) + 0.05]
    x_plot = np.linspace(*lims, 500)
    y_low = x_plot + lower
    y_up = x_plot + upper

    # === Plot ===
    # Correlation
    plt.figure(figsize=(6, 6))
    plt.plot(lims, lims, "k--", linewidth=1, label="y = x")
    plt.plot(x_plot, y_low, "r--", linewidth=1, label="Lower 90% conf.")
    plt.plot(x_plot, y_up, "r--", linewidth=1, label="Upper 90% conf.")

    # Structuring thresholds
    x90, y90 = np.quantile(vals_a_quantile, 0.9), np.quantile(vals_w_quantile, 0.9)
    plt.axvline(
        x=x90, color="blue", linestyle="--", label="Spatially structuring (Alpine)"
    )
    plt.axhline(
        y=y90, color="green", linestyle="--", label="Spatially structuring (Warmed)"
    )

    # Species points + labels
    x_all, y_all = [], []
    for x, y, sp in zip(vals_a, vals_w, sp_labels):
        x_all.append(x)
        y_all.append(y)
        plt.plot(x, y, "o", color=origin.get(sp, "black"), markersize=8)
        plt.text(x, y, sp, fontsize=9, ha="left", va="bottom", clip_on=True)

    # Correlation
    rho, p_val = spearmanr(x_all, y_all)
    print("=== Degrees spearman correlations (alpine DCi vs alpine warmed DCi) ===")
    print(f"Spearman's ρ = {rho:.3f}, p-value = {p_val:.1e}")

    # Formatting
    plt.xlabel("Degrees in alpine site")
    plt.ylabel("Degrees in alpine warmed treatment")
    plt.title("Degrees shift under artificial warming", pad=10)
    plt.grid(True)

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#0072b2ff",
            label="Alpine specialist",
            linestyle="",
        ),
        Line2D(
            [0], [0], marker="o", color="#e69f00fe", label="Colonizer", linestyle=""
        ),
        Line2D(
            [0], [0], marker="o", color="grey", label="Non-specialist", linestyle=""
        ),
    ]
    plt.legend(handles=legend_elements, loc="best")

    # Axis limits with padding
    x_pad = (max(vals_a) - min(vals_a)) * 0.01
    y_pad = (max(vals_w) - min(vals_w)) * 0.015
    plt.xlim(min(vals_a) - x_pad, max(vals_a) + 0.05)
    plt.ylim(min(vals_w) - y_pad, max(vals_w) + 0.05)

    if not pseudo_degrees:
        plt.tight_layout()
        plt.savefig(
            f"save_result/Degree_shift.svg",
            format="svg",
            dpi=300,
        )
    else :
        plt.tight_layout()
        plt.savefig(
            f"save_result/graphs/Pseudo_degree_shift.svg",
            format="svg",
            dpi=300,
        )