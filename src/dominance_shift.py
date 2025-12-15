import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import compute_DCi
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


def dominance_shift(df, origin, year):
    """
    Analysis of dominance structure shifts under experimental warming.
    """
    df_y = df[df["Year"] == year]
    dci_a = compute_DCi(df_y[df_y["Site_Treatment"] == "G_CP"])
    dci_w = compute_DCi(df_y[df_y["Site_Treatment"] == "L_TP"])

    vals_a, vals_w, sp_labels = [], [], []

    for sp in sorted(set(dci_a) | set(dci_w)):
        vals_a.append(dci_a.get(sp, 0))
        vals_w.append(dci_w.get(sp, 0))
        sp_labels.append(sp)
    vals_a_quantile = [x for x in vals_a if x != 0]
    vals_w_quantile = [x for x in vals_w if x != 0]

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

    # Dominant thresholds
    x90, y90 = np.quantile(vals_a_quantile, 0.9), np.quantile(vals_w_quantile, 0.9)
    plt.axvline(x=x90, color="blue", linestyle="--", label="Dominant (Alpine)")
    plt.axhline(y=y90, color="green", linestyle="--", label="Dominant (Warmed)")

    # Species points + labels
    a, ns, c = 0, 0, 0
    x_all, y_all = [], []
    for x, y, sp in zip(vals_a, vals_w, sp_labels):
        x_all.append(x)
        y_all.append(y)
        plt.plot(x, y, "o", color=origin.get(sp, "black"), markersize=8)
        plt.text(x, y, sp, fontsize=9, ha="left", va="bottom", clip_on=True)
        if origin.get(sp, "black")=="grey":
            ns+=1
        elif origin.get(sp, "black")=="#0072b2ff":
            a+=1
        elif origin.get(sp, "black")=="#e69f00fe":
            c+=1
    if year == 2021:
        print("=== Spearman correlations (alpine DCi vs alpine warmed DCi) ===")
    print(year)
    print("Nb alpine specialist:", a)
    print("Nb non-specialist:", ns)
    print("Nb colonizer:", c)

    # Correlation
    rho, p_val = spearmanr(x_all, y_all)
    print(f"{year}: Spearman's ρ = {rho:.3f}, p-value = {p_val:.1e}")

    # Formatting
    plt.xlabel("Dominance candidate index in alpine site")
    plt.ylabel("Dominance candidate index in alpine warmed treatment")
    plt.title("Dominance structure shift after 5 years of artificial warming", pad=10)
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

    # plt.tight_layout(pad=0)
    plt.tight_layout()
    plt.savefig(
        f"save_result/Dominance/Dominance_shift_{year}.svg",
        format="svg",
        dpi=300,
    )
