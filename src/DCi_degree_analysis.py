import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def degree_vs_DCi(degrees, DCis, origin):
    """
    Plot relationship between degree and dominance candidate index (DCi)
    across treatments with exponential fit and quantile thresholds.

    Parameters
    ----------
    degrees : dict of dict
        Species degree counts per treatment, e.g., degrees['G_CP'][species].
    DCis : dict of dict
        Dominance candidate index per treatment, e.g., DCis['G_CP'][species].
    origin : dict
        Mapping species to colors for plotting.
    """
    treatments = ["G_CP", "L_TP", "L_CP"]
    _, axes = plt.subplots(1, 3, figsize=(11, 5))
    for ax, t in zip(axes, treatments):
        x = np.array(list(DCis[t].values()))
        y = np.array([degrees[t].get(sp, 0) for sp in DCis[t].keys()])
        sp_color = [origin.get(sp, "black") for sp in DCis[t].keys()]
        ax.scatter(x, y, c=sp_color, zorder=5)
        for xi, yi, sp in zip(x, y, list(DCis[t].keys())):
            if yi != 0 or xi >= 0.3:
                ax.text(xi, yi, sp, ha="right", va="bottom", rotation=-45, zorder=6)

        def exp_offset(x, a, b, c):
            return c + a * np.exp(b * x)

        # Ajustement of the exponential model
        popt, _ = curve_fit(exp_offset, x, y, maxfev=10000)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = exp_offset(x_fit, *popt)
        r2 = 1 - (
            np.sum((y - exp_offset(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)
        )
        p_val = 1 - stats.f.cdf(
            (
                (
                    np.sum((y - np.mean(y)) ** 2)
                    - np.sum((y - exp_offset(x, *popt)) ** 2)
                )
                / 2
            )
            / (np.sum((y - exp_offset(x, *popt)) ** 2) / (len(y) - 3)),
            2,
            len(y) - 3,
        )

        ax.plot(x_fit, y_fit, color="red", label=rf"$R^2$={r2:.2f}, $p$={p_val:.2e}")
        ax.axvline(np.quantile(x, 0.9), color="blue", linestyle="--", label="Dominant")
        ax.axhline(
            np.quantile(y, 0.9), color="green", linestyle="--", label="Structuring"
        )
        ax.set_ylim(-0.02 * (max(y)), max(y) * 1.13)
        ax.set_title(t)
        ax.set_xlabel("Dominance candidate index")
        ax.set_ylabel("Number of degree")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig(
        f"save_result/graphs/abundance_degree_change.svg",
        format="SVG",
        dpi=300,
    )
