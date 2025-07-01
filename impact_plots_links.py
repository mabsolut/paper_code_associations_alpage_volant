import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def extract(result):
    """
    Extracts key variables from the input data for analysis of network links.

    Parameters
    ----------
    result : dict
        Nested dictionary where each key corresponds to a number of plots and maps
        to a list of repetitions containing 'under' and 'over' link data.

    Returns
    -------
    x : np.ndarray
        Array of plot counts corresponding to each repetition.
    y : np.ndarray
        Array of total link counts (under + over) for each repetition.
    """
    x = np.array([int(k) for k, v in result.items() for _ in v])
    y = np.array(
        [len(r["under"]) + len(r["over"]) for reps in result.values() for r in reps]
    )
    return x, y


def regress(x, y):
    """
    Performs linear regression through the origin (no intercept) to model the relationship between x and y.

    Parameters
    ----------
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.

    Returns
    -------
    slope : float
        Estimated slope of the regression line.
    stderr : float
        Standard error of the slope estimate.
    r2 : float
        Coefficient of determination, indicating the proportion of variance in y explained by the model.
    """
    slope = np.dot(x, y) / np.dot(x, x)
    y_pred = slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - ss_res / np.sum((y - np.mean(y)) ** 2)
    stderr = np.sqrt(ss_res / (len(x) - 1)) / np.sqrt(np.dot(x, x))
    return slope, stderr, r2


def impact_plots_links(treatment, result):
    """
    Visualizes the accumulation of detected species associations (links) with increasing numbers of sampled plots
    for a single treatment. Fits a linear regression model without intercept to quantify the relationship,
    and plots the observed data, the fitted line, and the 95% confidence interval of the slope estimate.

    Parameters
    ----------
    treatment : str
        Identifier for the treatment or experimental condition, used in the plot title and filename.
    result : dict
        Nested dictionary containing link data structured for extraction by the `extract` function.
    """
    x, y = extract(result)
    slope, stderr, r2 = regress(x, y)
    t_stat = slope / stderr
    p_val = 2 * stats.t.sf(np.abs(t_stat), df=len(x) - 1)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.7, color="blue", label="Observed data")
    x_vals = np.linspace(0, max(x), 100)
    y_vals = slope * x_vals
    y_upper = (slope + 1.96 * stderr) * x_vals
    y_lower = (slope - 1.96 * stderr) * x_vals
    plt.plot(x_vals, y_vals, color="red", label="Linear fit")
    plt.fill_between(x_vals, y_lower, y_upper, color="grey", alpha=0.3, label="95% CI")

    plt.title(
        f"{treatment} - Link accumulation\nSlope = {slope:.2f}±{stderr:.2f} R² = {r2:.2f}, p = {p_val:.1e}"
    )
    plt.xlabel("Number of used plots")
    plt.ylabel("Number of network links")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"/home/alpage_volant/alpage_volant_interaction/save_result/impact_plots_links_{treatment}.svg",
        format="svg",
        dpi=300,
    )
    plt.close()


def difference_impact_plots_links(result_alpine, result_warmed):
    """
    Compares the slopes of link accumulation between Alpine and Warmed treatments
    by performing linear regressions without intercept and testing for differences
    in slope estimates.

    Parameters
    ----------
    result_alpine : dict
        Nested dictionary containing link data for the Alpine treatment.
    result_warmed : dict
        Nested dictionary containing link data for the Warmed treatment.
    """
    x_a, y_a = extract(result_alpine)
    slope_a, stderr_a, r2_a = regress(x_a, y_a)
    t_stat_a = slope_a / stderr_a
    p_val_a = 2 * stats.t.sf(np.abs(t_stat_a), df=len(x_a) - 1)

    x_w, y_w = extract(result_warmed)
    slope_w, stderr_w, r2_w = regress(x_w, y_w)
    t_stat_w = slope_w / stderr_w
    p_val_w = 2 * stats.t.sf(np.abs(t_stat_w), df=len(x_w) - 1)

    slope_diff = slope_w - slope_a
    stderr_diff = np.sqrt(stderr_a**2 + stderr_w**2)
    t_diff = slope_diff / stderr_diff
    p_diff = 2 * stats.t.sf(np.abs(t_diff), df=min(len(x_a), len(x_w)) - 1)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x_a, y_a, color="blue", alpha=0.3, label="Alpine")
    plt.scatter(x_w, y_w, color="red", alpha=0.3, label="Warmed")

    for x, slope, stderr, color, label in [
        (x_a, slope_a, stderr_a, "blue", "Alpine"),
        (x_w, slope_w, stderr_w, "red", "Warmed"),
    ]:
        x_vals = np.linspace(1, max(x), 100)
        y_fit = slope * x_vals
        y_upper = (slope + 1.96 * stderr) * x_vals
        y_lower = (slope - 1.96 * stderr) * x_vals
        plt.plot(x_vals, y_fit, color=color, label=f"{label} fit")
        plt.fill_between(x_vals, y_lower, y_upper, color=color, alpha=0.2)

    plt.title(
        f"Comparison of link accumulation slopes\n"
        f"Alpine: {slope_a:.2f}±{stderr_a:.2f}, R²={r2_a:.2f}, p = {p_val_a:.1e}\n"
        f"Warmed: {slope_w:.2f}±{stderr_w:.2f}, R²={r2_w:.2f}, p = {p_val_w:.1e}\n"
        f"Δslope = {slope_diff:.2f}±{stderr_diff:.2f}, p = {p_diff:.1e}"
    )
    plt.xlabel("Number of used plots")
    plt.ylabel("Number network links")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/home/alpage_volant/alpage_volant_interaction/save_result/number_links_difference.svg",
        format="svg",
        dpi=300,
    )
    plt.close()


def difference_impact_plots_links_sp(result_alpine, result_warmed, list_sp):
    """
    Compares the accumulation of network links involving individual species across
    increasing numbers of sampled plots, between Alpine and Warmed treatments.

    Parameters
    ----------
    result_alpine : dict
        Nested dictionary with link data for the Alpine treatment.
    result_warmed : dict
        Nested dictionary with link data for the Warmed treatment.
    list_sp : list of str
        List of species identifiers to analyze and compare individually.
    """
    for sp in list_sp:
        x_a, y_a = [], []
        for n in result_alpine:
            for res_dict in result_alpine[str(n)]:
                if len(res_dict["under"]) + len(res_dict["over"]) > 0:
                    x_a.append(int(n))
                    count = sum(
                        sp in pair.split("|") for pair in res_dict["under"]
                    ) + sum(sp in pair.split("|") for pair in res_dict["over"])
                    y_a.append(count)
        x_a, y_a = np.array(x_a), np.array(y_a)
        slope_a, stderr_a, r2_a = regress(x_a, y_a)
        p_val_a = 2 * stats.t.sf(np.abs(slope_a / stderr_a), df=len(x_a) - 1)

        x_w, y_w = [], []
        for n in result_warmed:
            for res_dict in result_warmed[str(n)]:
                if res_dict["under"] or res_dict["over"]:
                    x_w.append(int(n))
                    count = sum(
                        sp in pair.split("|") for pair in res_dict["under"]
                    ) + sum(sp in pair.split("|") for pair in res_dict["over"])
                    y_w.append(count)
        x_w, y_w = np.array(x_w), np.array(y_w)
        slope_w, stderr_w, r2_w = regress(x_w, y_w)
        p_val_w = 2 * stats.t.sf(np.abs(slope_w / stderr_w), df=len(x_w) - 1)

        slope_diff = slope_w - slope_a
        stderr_diff = np.sqrt(stderr_a**2 + stderr_w**2)
        t_diff = slope_diff / stderr_diff
        p_diff = 2 * stats.t.sf(np.abs(t_diff), df=min(len(x_a), len(x_w)) - 1)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.scatter(x_a, y_a, color="blue", alpha=0.3, label="Alpine")
        plt.scatter(x_w, y_w, color="red", alpha=0.3, label="Warmed")

        for x, slope, stderr, color, label in [
            (x_a, slope_a, stderr_a, "blue", "Alpine"),
            (x_w, slope_w, stderr_w, "red", "Warmed"),
        ]:
            x_vals = np.linspace(1, max(x), 100)
            y_fit = slope * x_vals
            y_upper = (slope + 1.96 * stderr) * x_vals
            y_lower = (slope - 1.96 * stderr) * x_vals
            plt.plot(x_vals, y_fit, color=color, label=f"{label} fit")
            plt.fill_between(x_vals, y_lower, y_upper, color=color, alpha=0.2)

        plt.title(
            f"{sp}\n"
            f"Alpine: {slope_a:.2f}±{stderr_a:.2f}, R²={r2_a:.2f}, p = {p_val_a:.1e}\n"
            f"Warmed: {slope_w:.2f}±{stderr_w:.2f}, R²={r2_w:.2f}, p = {p_val_w:.1e}\n"
            f"Δslope = {slope_diff:.2f}±{stderr_diff:.2f}, p = {p_diff:.1e}"
        )
        plt.xlabel("Number of used plots")
        plt.ylabel("Number network links")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            "/home/alpage_volant/alpage_volant_interaction/save_result/relative_number_links_difference_sp/number_link_difference_{sp}.svg",
            format="svg",
            dpi=300,
        )
        plt.close()
