import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def extract(result):
    """
    Extracts key variables from the input data for analysis of network links.
    """
    x = np.array([int(k) for k, v in result.items() for _ in v])
    y = np.array(
        [len(r["under"]) + len(r["over"]) for reps in result.values() for r in reps]
    )
    return x, y


def regress(x, y):
    """
    Performs linear regression through the origin (no intercept) to model the relationship between x and y.
    """
    slope = np.dot(x, y) / np.dot(x, x)
    y_pred = slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    den = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if den == 0 or np.isnan(den) else 1 - ss_res / den
    stderr = np.sqrt(ss_res / (len(x) - 1)) / np.sqrt(np.dot(x, x))
    return slope, stderr, r2


def model(x, L, k, x0):
    """
    Logistic model constrained to pass through the origin (0, 0).
    """
    return L * (1 / (1 + np.exp(-k * (x - x0))) - 1 / (1 + np.exp(k * x0)))


def prediction_std(x, popt, pcov):
    """
    Estimates the standard error of predictions for the logistic model using error propagation.
    """
    L, k, x0 = popt
    exp1 = np.exp(-k * (x - x0))
    exp2 = np.exp(k * x0)
    denom1 = 1 + exp1
    denom2 = 1 + exp2
    denom1_sq = denom1**2
    denom2_sq = denom2**2

    dy_dL = (1 / denom1) - (1 / denom2)
    dy_dk = L * ((x - x0) * exp1 / denom1_sq + x0 * exp2 / denom2_sq)
    dy_dx0 = L * (k * exp1 / denom1_sq - k * exp2 / denom2_sq)

    J = np.vstack([dy_dL, dy_dk, dy_dx0]).T
    var_pred = np.sum(J @ pcov * J, axis=1)
    return np.sqrt(var_pred)


def fit_model(x, y):
    """
    Fits a logistic growth model to the given data and evaluates model performance.
    """
    p0 = [max(y), 1, np.median(x)]
    bounds = ([0, 0, min(x)], [np.inf, 10, max(x)])

    popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=10000)
    y_pred = model(x, *popt)
    y_std = prediction_std(x, popt, pcov)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    dof_model = len(popt)
    dof_error = len(y) - dof_model
    F_stat = ((ss_tot - ss_res) / dof_model) / (ss_res / dof_error)
    p_val = 1 - stats.f.cdf(F_stat, dof_model, dof_error)

    return popt, y_pred, y_std, r2, p_val


def impact_plots_links(treatment, result):
    """
    Visualizes the accumulation of detected species associations (links) with increasing numbers of sampled communities for a single treatment.
    """
    x, y = extract(result)
    popt, y_pred, y_std, r2_S, p_val_S = fit_model(x, y)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.7, color="blue", label="Observed links")
    plt.plot(x, y_pred, color="red", label="Model fit")
    plt.fill_between(
        x, y_pred - y_std, y_pred + y_std, color="grey", alpha=0.2, label="±1σ"
    )

    plt.title(
        f"{treatment} – Links accumulation\nLogistic model: y = {popt[0]:.0f}·(1/(1 + exp(-{popt[1]:.2f}·(x - {popt[1]:.2f}))) - 1/(1 + exp({popt[1]:.2f}·{popt[1]:.2f}))),\n R² = {r2_S:.2f}, p = {p_val_S:.1e}"
    )
    plt.xlabel("Number of used communities")
    plt.ylabel("Network size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"save_result/effect_size/effect_size_links_{treatment}.svg",
        format="svg",
        dpi=300,
    )
    plt.close()


def difference_impact_plots_links(result_alpine, result_warmed):
    """
    Compares the slopes of link accumulation between Alpine and Warmed treatments by performing linear regressions without intercept and testing for differences in slope estimates.
    """
    x_a, y_a = extract(result_alpine)
    slope_a, stderr_a, r2_a = regress(x_a, y_a)
    t_stat_a = slope_a / stderr_a
    p_val_a = 2 * stats.t.sf(np.abs(t_stat_a), df=len(x_a) - 1)

    residuals_a = y_a - slope_a * x_a
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(residuals_a, bins=30, color="skyblue", edgecolor="black", density=True)
    xmin, xmax = plt.xlim()
    x_vals = np.linspace(xmin, xmax, 100)
    plt.plot(
        x_vals, stats.norm.pdf(x_vals, np.mean(residuals_a), np.std(residuals_a)), "r--"
    )
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Density")

    # QQ plot
    plt.subplot(1, 2, 2)
    stats.probplot(residuals_a, dist="norm", plot=plt)
    plt.title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig(
        "save_result/effect_size/effect_size_links_difference_residuals_alpine.svg",
        format="svg",
        dpi=300,
    )
    plt.close()

    x_w, y_w = extract(result_warmed)
    x_w, y_w = x_w[x_w <= 30], y_w[x_w <= 30]
    slope_w, stderr_w, r2_w = regress(x_w, y_w)
    t_stat_w = slope_w / stderr_w
    p_val_w = 2 * stats.t.sf(np.abs(t_stat_w), df=len(x_w) - 1)

    residuals_w = y_w - slope_w * x_w
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(residuals_w, bins=30, color="skyblue", edgecolor="black", density=True)
    xmin, xmax = plt.xlim()
    x_vals = np.linspace(xmin, xmax, 100)
    plt.plot(
        x_vals, stats.norm.pdf(x_vals, np.mean(residuals_w), np.std(residuals_w)), "r--"
    )
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Density")

    # QQ plot
    plt.subplot(1, 2, 2)
    stats.probplot(residuals_w, dist="norm", plot=plt)
    plt.title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig(
        "save_result/effect_size/effect_size_links_difference_residuals_warmed.svg",
        format="svg",
        dpi=300,
    )
    plt.close()

    slope_diff = slope_w - slope_a
    stderr_diff = np.sqrt(stderr_a**2 + stderr_w**2)
    t_diff = slope_diff / stderr_diff
    p_diff = 2 * stats.t.sf(np.abs(t_diff), df=min(len(x_a), len(x_w)) - 1)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x_a, y_a, color="blue", alpha=0.3)
    plt.scatter(x_w, y_w, color="red", alpha=0.3)

    for x, slope, stderr, r2, pval, color, label in [
        (x_a, slope_a, stderr_a, r2_a, p_val_a, "blue", "Alpine"),
        (x_w, slope_w, stderr_w, r2_w, p_val_w, "red", "Warmed"),
    ]:
        x_vals = np.linspace(1, max(x), 100)
        y_fit = slope * x_vals
        y_upper = (slope + 1.96 * stderr) * x_vals
        y_lower = (slope - 1.96 * stderr) * x_vals
        plt.plot(
            x_vals,
            y_fit,
            color=color,
            label=f"{label} fit\nSlope = {slope:.2f}±{stderr:.2f}, R²={r2:.2f}, p={pval:.1e}",
        )
        plt.fill_between(x_vals, y_lower, y_upper, color=color, alpha=0.2)

    plt.title(
        f"Comparison of link accumulation slopes\n"
        f"Alpine: {slope_a:.2f}±{stderr_a:.2f}, R²={r2_a:.2f}, p = {p_val_a:.1e}\n"
        f"Warmed: {slope_w:.2f}±{stderr_w:.2f}, R²={r2_w:.2f}, p = {p_val_w:.1e}\n"
        f"Δslope = {slope_diff:.2f}±{stderr_diff:.2f}, p = {p_diff:.1e}"
    )
    plt.xlabel("Number of used communities")
    plt.ylabel("Network size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "save_result/effect_size/effect_size_links_difference.svg",
        format="svg",
        dpi=300,
    )
    plt.close()


def difference_impact_plots_positive_links(result_alpine, result_warmed):
    """
    Compares the slopes of positive link accumulation between Alpine and Warmed treatments by performing linear regressions without intercept and testing for differences in slope estimates.
    """
    result_alpine = {
        n: [{**{k: v for k, v in r.items() if k != "under"}, "under": {}} for r in reps]
        for n, reps in result_alpine.items()
    }
    x_a, y_a = extract(result_alpine)
    slope_a, stderr_a, r2_a = regress(x_a, y_a)
    t_stat_a = np.nan if stderr_a == 0 or not np.isfinite(stderr_a) else slope_a / stderr_a
    p_val_a = 2 * stats.t.sf(np.abs(t_stat_a), df=len(x_a) - 1)

    result_warmed = {
        n: [{**{k: v for k, v in r.items() if k != "under"}, "under": {}} for r in reps]
        for n, reps in result_warmed.items()
    }
    x_w, y_w = extract(result_warmed)
    x_w, y_w = x_w[x_w <= 30], y_w[x_w <= 30]
    slope_w, stderr_w, r2_w = regress(x_w, y_w)
    t_stat_w = slope_w / stderr_w
    p_val_w = 2 * stats.t.sf(np.abs(t_stat_w), df=len(x_w) - 1)

    slope_diff = slope_w - slope_a
    stderr_diff = np.sqrt(stderr_a**2 + stderr_w**2)
    t_diff = slope_diff / stderr_diff
    p_diff = 2 * stats.t.sf(np.abs(t_diff), df=min(len(x_a), len(x_w)) - 1)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x_a, y_a, color="blue", alpha=0.3)
    plt.scatter(x_w, y_w, color="red", alpha=0.3)

    for x, slope, stderr, r2, pval, color, label in [
        (x_a, slope_a, stderr_a, r2_a, p_val_a, "blue", "Alpine"),
        (x_w, slope_w, stderr_w, r2_w, p_val_w, "red", "Warmed"),
    ]:
        x_vals = np.linspace(1, max(x), 100)
        y_fit = slope * x_vals
        y_upper = (slope + 1.96 * stderr) * x_vals
        y_lower = (slope - 1.96 * stderr) * x_vals
        plt.plot(
            x_vals,
            y_fit,
            color=color,
            label=f"{label} fit\nSlope = {slope:.2f}±{stderr:.2f}, R²={r2:.2f}, p={pval:.1e}",
        )
        plt.fill_between(x_vals, y_lower, y_upper, color=color, alpha=0.2)

    plt.title(
        f"Comparison of positive link accumulation slopes\n"
        f"Δslope = {slope_diff:.2f}±{stderr_diff:.2f}, p = {p_diff:.1e}"
    )
    plt.xlabel("Number of used communities")
    plt.ylabel("Number of positive associations in the network")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "save_result/effect_size/effect_size_positive_links_difference.svg",
        format="svg",
        dpi=300,
    )
    plt.close()
