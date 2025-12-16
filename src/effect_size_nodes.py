import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def extract_xy(result):
    """
    Extracts two numerical features from the input data for analysis:

    - x: Represents the number of communities (or experimental repetitions),
      derived from the keys of the input dictionary.
    - y: Corresponds to the number of unique nodes involved in each repetition,
      calculated by aggregating nodes found in the 'under' and 'over' relationship pairs.
    """
    x = np.array([int(k) for k, v in result.items() for _ in v])
    y = []
    for res_list in result.values():
        for res_dict in res_list:
            nodes = set()
            for pair in list(res_dict["under"].keys()) + list(res_dict["over"].keys()):
                nodes.update(pair.split("|"))
            y.append(len(nodes))
    return x, np.array(y)


def model(x, A, a):
    """
    Defines a saturating exponential model.
    """
    return A * (1 - np.exp(-a * x))


def prediction_std(x, popt, pcov):
    """
    Estimates the standard error of the model predictions by applying the law of error 
    propagation to the saturating exponential model.
    """
    A, a = popt
    J = np.vstack([1 - np.exp(-a * x), A * x * np.exp(-a * x)]).T
    var_pred = np.sum(J @ pcov * J, axis=1)
    return np.sqrt(var_pred)


def fit_saturating_model(x, y):
    """
    Fits a saturating exponential model to the given data and evaluates model performance.

    Returns
    -------
    popt : np.ndarray
        Optimized model parameters [A, a].
    y_pred : np.ndarray
        Model predictions corresponding to input x.
    y_std : np.ndarray
        Standard deviations of the predicted values, representing uncertainty from parameter estimation.
    r2 : float
        Coefficient of determination, indicating the proportion of variance explained by the model.
    p_val : float
        p-value from the F-test assessing the overall significance of the fitted model.
    """
    popt, pcov = curve_fit(
        model, x, y, p0=[max(y), 0.01], bounds=([0, 0], [np.inf, 1]), maxfev=10000
    )
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


def impact_plots_nodes(treatment, result):
    """
    Visualizes the relationship between the number of communities and the number of nodes,
    fitting a saturating exponential model to quantify node accumulation.
    """
    x, y = extract_xy(result)
    popt, y_pred, y_std, r2, p_val = fit_saturating_model(x, y)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.7, color="blue", label="Observed nodes")
    plt.plot(x, y_pred, color="red", label="Model fit")
    plt.fill_between(
        x, y_pred - y_std, y_pred + y_std, color="grey", alpha=0.2, label="±1σ"
    )
    plt.xlabel("Number of used communities")
    plt.ylabel("Network order")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"save_result/effect_size/effect_size_nodes_{treatment}.svg",
        format="svg",
        dpi=300,
    )
    plt.close()


def difference_impact_plots_nodes(result_alpine, result_warmed):
    """
    Compares node accumulation between two treatments (Alpine and Warmed) by fitting
    saturating exponential models to each dataset and testing whether the fits differ significantly.
    """
    x_a, y_a = extract_xy(result_alpine)
    popt_a, y_pred_a, y_std_a, r2_a, p_val_a = fit_saturating_model(x_a, y_a)

    x_w, y_w = extract_xy(result_warmed)
    x_w, y_w = x_w[x_w <= 30], y_w[x_w <= 30]
    popt_w, y_pred_w, y_std_w, r2_w, p_val_w = fit_saturating_model(x_w, y_w)

    # Compare fits
    y_all = np.concatenate([y_a, y_w])
    x_all = np.concatenate([x_a, x_w])
    popt_all, _ = curve_fit(
        model,
        x_all,
        y_all,
        p0=[max(y_all), 0.01],
        bounds=([0, 0], [np.inf, 1]),
        maxfev=10000,
    )
    ssr_all = np.sum((y_all - model(x_all, *popt_all)) ** 2)
    ssr_sep = np.sum((y_a - model(x_a, *popt_a)) ** 2) + np.sum(
        (y_w - model(x_w, *popt_w)) ** 2
    )
    F = ((ssr_all - ssr_sep) / 2) / (ssr_sep / (len(y_all) - 4))
    p_diff = 1 - stats.f.cdf(F, 2, len(y_all) - 4)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x_a, y_a, alpha=0.2, color="blue")
    plt.scatter(x_w, y_w, alpha=0.2, color="red")
    plt.plot(
        x_a,
        y_pred_a,
        color="blue",
        label=f"Alpine fit\ny = {popt_a[0]:.0f}·(1-exp(-{popt_a[1]:.2f}·x)), R²={r2_a:.2f}, p={p_val_a:.1e}",
    )
    plt.plot(
        x_w,
        y_pred_w,
        color="red",
        label=f"Warmed fit\ny = {popt_w[0]:.0f}·(1-exp(-{popt_w[1]:.2f}·x)), R²={r2_w:.2f}, p={p_val_w:.1e}",
    )
    plt.fill_between(
        x_a, y_pred_a - y_std_a, y_pred_a + y_std_a, color="grey", alpha=0.2
    )
    plt.fill_between(
        x_w, y_pred_w - y_std_w, y_pred_w + y_std_w, color="grey", alpha=0.2
    )

    plt.title(
        f"Node accumulation – Alpine vs Warmed\n"
        f"Alpine: y = {popt_a[0]:.0f}·(1-exp(-{popt_a[1]:.2f}·x)), R²={r2_a:.2f}, p={p_val_a:.1e}\n"
        f"Warmed: y = {popt_w[0]:.0f}·(1-exp(-{popt_w[1]:.2f}·x)), R²={r2_w:.2f}, p={p_val_w:.1e}\n"
        f"Model difference p = {p_diff:.1e}"
    )
    plt.xlabel("Number of used communities")
    plt.ylabel("Network order")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "save_result/effect_size/effect_size_nodes_difference.svg",
        format="svg",
        dpi=300,
    )
    plt.close()
