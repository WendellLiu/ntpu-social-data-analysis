import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
from scipy import stats


def crosstab_with_residuals(row_series, col_series):
    # row_series:
    observed = pd.crosstab(row_series, col_series)

    # 百分比（佔總數）
    ct_pct = pd.crosstab(row_series, col_series, normalize="all") * 100

    # 列百分比
    ct_row_pct = pd.crosstab(row_series, col_series, normalize="index") * 100

    # 欄百分比
    ct_col_pct = pd.crosstab(row_series, col_series, normalize="columns") * 100

    chi2, p_value, dof, expected = chi2_contingency(observed)
    expected = pd.DataFrame(expected, index=observed.index, columns=observed.columns)

    # 1. Residuals
    residuals = observed - expected

    # 2. Standardized Residuals
    std_residuals = residuals / np.sqrt(expected)

    # 3. Adjusted Standardized Residuals
    n = observed.sum().sum()
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)

    adjusted_residuals = pd.DataFrame(index=observed.index, columns=observed.columns)

    for i in observed.index:
        for j in observed.columns:
            adj_res = (observed.loc[i, j] - expected.loc[i, j]) / np.sqrt(
                expected.loc[i, j] * (1 - row_totals[i] / n) * (1 - col_totals[j] / n)
            )
            adjusted_residuals.loc[i, j] = adj_res

    chi2_table = pd.DataFrame(
        {
            "Value": [chi2, p_value, dof],
        },
        index=["chi2", "p_value", "dof"],
    )

    return {
        "ct": observed,
        "ct_pct": ct_pct,
        "ct_row_pct": ct_row_pct,
        "ct_col_pct": ct_col_pct,
        "expected": expected,
        "residuals": residuals,
        "std_residuals": std_residuals,
        "adjusted_residuals": adjusted_residuals,
        "chi2_table": chi2_table,
    }


def cramers_v(row_series, col_series):
    """
    Calculate Cramer's V for two categorical variables.

    Args:
        row_series (pd.Series): A pandas Series representing the rows.
        col_series (pd.Series): A pandas Series representing the columns.

    Returns:
        dict: A dictionary containing the Cramer's V value.
    """
    observed = pd.crosstab(row_series, col_series)

    v = association(observed, method="cramer")

    return {"cramers_v": v}


def goodman_kruskal_measures(x, y):
    """
    Calculate Goodman and Kruskal's lambda and tau measures with significance tests.

    Parameters:
    -----------
    x : pd.Series
        Independent variable (predictor)
    y : pd.Series
        Dependent variable (response)

    Returns:
    --------
    dict : Dictionary containing lambda and tau measures with their significance
    """

    # Remove missing values
    mask = ~(x.isna() | y.isna())
    x = x[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    n = len(x)

    # Create contingency table
    ct = pd.crosstab(x, y)
    ct_array = ct.values

    # --- Lambda Calculation ---
    # Lambda (y dependent on x)
    max_in_rows = ct.max(axis=1).sum()
    max_y_overall = ct.sum().max()
    lambda_yx = (
        (max_in_rows - max_y_overall) / (n - max_y_overall)
        if (n - max_y_overall) != 0
        else 0
    )

    # Lambda (x dependent on y)
    max_in_cols = ct.max(axis=0).sum()
    max_x_overall = ct.sum(axis=0).idxmax()
    max_x_overall = ct.sum(axis=0).max()
    lambda_xy = (
        (max_in_cols - max_x_overall) / (n - max_x_overall)
        if (n - max_x_overall) != 0
        else 0
    )

    # Symmetric lambda
    lambda_sym = (
        ((max_in_rows + max_in_cols) - (max_y_overall + max_x_overall))
        / ((2 * n) - (max_y_overall + max_x_overall))
        if ((2 * n) - (max_y_overall + max_x_overall)) != 0
        else 0
    )

    # Standard errors for lambda (Goodman & Kruskal asymptotic formulas)
    # For lambda Y|X
    if lambda_yx > 0 and (n - max_y_overall) > 0:
        # Simplified asymptotic SE
        lambda_yx_se = np.sqrt(
            (n - max_in_rows) * (max_in_rows + max_y_overall - 2 * ct.max(axis=1).max())
        ) / (n - max_y_overall)
        lambda_yx_se = lambda_yx_se / np.sqrt(n) if lambda_yx_se > 0 else 0
    else:
        lambda_yx_se = 0

    lambda_yx_z = lambda_yx / lambda_yx_se if lambda_yx_se > 0 else 0
    lambda_yx_p = (
        2 * (1 - stats.norm.cdf(abs(lambda_yx_z))) if lambda_yx_se > 0 else 1.0
    )

    # For lambda X|Y
    if lambda_xy > 0 and (n - max_x_overall) > 0:
        lambda_xy_se = np.sqrt(
            (n - max_in_cols) * (max_in_cols + max_x_overall - 2 * ct.max(axis=0).max())
        ) / (n - max_x_overall)
        lambda_xy_se = lambda_xy_se / np.sqrt(n) if lambda_xy_se > 0 else 0
    else:
        lambda_xy_se = 0

    lambda_xy_z = lambda_xy / lambda_xy_se if lambda_xy_se > 0 else 0
    lambda_xy_p = (
        2 * (1 - stats.norm.cdf(abs(lambda_xy_z))) if lambda_xy_se > 0 else 1.0
    )

    # For symmetric lambda
    if lambda_sym > 0:
        lambda_sym_se = np.sqrt(lambda_sym * (1 - lambda_sym) / n)
    else:
        lambda_sym_se = 0

    lambda_sym_z = lambda_sym / lambda_sym_se if lambda_sym_se > 0 else 0
    lambda_sym_p = (
        2 * (1 - stats.norm.cdf(abs(lambda_sym_z))) if lambda_sym_se > 0 else 1.0
    )

    # --- Tau Calculation ---
    row_totals = ct.sum(axis=1).values
    col_totals = ct.sum(axis=0).values

    # Tau (y dependent on x)
    e1_y = n - col_totals.max()
    e2_y = n - ct.max(axis=1).sum()
    tau_yx = (e1_y - e2_y) / e1_y if e1_y != 0 else 0

    # Tau (x dependent on y)
    e1_x = n - row_totals.max()
    e2_x = n - ct.max(axis=0).sum()
    tau_xy = (e1_x - e2_x) / e1_x if e1_x != 0 else 0

    # Standard errors for tau (using asymptotic formulas)
    # For tau Y|X
    if tau_yx > 0:
        # Asymptotic SE based on Goodman & Kruskal (1972)
        sum_term = 0
        for i in range(len(row_totals)):
            for j in range(len(col_totals)):
                p_ij = ct_array[i, j] / n
                p_i = row_totals[i] / n
                p_j = col_totals[j] / n
                p_j_max = col_totals.max() / n

                if p_i > 0 and p_j > 0:
                    term = (
                        p_ij
                        * (
                            (1 if j == np.argmax(col_totals) else 0)
                            - (1 if ct_array[i, j] == ct_array[i].max() else 0)
                        )
                        ** 2
                    )
                    sum_term += term

        tau_yx_se = (
            np.sqrt(sum_term / (e1_y / n) ** 2) / np.sqrt(n) if sum_term > 0 else 0
        )
    else:
        tau_yx_se = 0

    tau_yx_z = tau_yx / tau_yx_se if tau_yx_se > 0 else 0
    tau_yx_p = 2 * (1 - stats.norm.cdf(abs(tau_yx_z))) if tau_yx_se > 0 else 1.0

    # For tau X|Y
    if tau_xy > 0:
        sum_term = 0
        for i in range(len(row_totals)):
            for j in range(len(col_totals)):
                p_ij = ct_array[i, j] / n
                p_i = row_totals[i] / n
                p_j = col_totals[j] / n

                if p_i > 0 and p_j > 0:
                    term = (
                        p_ij
                        * (
                            (1 if i == np.argmax(row_totals) else 0)
                            - (1 if ct_array[i, j] == ct_array[:, j].max() else 0)
                        )
                        ** 2
                    )
                    sum_term += term

        tau_xy_se = (
            np.sqrt(sum_term / (e1_x / n) ** 2) / np.sqrt(n) if sum_term > 0 else 0
        )
    else:
        tau_xy_se = 0

    tau_xy_z = tau_xy / tau_xy_se if tau_xy_se > 0 else 0
    tau_xy_p = 2 * (1 - stats.norm.cdf(abs(tau_xy_z))) if tau_xy_se > 0 else 1.0

    # Results dictionary
    results = {
        "Lambda": {
            "Symmetric": {
                "Value": lambda_sym,
                "Approx_SE": lambda_sym_se,
                "Approx_Sig": lambda_sym_p,
            },
            "Y_dependent_on_X": {
                "Value": lambda_yx,
                "Approx_SE": lambda_yx_se,
                "Approx_Sig": lambda_yx_p,
            },
            "X_dependent_on_Y": {
                "Value": lambda_xy,
                "Approx_SE": lambda_xy_se,
                "Approx_Sig": lambda_xy_p,
            },
        },
        "Tau": {
            "Y_dependent_on_X": {
                "Value": tau_yx,
                "Approx_SE": tau_yx_se,
                "Approx_Sig": tau_yx_p,
            },
            "X_dependent_on_Y": {
                "Value": tau_xy,
                "Approx_SE": tau_xy_se,
                "Approx_Sig": tau_xy_p,
            },
        },
        "N": n,
    }

    return results
