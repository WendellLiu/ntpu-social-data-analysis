import numpy as np
import pandas as pd
from scipy import stats


def goodman_kruskal_lambda(data, col1, col2):
    """
    Calculate Goodman-Kruskal Lambda with ASE and Significance (SPSS-style)

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the categorical variables
    col1 : str
        Name of first categorical variable (row variable)
    col2 : str
        Name of second categorical variable (column variable)

    Returns:
    --------
    pd.DataFrame : DataFrame with columns ['Measure', 'Type', 'Value', 'ASE', 'Approx_T', 'Approx_Sig']

    Examples:
    ---------
    >>> df = pd.DataFrame({'Education': [...], 'Income': [...]})
    >>> lambda_df = goodman_kruskal_lambda(df, 'Education', 'Income')
    """

    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if col1 not in data.columns:
        raise ValueError(f"Column '{col1}' not found in DataFrame")

    if col2 not in data.columns:
        raise ValueError(f"Column '{col2}' not found in DataFrame")

    # Extract series
    series1 = data[col1]
    series2 = data[col2]

    # Remove missing values
    mask = series1.notna() & series2.notna()
    s1 = series1[mask]
    s2 = series2[mask]

    if len(s1) == 0:
        raise ValueError("No valid observations after removing missing values")

    # Create contingency table
    crosstab = pd.crosstab(s1, s2)
    n = crosstab.sum().sum()

    results = []

    # Lambda: Symmetric
    lambda_val, ase = _lambda_symmetric(crosstab, n)
    z_score = lambda_val / ase if ase > 0 else (0 if lambda_val == 0 else np.nan)
    p_value = (
        2 * (1 - stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
    )

    results.append(
        {
            "Measure": "Lambda",
            "Type": "Symmetric",
            "Value": lambda_val,
            "ASE": ase,
            "Approx_T": z_score,
            "Approx_Sig": p_value,
        }
    )

    # Lambda: col1 Dependent
    lambda_val, ase = _lambda_series1_dependent(crosstab, n)
    z_score = lambda_val / ase if ase > 0 else (0 if lambda_val == 0 else np.nan)
    p_value = (
        2 * (1 - stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
    )

    results.append(
        {
            "Measure": "Lambda",
            "Type": f"{col1} Dependent",
            "Value": lambda_val,
            "ASE": ase,
            "Approx_T": z_score,
            "Approx_Sig": p_value,
        }
    )

    # Lambda: col2 Dependent
    lambda_val, ase = _lambda_series2_dependent(crosstab, n)
    z_score = lambda_val / ase if ase > 0 else (0 if lambda_val == 0 else np.nan)
    p_value = (
        2 * (1 - stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
    )

    results.append(
        {
            "Measure": "Lambda",
            "Type": f"{col2} Dependent",
            "Value": lambda_val,
            "ASE": ase,
            "Approx_T": z_score,
            "Approx_Sig": p_value,
        }
    )

    return pd.DataFrame(results)


def goodman_kruskal_tau(data, col1, col2):
    """
    Calculate Goodman-Kruskal Tau with ASE and Significance (SPSS-style)

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the categorical variables
    col1 : str
        Name of first categorical variable (row variable, predictor)
    col2 : str
        Name of second categorical variable (column variable, dependent)

    Returns:
    --------
    pd.DataFrame : DataFrame with columns ['Measure', 'Type', 'Value', 'ASE', 'Approx_T', 'Approx_Sig']

    Examples:
    ---------
    >>> df = pd.DataFrame({'Education': [...], 'Income': [...]})
    >>> tau_df = goodman_kruskal_tau(df, 'Education', 'Income')
    """

    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if col1 not in data.columns:
        raise ValueError(f"Column '{col1}' not found in DataFrame")

    if col2 not in data.columns:
        raise ValueError(f"Column '{col2}' not found in DataFrame")

    # Extract series
    series1 = data[col1]
    series2 = data[col2]

    # Remove missing values
    mask = series1.notna() & series2.notna()
    s1 = series1[mask]
    s2 = series2[mask]

    if len(s1) == 0:
        raise ValueError("No valid observations after removing missing values")

    # Create contingency table
    crosstab = pd.crosstab(s1, s2)
    n = crosstab.sum().sum()

    results = []

    # Tau: col1 Dependent (col2 predicts col1)
    tau_val, ase = _tau_series1_dependent(crosstab, n)
    z_score = tau_val / ase if ase > 0 else (0 if tau_val == 0 else np.nan)
    p_value = (
        2 * (1 - stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
    )

    results.append(
        {
            "Measure": "Goodman-Kruskal Tau",
            "Type": f"{col1} Dependent",
            "Value": tau_val,
            "ASE": ase,
            "Approx_T": z_score,
            "Approx_Sig": p_value,
        }
    )

    # Tau: col2 Dependent (col1 predicts col2)
    tau_val, ase = _tau_series2_dependent(crosstab, n)
    z_score = tau_val / ase if ase > 0 else (0 if tau_val == 0 else np.nan)
    p_value = (
        2 * (1 - stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
    )

    results.append(
        {
            "Measure": "Goodman-Kruskal Tau",
            "Type": f"{col2} Dependent",
            "Value": tau_val,
            "ASE": ase,
            "Approx_T": z_score,
            "Approx_Sig": p_value,
        }
    )

    return pd.DataFrame(results)


# ============================================================================
# Helper Functions (Lambda)
# ============================================================================


def _lambda_series2_dependent(crosstab, n):
    """Lambda with series2 as dependent (series1 predicts series2)"""
    r, c = crosstab.shape

    col_totals = crosstab.sum(axis=0).values
    row_totals = crosstab.sum(axis=1).values

    f_plus_c = col_totals.max()
    f_r_max = np.array([crosstab.iloc[i, :].max() for i in range(r)])
    sum_f_r_max = f_r_max.sum()

    denominator = n - f_plus_c
    if denominator == 0:
        return 0.0, 0.0

    lambda_val = (sum_f_r_max - f_plus_c) / denominator

    # ASE using Delta Method
    variance_sum = 0

    for i in range(r):
        row_max = crosstab.iloc[i, :].max()
        max_indices = np.where(crosstab.iloc[i, :].values == row_max)[0]
        n_max = len(max_indices)

        for j in range(c):
            f_ij = crosstab.iloc[i, j]
            I_ij = 1 if j in max_indices else 0
            I_j_modal = 1 if col_totals[j] == f_plus_c else 0

            term = (I_ij / n_max - I_j_modal) / denominator
            p_ij = f_ij / n
            variance_sum += term**2 * p_ij * (1 - p_ij)

    ase = np.sqrt(variance_sum / n)

    return lambda_val, ase


def _lambda_series1_dependent(crosstab, n):
    """Lambda with series1 as dependent (series2 predicts series1)"""
    crosstab_T = crosstab.T
    return _lambda_series2_dependent(crosstab_T, n)


def _lambda_symmetric(crosstab, n):
    """Symmetric Lambda"""
    r, c = crosstab.shape

    col_totals = crosstab.sum(axis=0).values
    row_totals = crosstab.sum(axis=1).values

    f_plus_c = col_totals.max()
    f_r_plus = row_totals.max()

    f_r_max = np.array([crosstab.iloc[i, :].max() for i in range(r)])
    f_c_max = np.array([crosstab.iloc[:, j].max() for j in range(c)])

    sum_f_r_max = f_r_max.sum()
    sum_f_c_max = f_c_max.sum()

    numerator = sum_f_r_max + sum_f_c_max - f_plus_c - f_r_plus
    denominator = 2 * n - f_plus_c - f_r_plus

    if denominator == 0:
        return 0.0, 0.0

    lambda_val = numerator / denominator

    # ASE using Delta Method
    variance_sum = 0

    for i in range(r):
        row_max = crosstab.iloc[i, :].max()
        row_max_indices = np.where(crosstab.iloc[i, :].values == row_max)[0]
        n_row_max = len(row_max_indices)

        for j in range(c):
            col_max = crosstab.iloc[:, j].max()
            col_max_indices = np.where(crosstab.iloc[:, j].values == col_max)[0]
            n_col_max = len(col_max_indices)

            f_ij = crosstab.iloc[i, j]

            I_ij_row = 1 if j in row_max_indices else 0
            I_ij_col = 1 if i in col_max_indices else 0
            I_j_modal = 1 if col_totals[j] == f_plus_c else 0
            I_i_modal = 1 if row_totals[i] == f_r_plus else 0

            term = (
                I_ij_row / n_row_max + I_ij_col / n_col_max - I_j_modal - I_i_modal
            ) / denominator

            p_ij = f_ij / n
            variance_sum += term**2 * p_ij * (1 - p_ij)

    ase = np.sqrt(variance_sum / n)

    return lambda_val, ase


# ============================================================================
# Helper Functions (Tau)
# ============================================================================


def _tau_series2_dependent(crosstab, n):
    """Goodman-Kruskal Tau with series2 as dependent"""
    r, c = crosstab.shape

    P = crosstab.values / n
    p_i_plus = P.sum(axis=1)
    p_plus_j = P.sum(axis=0)

    var_y = 1 - (p_plus_j**2).sum()

    if var_y == 0:
        return 0.0, 0.0

    e_var_y_given_x = 0
    for i in range(r):
        if p_i_plus[i] > 0:
            p_j_given_i = P[i, :] / p_i_plus[i]
            var_y_given_i = 1 - (p_j_given_i**2).sum()
            e_var_y_given_x += p_i_plus[i] * var_y_given_i

    tau = (var_y - e_var_y_given_x) / var_y

    # ASE using Delta Method
    variance_sum = 0

    for i in range(r):
        for j in range(c):
            p_ij = P[i, j]

            if p_i_plus[i] > 0:
                d_var_y = -2 * p_plus_j[j]

                p_j_given_i = P[i, :] / p_i_plus[i]
                var_y_given_i = 1 - (p_j_given_i**2).sum()

                d_e_var = var_y_given_i - 2 * p_j_given_i[j] * (1 - p_j_given_i[j])

                d_tau = (
                    d_var_y * var_y
                    - (var_y - e_var_y_given_x) * d_var_y
                    - var_y * d_e_var
                ) / var_y**2

                variance_sum += d_tau**2 * p_ij * (1 - p_ij)

    ase = np.sqrt(variance_sum / n)

    return tau, ase


def _tau_series1_dependent(crosstab, n):
    """Goodman-Kruskal Tau with series1 as dependent"""
    crosstab_T = crosstab.T
    return _tau_series2_dependent(crosstab_T, n)


# ============================================================================
# Utility Functions
# ============================================================================


def directional_measures(data, col1, col2):
    """
    Calculate both Lambda and Tau in one call

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the categorical variables
    col1 : str
        Name of first categorical variable (row variable)
    col2 : str
        Name of second categorical variable (column variable)

    Returns:
    --------
    pd.DataFrame : Combined DataFrame with both Lambda and Tau results

    Examples:
    ---------
    >>> df = pd.DataFrame({'Education': [...], 'Income': [...]})
    >>> all_measures = directional_measures(df, 'Education', 'Income')
    """

    lambda_df = goodman_kruskal_lambda(data, col1, col2)
    tau_df = goodman_kruskal_tau(data, col1, col2)

    combined_df = pd.concat([lambda_df, tau_df], ignore_index=True)

    return combined_df
