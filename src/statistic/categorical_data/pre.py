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

    # Calculate E1 (errors without knowing independent variable)
    max_col_total = col_totals.max()
    E1 = n - max_col_total
    if E1 == 0:  # Avoid division by zero
        return 0.0, 0.0

    # Calculate E2 (errors with knowing independent variable)
    # Sum of maximum frequencies in each row
    sum_max_in_each_row = np.array([crosstab.iloc[i, :].max() for i in range(r)]).sum()
    E2 = n - sum_max_in_each_row

    # Calculate Lambda value using the (E1 - E2) / E1 formula
    lambda_val = (E1 - E2) / E1

    # ASE using Delta Method (corrected)
    # Based on delta method, Var(lambda) = (sum(d_ij^2 * p_ij) - (sum(d_ij * p_ij))^2) / n
    prob_E1 = E1 / n  # E1 in probability terms
    if prob_E1 == 0:
        return lambda_val, 0.0

    # Handle ties for the modal column
    modal_col_indices = np.where(col_totals == max_col_total)[0]
    n_modal_col = len(modal_col_indices)

    sum_d_sq_p = 0
    sum_d_p = 0

    for i in range(r):
        row_max = crosstab.iloc[i, :].max()
        row_max_indices = np.where(crosstab.iloc[i, :].values == row_max)[0]
        n_row_max = len(row_max_indices)

        for j in range(c):
            p_ij = crosstab.iloc[i, j] / n
            if p_ij == 0:  # Skip zero probabilities
                continue

            I_ij_row_max = 1 if j in row_max_indices else 0
            I_j_modal_col = 1 if j in modal_col_indices else 0

            # Derivative of N and D w.r.t p_ij, handling ties by averaging.
            dN_dp_ij = (I_ij_row_max / n_row_max) - (I_j_modal_col / n_modal_col)
            dD_dp_ij = -(I_j_modal_col / n_modal_col)

            # Derivative of lambda w.r.t p_ij
            # d_lambda/d_p_ij = (1/D) * (dN/dp_ij - lambda * dD/dp_ij)
            d_lambda_dp_ij = (dN_dp_ij - lambda_val * dD_dp_ij) / prob_E1

            sum_d_sq_p += (d_lambda_dp_ij**2) * p_ij
            sum_d_p += d_lambda_dp_ij * p_ij

    variance = (sum_d_sq_p - sum_d_p**2) / n
    ase = np.sqrt(max(0, variance))

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

    # ASE using Delta Method (corrected)
    # Based on delta method, Var(lambda) = (sum(d_ij^2 * p_ij) - (sum(d_ij * p_ij))^2) / n
    prob_denominator = denominator / n
    if prob_denominator == 0:
        return lambda_val, 0.0

    # Handle ties for modal row and column totals
    modal_col_indices = np.where(col_totals == f_plus_c)[0]
    n_modal_col = len(modal_col_indices)

    modal_row_indices = np.where(row_totals == f_r_plus)[0]
    n_modal_row = len(modal_row_indices)

    sum_d_sq_p = 0
    sum_d_p = 0

    for i in range(r):
        row_max = crosstab.iloc[i, :].max()
        row_max_indices = np.where(crosstab.iloc[i, :].values == row_max)[0]
        n_row_max = len(row_max_indices)

        for j in range(c):
            col_max = crosstab.iloc[:, j].max()
            col_max_indices = np.where(crosstab.iloc[:, j].values == col_max)[0]
            n_col_max = len(col_max_indices)

            p_ij = crosstab.iloc[i, j] / n
            if p_ij == 0:  # Skip zero probabilities
                continue

            I_ij_row_max = 1 if j in row_max_indices else 0
            I_ij_col_max = 1 if i in col_max_indices else 0
            I_j_modal_col = 1 if j in modal_col_indices else 0
            I_i_modal_row = 1 if i in modal_row_indices else 0

            # Derivative of Num and Den w.r.t p_ij, handling ties by averaging.
            dN_dp_ij = (I_ij_row_max / n_row_max) + (I_ij_col_max / n_col_max) - \
                       (I_j_modal_col / n_modal_col) - (I_i_modal_row / n_modal_row)

            dDen_dp_ij = -(I_j_modal_col / n_modal_col) - (I_i_modal_row / n_modal_row)

            # Derivative of symmetric lambda w.r.t p_ij
            # d_lambda_sym/d_p_ij = (1/Den) * (dNum/dp_ij - lambda_sym * dDen/dp_ij)
            d_lambda_sym_dp_ij = (dN_dp_ij - lambda_val * dDen_dp_ij) / prob_denominator

            sum_d_sq_p += (d_lambda_sym_dp_ij**2) * p_ij
            sum_d_p += d_lambda_sym_dp_ij * p_ij

    variance = (sum_d_sq_p - sum_d_p**2) / n
    ase = np.sqrt(max(0, variance))

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

    # ASE using Delta Method (corrected)
    # Based on delta method, Var(tau) = (sum(d_ij^2 * p_ij) - (sum(d_ij * p_ij))^2) / n
    sum_d_sq_p = 0
    sum_d_p = 0

    for i in range(r):
        p_i_plus_val = p_i_plus[i]
        if p_i_plus_val == 0:
            continue

        p_j_given_i = P[i, :] / p_i_plus_val
        var_y_given_i = 1 - np.sum(p_j_given_i**2)

        for j in range(c):
            p_ij = P[i, j]
            if p_ij == 0:
                continue

            # Derivative of E[Var(Y|X)] w.r.t p_ij
            d_e_var_dp_ij = (1 - var_y_given_i) - 2 * p_j_given_i[j]

            # Derivative of Var(Y) w.r.t p_ij
            d_var_y_dp_ij = -2 * p_plus_j[j]

            # Derivative of tau w.r.t p_ij using quotient rule for tau = 1 - N/D
            # d(tau)/dp = - (d(N)/dp * D - N * d(D)/dp) / D^2
            d_tau_dp_ij = - (
                d_e_var_dp_ij * var_y - e_var_y_given_x * d_var_y_dp_ij
            ) / (var_y**2)

            sum_d_sq_p += (d_tau_dp_ij**2) * p_ij
            sum_d_p += d_tau_dp_ij * p_ij

    variance = (sum_d_sq_p - sum_d_p**2) / n
    ase = np.sqrt(max(0, variance))

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
