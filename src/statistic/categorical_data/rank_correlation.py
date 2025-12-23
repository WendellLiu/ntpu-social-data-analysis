import numpy as np
import pandas as pd
from scipy import stats


def rank_correlation_measures(data, col1, col2):
    """
    Calculate Goodman & Kruskal's Gamma and Kendall's Tau-c with ASE and Significance.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the ordinal variables.
    col1 : str
        Name of the first ordinal variable (row variable).
    col2 : str
        Name of the second ordinal variable (column variable).

    Returns:
    --------
    pd.DataFrame : DataFrame with columns ['Measure', 'Value', 'ASE', 'Approx_T', 'Approx_Sig']
    """
    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if col1 not in data.columns:
        raise ValueError(f"Column '{col1}' not found in DataFrame")
    if col2 not in data.columns:
        raise ValueError(f"Column '{col2}' not found in DataFrame")

    # Extract series and remove missing values
    mask = data[col1].notna() & data[col2].notna()
    s1 = data[col1][mask]
    s2 = data[col2][mask]

    if len(s1) == 0:
        raise ValueError("No valid observations after removing missing values")

    # Create contingency table
    crosstab = pd.crosstab(s1, s2)
    n = crosstab.sum().sum()

    results = []
    intermediate_results = _calculate_concordant_discordant_matrices(crosstab)

    # Goodman & Kruskal's Gamma
    gamma_val, gamma_ase = _goodman_kruskal_gamma(crosstab, n, intermediate_results)
    z_score_gamma = gamma_val / gamma_ase if gamma_ase > 0 else 0
    p_value_gamma = 2 * (1 - stats.norm.cdf(abs(z_score_gamma)))
    results.append(
        {
            "Measure": "Goodman & Kruskal's Gamma",
            "Value": gamma_val,
            "ASE": gamma_ase,
            "Approx_T": z_score_gamma,
            "Approx_Sig": p_value_gamma,
        }
    )

    # Kendall's Tau-c
    tau_c_val, tau_c_ase = _kendalls_tau_c(crosstab, n, intermediate_results)
    z_score_tau_c = tau_c_val / tau_c_ase if tau_c_ase > 0 else 0
    p_value_tau_c = 2 * (1 - stats.norm.cdf(abs(z_score_tau_c)))
    results.append(
        {
            "Measure": "Kendall's Tau-c",
            "Value": tau_c_val,
            "ASE": tau_c_ase,
            "Approx_T": z_score_tau_c,
            "Approx_Sig": p_value_tau_c,
        }
    )

    return pd.DataFrame(results)


def _calculate_concordant_discordant_matrices(crosstab):
    """
    Calculates Nc, Nd, and matrices C and D for variance calculation.
    C_ij = concordant pairs with cell (i,j)
    D_ij = discordant pairs with cell (i,j)
    """
    values = crosstab.values.astype(float)
    r, c = values.shape

    C_matrix = np.zeros_like(values)
    D_matrix = np.zeros_like(values)
    Nc = 0
    Nd = 0

    for i in range(r):
        for j in range(c):
            # Concordant pairs
            sum_greater = np.sum(values[i + 1 :, j + 1 :])
            sum_lesser = np.sum(values[:i, :j])
            C_matrix[i, j] = sum_greater + sum_lesser
            Nc += values[i, j] * sum_greater

            # Discordant pairs
            sum_greater_discordant = np.sum(values[i + 1 :, :j])
            sum_lesser_discordant = np.sum(values[:i, j + 1 :])
            D_matrix[i, j] = sum_greater_discordant + sum_lesser_discordant
            Nd += values[i, j] * sum_greater_discordant

    return {"Nc": Nc, "Nd": Nd, "C_matrix": C_matrix, "D_matrix": D_matrix}


def _goodman_kruskal_gamma(crosstab, n, intermediate_results):
    """Calculates Goodman & Kruskal's Gamma and its ASE."""
    Nc = intermediate_results["Nc"]
    Nd = intermediate_results["Nd"]
    C_matrix = intermediate_results["C_matrix"]
    D_matrix = intermediate_results["D_matrix"]
    values = crosstab.values.astype(float)

    # Calculate Gamma
    denominator_g = Nc + Nd
    if denominator_g == 0:
        return 0.0, 0.0
    gamma = (Nc - Nd) / denominator_g

    # Calculate ASE for Gamma
    # Using formula from Brown & Benedetti (1977)
    psi_matrix = Nd * C_matrix - Nc * D_matrix
    variance_g = 16 * np.sum(values * (psi_matrix**2)) / (denominator_g**4)
    ase_g = np.sqrt(max(0, variance_g))

    return gamma, ase_g


def _kendalls_tau_c(crosstab, n, intermediate_results):
    """Calculates Kendall's Tau-c and its ASE."""
    Nc = intermediate_results["Nc"]
    Nd = intermediate_results["Nd"]
    C_matrix = intermediate_results["C_matrix"]
    D_matrix = intermediate_results["D_matrix"]
    values = crosstab.values.astype(float)
    r, c = values.shape
    m = min(r, c)

    # Calculate Tau-c
    denominator_t = (n**2 * (m - 1)) / (2 * m)
    if denominator_t == 0:
        return 0.0, 0.0
    tau_c = (Nc - Nd) / denominator_t

    # Calculate ASE for Tau-c
    # Using an estimator for the variance of (Nc - Nd)
    # Var(Nc - Nd) approx. sum(f_ij * [ (C_ij - D_ij) - (Nc - Nd)/n ]^2)
    var_pq = np.sum(values * (C_matrix - D_matrix - (Nc - Nd) / n) ** 2)

    # Var(tau_c) = Var(Nc - Nd) / denominator_t^2
    var_tau_c = var_pq / (denominator_t**2)

    ase_t = np.sqrt(max(0, var_tau_c))

    return tau_c, ase_t
