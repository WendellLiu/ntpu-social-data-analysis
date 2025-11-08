import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import pingouin as pg
from itertools import product


def zero_order_correlation_matrix(df, variables, method="pearson"):
    """
    Calculate zero-order correlation matrix for multiple variables using functional programming

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    variables : list
        List of variable names to analyze
    method : str, default 'pearson'
        Correlation method, either 'pearson' or 'spearman'

    Returns:
    --------
    pandas.DataFrame
        Zero-order correlation matrix result table
    """
    # Validate method parameter
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be 'pearson' or 'spearman'")

    # Remove missing values
    data_clean = df[variables].dropna()
    n = len(data_clean)

    # Select correlation function based on method
    corr_func = pearsonr if method == "pearson" else spearmanr

    # Calculate correlation coefficient matrix (vectorized)
    corr_matrix = data_clean.corr(method=method)

    # Calculate p-value matrix
    def calc_pvalue(var1, var2):
        if var1 == var2:
            return np.nan
        _, p = corr_func(data_clean[var1], data_clean[var2])
        return p

    pval_matrix = pd.DataFrame(
        [[calc_pvalue(v1, v2) for v2 in variables] for v1 in variables],
        index=variables,
        columns=variables,
    )

    # Define function to extract statistic values
    def get_stat_value(var1, var2, stat):
        if var1 == var2:
            return {"Correlation": 1.000, "Significance (2-tailed)": ".", "df": 0}[stat]
        return {
            "Correlation": f"{corr_matrix.loc[var1, var2]:.3f}",
            "Significance (2-tailed)": f"{pval_matrix.loc[var1, var2]:.3f}",
            "df": n - 2,
        }[stat]

    # Generate all combinations using product and map to results
    statistics = ["Correlation", "Significance (2-tailed)", "df"]

    results = pd.DataFrame(
        list(
            map(
                lambda combo: {
                    "Variable": combo[0],
                    "Statistic": combo[1],
                    **{
                        var: get_stat_value(combo[0], var, combo[1])
                        for var in variables
                    },
                },
                product(variables, statistics),
            )
        )
    )

    return results


def partial_correlation_matrix(df, variables, covar, method="pearson"):
    """
    Calculate partial correlation matrix for multiple variables using functional programming

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    variables : list
        List of variable names to analyze
    covar : str or list
        Control variable name(s), can be single variable or list of variables
    method : str, default 'pearson'
        Correlation method, either 'pearson' or 'spearman'

    Returns:
    --------
    pandas.DataFrame
        Partial correlation matrix result table
    """
    # Validate method parameter
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be 'pearson' or 'spearman'")

    # Ensure covar is in list format
    covar_list = [covar] if isinstance(covar, str) else covar

    # Remove missing values
    all_vars = variables + covar_list
    data_clean = df[all_vars].dropna()
    n = len(data_clean)

    # Calculate partial correlation for all variable pairs
    def calc_partial_corr(pair):
        var1, var2 = pair
        if var1 == var2:
            return {"r": 1.0, "p": np.nan}
        result = pg.partial_corr(
            data=data_clean, x=var1, y=var2, covar=covar_list, method=method
        )
        return {"r": result["r"].values[0], "p": result["p-val"].values[0]}

    # Vectorized calculation of all partial correlations
    var_pairs = list(product(variables, variables))
    partial_results = dict(zip(var_pairs, map(calc_partial_corr, var_pairs)))

    # Define function to extract statistic values
    def get_stat_value(var1, var2, stat):
        result = partial_results[(var1, var2)]
        if var1 == var2:
            return {"Correlation": 1.000, "Significance (2-tailed)": ".", "df": 0}[stat]
        return {
            "Correlation": f"{result['r']:.3f}",
            "Significance (2-tailed)": f"{result['p']:.3f}",
            "df": n - 2 - len(covar_list),
        }[stat]

    # Generate all combinations using product and map to results
    statistics = ["Correlation", "Significance (2-tailed)", "df"]
    control_var_str = ", ".join(covar_list)

    results = pd.DataFrame(
        list(
            map(
                lambda combo: {
                    "Control Variable": control_var_str,
                    "Variable": combo[0],
                    "Statistic": combo[1],
                    **{
                        var: get_stat_value(combo[0], var, combo[1])
                        for var in variables
                    },
                },
                product(variables, statistics),
            )
        )
    )

    return results
