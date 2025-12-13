import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


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
