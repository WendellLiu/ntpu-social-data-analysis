import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association


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


# def lambda_cramer_report(df, var1="a1", var2="c1b"):
#     """
#     Generate a report for Lambda Test and Cramer's V for two categorical variables.
#
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         Input dataframe
#     var1 : str
#         Name of first variable
#     var2 : str
#         Name of second variable
#
#     Returns:
#     --------
#     pandas.DataFrame
#         Report table with test statistics
#     """
#
#     # Create contingency table
#     contingency_table = pd.crosstab(df[var1], df[var2])
#
#     # Chi-square test
#     chi2, p_value, dof, expected = chi2_contingency(contingency_table)
#
#     # Sample size
#     n = contingency_table.sum().sum()
#
#     # Cramer's V
#     min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
#     cramers_v = np.sqrt(chi2 / (n * min_dim))
#
#     # Lambda (Goodman-Kruskal's Lambda)
#     # Lambda (row|col) - predicting row given column
#     col_max = contingency_table.max(axis=0).sum()
#     row_total_max = contingency_table.sum(axis=1).max()
#     lambda_rc = (col_max - row_total_max) / (n - row_total_max)
#
#     # Lambda (col|row) - predicting column given row
#     row_max = contingency_table.max(axis=1).sum()
#     col_total_max = contingency_table.sum(axis=0).max()
#     lambda_cr = (row_max - col_total_max) / (n - col_total_max)
#
#     # Symmetric Lambda
#     lambda_symmetric = (col_max + row_max - row_total_max - col_total_max) / (
#         2 * n - row_total_max - col_total_max
#     )
#
#     # Create report
#     report = pd.DataFrame(
#         {
#             "Statistic": [
#                 "Sample Size (N)",
#                 "Chi-Square",
#                 "Degrees of Freedom",
#                 "P-Value",
#                 f"Cramer's V",
#                 f"Lambda ({var1}|{var2})",
#                 f"Lambda ({var2}|{var1})",
#                 "Lambda (Symmetric)",
#                 "Interpretation",
#             ],
#             "Value": [
#                 f"{n:.0f}",
#                 f"{chi2:.4f}",
#                 f"{dof:.0f}",
#                 f"{p_value:.4f}",
#                 f"{cramers_v:.4f}",
#                 f"{lambda_rc:.4f}",
#                 f"{lambda_cr:.4f}",
#                 f"{lambda_symmetric:.4f}",
#                 "Significant" if p_value < 0.05 else "Not Significant",
#             ],
#         }
#     )
#
#     return report
