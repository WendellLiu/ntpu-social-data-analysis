import pandas as pd
from scipy.stats.contingency import association


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

    return pd.DataFrame(
        {
            "Value": [v],
        },
        index=["Cramer's V"],
    )
