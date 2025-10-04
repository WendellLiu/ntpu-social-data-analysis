import numpy as np
import pandas as pd

from statistic.percentile_rank import create_percentile_rank_calculator


def get_distribution_summary(series: pd.Series, sort_by="value", ascending=True):
    """
    Get a summary DataFrame with value, percentile, and counts from a series.

    Parameters:
    -----------
    series : pd.Series
        The reference data
    sort_by : str, default 'value'
        Column to sort by ('value', 'percentile', or 'counts')
    ascending : bool, default True
        Sort order

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['value', 'percentile', 'counts']
    """
    # Remove NaN values
    clean_data = series.dropna()

    # Create percentile rank calculator
    rank_calculator = create_percentile_rank_calculator(series)

    # Get unique values and their counts
    unique_values, counts = np.unique(clean_data, return_counts=True)

    # Calculate percentile for each unique value
    percentiles = [rank_calculator(val) for val in unique_values]

    # Calculate z-score for each unique value
    mean = series.mean()
    std = series.std()
    z_scores = (unique_values - mean) / std

    # Create DataFrame
    df = pd.DataFrame(
        {
            "value": unique_values,
            "percentile": percentiles,
            "z-score": z_scores,
            "counts": counts,
        }
    )

    # Sort by specified column
    df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    return df
