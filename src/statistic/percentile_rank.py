import numpy as np
import pandas as pd
from scipy import stats


def create_percentile_rank_calculator(series: pd.Series):
    clean_data = series.dropna().sort_values().values

    def percentile_rank(value):
        percentile = stats.percentileofscore(clean_data, value)

        return percentile

    return percentile_rank
