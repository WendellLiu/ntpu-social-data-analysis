import pandas as pd


def check_missing_value(df):
    missing_counts = df.isna().sum()
    result_df = pd.DataFrame(
        {"# of Missing Values": missing_counts.values}, index=missing_counts.index
    )
    return result_df
