import pandas as pd


def describe(series):
    df_describe = series.describe()
    mean = df_describe["mean"]
    std = df_describe["std"]
    median = df_describe["50%"]
    _min = df_describe["min"]
    _max = df_describe["max"]
    q1 = df_describe["25%"]
    q3 = df_describe["75%"]
    mode = series.mode()
    interquartile_range = q3 - q1

    answer = pd.DataFrame(
        {"Value": [mean, median, mode, _max - _min, q1, q3, interquartile_range, std]},
        index=[
            "Mean",
            "Median",
            "Mode",
            "Range",
            "Q1",
            "Q3",
            "Interquartile Range",
            "Standard Deviation",
        ],
    )
    return answer
