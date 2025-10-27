from scipy.stats import levene, ttest_ind, ttest_rel
import pandas as pd


def compare_independent_samples(series1, series2):
    # Levene test
    levene_result = levene(series1, series2)

    # Student's t-test (equal variance)
    eqaul_var_result = ttest_ind(series1, series2, equal_var=True)

    # Welch's t-test (unequal variance)
    equal_var_not_assumed_result = ttest_ind(series1, series2, equal_var=False)

    # Compile results
    results = {
        "levene": {
            "statistic": levene_result.statistic,
            "pvalue": levene_result.pvalue,
        },
        "equal_var_assumed": {
            "statistic": eqaul_var_result.statistic,
            "pvalue": eqaul_var_result.pvalue,
        },
        "equal_var_not_assumed": {
            "statistic": equal_var_not_assumed_result.statistic,
            "pvalue": equal_var_not_assumed_result.pvalue,
        },
    }

    df = pd.DataFrame(
        {
            "Statistic": [
                results["levene"]["statistic"],
                results["equal_var_assumed"]["statistic"],
                results["equal_var_not_assumed"]["statistic"],
            ],
            "p-value": [
                results["levene"]["pvalue"],
                results["equal_var_assumed"]["pvalue"],
                results["equal_var_not_assumed"]["pvalue"],
            ],
        },
        index=[
            "Levene's Test for Equal Variances",
            "t-test (Equal Variance Assumed)",
            "t-test (Equal Variance Not Assumed)",
        ],
    )

    return df


def compare_paired_samples(series1, series2):
    # Check assumptions
    differences = series1 - series2
    mean = differences.mean()
    std = differences.std()

    # Perform paired t-test
    t_stat, p_value = ttest_rel(series1, series2)

    result = {
        "diff_mean": mean,
        "diff_std": std,
        "t_stat": t_stat,
        "p_value": p_value,
    }

    return result
