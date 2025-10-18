from scipy.stats import levene, ttest_ind, ttest_rel


def compare_independent_samples(series1, series2):
    # Levene test
    levene_result = levene(series1, series2)

    # Student's t-test (equal variance)
    student_result = ttest_ind(series1, series2, equal_var=True)

    # Welch's t-test (unequal variance)
    welch_result = ttest_ind(series1, series2, equal_var=False)

    # Compile results
    results = {
        "levene": {
            "statistic": levene_result.statistic,
            "pvalue": levene_result.pvalue,
        },
        "student_test": {
            "statistic": student_result.statistic,
            "pvalue": student_result.pvalue,
        },
        "welch_t": {"statistic": welch_result.statistic, "pvalue": welch_result.pvalue},
    }

    return results


def compare_paired_samples(series1, series2):
    # Check assumptions
    differences = series1 - series2
    mean = differences.mean()
    std = differences.std()

    # Perform paired t-test
    t_stat, p_value = ttest_rel(series1, series2)

    result = {
        "mean": mean,
        "std": std,
        "t_stat": t_stat,
        "p_value": p_value,
    }

    return result
