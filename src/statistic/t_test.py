from scipy.stats import levene, ttest_ind


def compare_two_samples(series1, series2):
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
