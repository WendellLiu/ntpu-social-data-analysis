import pandas as pd
import scipy.stats as stats
from scipy.stats import levene, f_oneway, tukey_hsd
import scikit_posthocs as sp
import numpy as np


def post_hoc_comparison(df, value_col, group_col, control=None, alpha=0.05):
    """
    Perform post-hoc multiple comparison tests after ANOVA.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    value_col : str
        Column name for the dependent variable (numeric values)
    group_col : str
        Column name for the grouping variable (categories)
    alpha : float, default=0.05
        Significance level for tests

    Returns:
    --------
    dict : Dictionary containing all test results
        - 'descriptive_stats': Descriptive statistics by group
        - 'levene_test': Levene's test results
        - 'anova_test': ANOVA results
        - 'post_hoc_results': Post-hoc comparison results dataframe
        - 'test_used': Name of post-hoc test used
    """

    # 1. Prepare data
    groups = {name: group[value_col].values for name, group in df.groupby(group_col)}

    # Descriptive statistics
    print("=" * 60)
    print("Descriptive Statistics")
    print("=" * 60)
    desc_stats = df.groupby(group_col)[value_col].agg(
        ["count", "mean", "std", "min", "max"]
    )
    print(desc_stats)

    # 2. Levene's test
    levene_stat, levene_p = levene(*groups.values())
    print(f"\n{'=' * 60}")
    print("Test of Homogeneity of Variances (Levene's Test)")
    print("=" * 60)
    print(f"Statistic: {levene_stat:.4f}")
    print(f"p-value: {levene_p:.4f}")
    print(
        f"Conclusion: {'Equal variances assumed' if levene_p > alpha else 'Equal variances not assumed'}"
    )

    # 3. ANOVA
    f_stat, anova_p = f_oneway(*groups.values())
    print(f"\n{'=' * 60}")
    print("One-Way ANOVA")
    print("=" * 60)
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {anova_p:.4f}")
    print(
        f"Conclusion: {'Significant difference between groups' if anova_p < alpha else 'No significant difference between groups'}"
    )

    # 4. Post-hoc Multiple Comparisons
    print(f"\n{'=' * 60}")

    if levene_p > alpha:
        # Tukey's HSD
        print("Post-hoc Test: Tukey's HSD Test (Equal variances assumed)")
        print("=" * 60)

        tukey_result = tukey_hsd(*groups.values())
        group_names = list(groups.keys())

        comparisons = []
        n_groups = len(group_names)

        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                group_i = group_names[i]
                group_j = group_names[j]
                mean_i = groups[group_i].mean()
                mean_j = groups[group_j].mean()
                mean_diff = mean_i - mean_j

                pvalue = tukey_result.pvalue[i, j]
                ci_lower = tukey_result.confidence_interval().low[i, j]
                ci_upper = tukey_result.confidence_interval().high[i, j]

                comparisons.append(
                    {
                        "Group (I)": group_i,
                        "Group (J)": group_j,
                        "Mean Diff (I-J)": mean_diff,
                        "p-value": pvalue,
                        "95% CI Lower": ci_lower,
                        "95% CI Upper": ci_upper,
                        "Sig.": (
                            "***"
                            if pvalue < 0.001
                            else (
                                "**"
                                if pvalue < 0.01
                                else "*" if pvalue < alpha else "ns"
                            )
                        ),
                    }
                )

        test_used = "Tukey's HSD"

    else:
        # Dunnett's T3
        print("Post-hoc Test: Dunnett's T3 Test (Equal variances not assumed)")
        print("=" * 60)

        dunnett_t3 = sp.posthoc_dunnett(
            df, val_col=value_col, group_col=group_col, control=control
        )

        comparisons = []
        group_names = list(groups.keys())

        for i, group_i in enumerate(group_names):
            for j, group_j in enumerate(group_names):
                if i < j:
                    mean_i = groups[group_i].mean()
                    mean_j = groups[group_j].mean()
                    mean_diff = mean_i - mean_j
                    pvalue = dunnett_t3.loc[group_i, group_j]

                    # Calculate standard error (Welch's formula)
                    n_i = len(groups[group_i])
                    n_j = len(groups[group_j])
                    var_i = groups[group_i].var()
                    var_j = groups[group_j].var()
                    se = np.sqrt(var_i / n_i + var_j / n_j)

                    # Calculate confidence interval
                    df_welch = (var_i / n_i + var_j / n_j) ** 2 / (
                        (var_i / n_i) ** 2 / (n_i - 1) + (var_j / n_j) ** 2 / (n_j - 1)
                    )
                    t_crit = stats.t.ppf(1 - alpha / 2, df_welch)
                    ci_lower = mean_diff - t_crit * se
                    ci_upper = mean_diff + t_crit * se

                    comparisons.append(
                        {
                            "Group (I)": group_i,
                            "Group (J)": group_j,
                            "Mean Diff (I-J)": mean_diff,
                            "Std. Error": se,
                            "p-value": pvalue,
                            "95% CI Lower": ci_lower,
                            "95% CI Upper": ci_upper,
                            "Sig.": (
                                "***"
                                if pvalue < 0.001
                                else (
                                    "**"
                                    if pvalue < 0.01
                                    else "*" if pvalue < alpha else "ns"
                                )
                            ),
                        }
                    )

        test_used = "Dunnett's T3"

    result_df = pd.DataFrame(comparisons)
    print(result_df.to_string(index=False))

    # Significance notation
    print(f"\n{'=' * 60}")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant")
    print("=" * 60)

    # Return results
    return {
        "descriptive_stats": desc_stats,
        "levene_test": {"statistic": levene_stat, "p_value": levene_p},
        "anova_test": {"f_statistic": f_stat, "p_value": anova_p},
        "post_hoc_results": result_df,
        "test_used": test_used,
    }


# Usage example:
# results = post_hoc_comparison(df, value_col='inc', group_col='agegroup', alpha=0.05)
#
# # Access specific results:
# print(results['test_used'])
# print(results['post_hoc_results'])
