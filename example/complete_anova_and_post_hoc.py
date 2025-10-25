import pandas as pd
import scipy.stats as stats
from scipy.stats import levene, f_oneway, tukey_hsd
import scikit_posthocs as sp
import numpy as np

# 1. Prepare data
groups = {name: group["inc"].values for name, group in df.groupby("agegroup")}

# Display descriptive statistics
print("=" * 60)
print("Descriptive Statistics")
print("=" * 60)
desc_stats = df.groupby("agegroup")["inc"].agg(["count", "mean", "std", "min", "max"])
print(desc_stats)

# 2. Levene's test
levene_stat, levene_p = levene(*groups.values())
print(f"\n{'=' * 60}")
print("Test of Homogeneity of Variances (Levene's Test)")
print("=" * 60)
print(f"Statistic: {levene_stat:.4f}")
print(f"p-value: {levene_p:.4f}")
print(
    f"Conclusion: {'Equal variances assumed' if levene_p > 0.05 else 'Equal variances not assumed'}"
)

# 3. ANOVA
f_stat, anova_p = f_oneway(*groups.values())
print(f"\n{'=' * 60}")
print("One-Way ANOVA")
print("=" * 60)
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {anova_p:.4f}")
print(
    f"Conclusion: {'Significant difference between groups' if anova_p < 0.05 else 'No significant difference between groups'}"
)

# 4. Post-hoc Multiple Comparisons
print(f"\n{'=' * 60}")
if levene_p > 0.05:
    print("Post-hoc Test: Tukey's HSD Test (Equal variances assumed)")
    print("=" * 60)

    # Tukey's HSD
    tukey_result = tukey_hsd(*groups.values())
    group_names = list(groups.keys())

    # Build comparison results table
    comparisons = []
    n_groups = len(group_names)

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            group_i = group_names[i]
            group_j = group_names[j]
            mean_i = groups[group_i].mean()
            mean_j = groups[group_j].mean()
            mean_diff = mean_i - mean_j

            # Get p-value and confidence interval from tukey_result
            pvalue = tukey_result.pvalue[i, j]
            ci_lower = tukey_result.confidence_interval().low[i, j]
            ci_upper = tukey_result.confidence_interval().high[i, j]

            comparisons.append(
                {
                    "Group (I)": group_i,
                    "Group (J)": group_j,
                    "Mean Diff (I-J)": mean_diff,
                    "Std. Error": (
                        tukey_result.statistic[i, j] / np.sqrt(2)
                        if hasattr(tukey_result, "statistic")
                        else np.nan
                    ),
                    "p-value": pvalue,
                    "95% CI Lower": ci_lower,
                    "95% CI Upper": ci_upper,
                    "Sig.": (
                        "***"
                        if pvalue < 0.001
                        else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "ns"
                    ),
                }
            )

    result_df = pd.DataFrame(comparisons)
    print(result_df.to_string(index=False))

else:
    print("Post-hoc Test: Dunnett's T3 Test (Equal variances not assumed)")
    print("=" * 60)

    # Dunnett's T3
    dunnett_t3 = sp.posthoc_dunnett(df, val_col="inc", group_col="agegroup")

    # Build detailed comparison table
    comparisons = []
    group_names = list(groups.keys())

    for i, group_i in enumerate(group_names):
        for j, group_j in enumerate(group_names):
            if i < j:  # Only show upper triangle
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

                # Calculate confidence interval (approximation)
                df_welch = (var_i / n_i + var_j / n_j) ** 2 / (
                    (var_i / n_i) ** 2 / (n_i - 1) + (var_j / n_j) ** 2 / (n_j - 1)
                )
                t_crit = stats.t.ppf(0.975, df_welch)
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
                                else "*" if pvalue < 0.05 else "ns"
                            )
                        ),
                    }
                )

    result_df = pd.DataFrame(comparisons)
    print(result_df.to_string(index=False))

# 5. Significance notation
print(f"\n{'=' * 60}")
print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant")
print("=" * 60)
