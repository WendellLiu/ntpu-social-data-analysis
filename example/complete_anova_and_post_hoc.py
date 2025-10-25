import pandas as pd
import scipy.stats as stats
from scipy.stats import levene, f_oneway, tukey_hsd
import scikit_posthocs as sp

# 1. Prepare data as dictionary with labels
groups = {name: group["inc"].values for name, group in df.groupby("agegroup")}

# Display group information
print("Group sizes:")
for name, data in groups.items():
    print(f"  {name}: n={len(data)}")

# 2. Levene's test (needs values only)
levene_stat, levene_p = levene(*groups.values())
print(f"\nLevene's Test: statistic={levene_stat:.4f}, p-value={levene_p:.4f}")

# 3. ANOVA (needs values only)
f_stat, anova_p = f_oneway(*groups.values())
print(f"ANOVA: F-statistic={f_stat:.4f}, p-value={anova_p:.4f}")

# 4. Post-hoc test
if levene_p > 0.05:
    print("\nEqual variances - Tukey's HSD test:")
    # tukey_hsd can use the dictionary directly (scipy >= 1.11)
    tukey_result = tukey_hsd(*groups.values())
    # Print with labels
    group_names = list(groups.keys())
    print(f"Groups: {group_names}")
    print(tukey_result)
else:
    print("\nUnequal variances - Dunnett's T3 test:")
    # scikit_posthocs works directly with dataframe
    dunnett_t3 = sp.posthoc_dunnett(df, val_col="inc", group_col="agegroup")
    print(dunnett_t3)
