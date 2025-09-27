import pandas as pd
from scipy.stats import pearsonr


def pearson_correlation_table(df, col1, col2):
    """Create detailed Pearson correlation table using scipy.stats"""
    # Calculate Pearson correlation
    corr_coef, p_value = pearsonr(df[col1], df[col2])

    # Create summary table
    results = pd.DataFrame(
        {
            "Metric": [
                "Pearson Correlation Coefficient",
                "P-value",
                "Sample Size",
                "Significance Level (α=0.05)",
                "Interpretation",
            ],
            "Value": [
                f"{corr_coef:.4f}",
                f"{p_value:.4f}",
                len(df),
                "Significant" if p_value < 0.05 else "Not Significant",
                interpret_correlation(corr_coef),
            ],
        }
    )

    return results, corr_coef, p_value


def interpret_correlation(r):
    """Interpret correlation strength"""
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "Strong"
    elif abs_r >= 0.3:
        strength = "Moderate"
    else:
        strength = "Weak"

    direction = "Positive" if r > 0 else "Negative"

    return f"{strength} {direction}"
