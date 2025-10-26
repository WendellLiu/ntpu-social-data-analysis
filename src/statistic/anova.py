import pingouin as pg
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def perform_anova(df, val_col, group_col):
    anova_result = pg.anova(data=df, dv=val_col, between=group_col)
    return anova_result


def perform_post_hoc(
    df,
    levene_p,
    val_col,
    group_col,
):
    if levene_p > 0.05:
        tukey_result = pg.pairwise_tukey(data=df, dv=val_col, between=group_col)
        return tukey_result
    else:
        gh_result = pg.pairwise_gameshowell(data=df, dv=val_col, between=group_col)
        return gh_result


def perform_ancova(df, val_col, group_col, covar_col, ss_type=1):
    """
    Perform ANCOVA analysis

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    val_col : str
        Dependent variable (outcome)
    group_col : str
        Independent variable (grouping factor)
    covar_col : str
        Covariate
    ss_type : int, default=2
        Sum of squares type: 1 (Type I), 2 (Type II), 3 (Type III)

    Returns:
    --------
    pd.DataFrame : ANCOVA table
    """

    # Check columns
    required_cols = [val_col, group_col, covar_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Check ss_type
    if ss_type not in [1, 2, 3]:
        raise ValueError("ss_type must be 1, 2, or 3")

    # Remove missing values
    df_clean = df[required_cols].dropna()

    # Build model
    formula = f"{val_col} ~ C({group_col}) + {covar_col}"
    model = ols(formula, data=df_clean).fit()

    # Calculate ANCOVA table
    ancova_table = anova_lm(model, typ=ss_type)

    return ancova_table


def check_linearity(df, val_col, covar_col, ss_type=3):
    """Check linearity: val ~ covar"""
    df_clean = df[[val_col, covar_col]].dropna()
    formula = f"{val_col} ~ {covar_col}"
    model = ols(formula, data=df_clean).fit()
    return anova_lm(model, typ=ss_type)


def check_interaction(df, val_col, group_col, covar_col, ss_type=3):
    """Check interaction: val ~ group * covar"""
    required_cols = [val_col, group_col, covar_col]
    df_clean = df[required_cols].dropna()
    formula = f"{val_col} ~ C({group_col}) * {covar_col}"
    model = ols(formula, data=df_clean).fit()
    return anova_lm(model, typ=ss_type)
