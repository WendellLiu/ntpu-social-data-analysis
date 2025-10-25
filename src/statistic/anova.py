import pingouin as pg


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
