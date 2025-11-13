import statsmodels.api as sm


def simple_linear_regression_model(y_series, x_series):
    X = sm.add_constant(x_series)  # Adds intercept
    Y = y_series
    model = sm.OLS(Y, X).fit()

    return model


def simple_linear_regression_report(y_series, x_series):
    model = simple_linear_regression_model(y_series, x_series)

    return model.summary()
