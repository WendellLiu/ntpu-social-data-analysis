import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def simple_linear_regression_model(y_series, x_series):
    X = sm.add_constant(x_series)  # Adds intercept
    Y = y_series
    model = sm.OLS(Y, X).fit()

    return model


def simple_linear_regression_report(y_series, x_series):
    model = simple_linear_regression_model(y_series, x_series)

    return model.summary()


def simultaneous_linear_regression_model(y_series, x_series):
    X = sm.add_constant(x_series)  # Adds intercept
    Y = y_series
    model = sm.OLS(Y, X).fit()

    return model


def simultaneous_linear_regression_report(y_series, x_series):
    model = simultaneous_linear_regression_model(y_series, x_series)

    return model.summary()


def condition_index_analysis(X):
    """
    Calculate Condition Index with variance decomposition proportions (SPSS style)

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Design matrix including intercept term

    Returns:
    --------
    pd.DataFrame : CI analysis report with variance decomposition proportions
    """
    X = sm.add_constant(X)

    # Get variable names
    var_names = (
        X.columns
        if isinstance(X, pd.DataFrame)
        else [f"X{i}" for i in range(X.shape[1])]
    )
    X_array = X.values if isinstance(X, pd.DataFrame) else X

    # Standardize the design matrix
    X_scaled = X_array / np.sqrt(np.sum(X_array**2, axis=0))

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(X_scaled.T @ X_scaled)

    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate Condition Index
    max_eigenvalue = eigenvalues.max()
    condition_indices = np.sqrt(max_eigenvalue / eigenvalues)

    # Calculate variance decomposition proportions
    phi = eigenvectors**2
    variance_props = phi / phi.sum(axis=1, keepdims=True)

    # Build report
    ci_report = pd.DataFrame(
        {
            "Dimension": range(1, len(eigenvalues) + 1),
            "Eigenvalue": eigenvalues,
            "Condition_Index": condition_indices,
        }
    )

    # Add variance decomposition proportions for each variable
    for i, var_name in enumerate(var_names):
        ci_report[f"Variance_Proportion_{var_name}"] = variance_props[i, :]

    return ci_report


def vif_analysis(X):
    """
    Calculate VIF and return DataFrame report

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Design matrix including intercept term

    Returns:
    --------
    pd.DataFrame : VIF analysis report
    """

    # Calculate VIF
    vif_values = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(
            X.values if isinstance(X, pd.DataFrame) else X, i
        )
        vif_values.append(vif)

    # Build report
    vif_report = pd.DataFrame(
        {
            "Variable": (
                X.columns
                if isinstance(X, pd.DataFrame)
                else [f"X{i}" for i in range(X.shape[1])]
            ),
            "VIF": vif_values,
        }
    )

    return vif_report


def mediation_analysis(X, Z, Y):
    """
    Mediation analysis using statsmodels summary

    Parameters:
    - X: pandas Series (independent variable)
    - Z: pandas Series (mediator variable)
    - Y: pandas Series (dependent variable)

    Returns:
    - Dictionary with model summaries and summary tables
    """

    # Model 1: Y = c + b*X (Total Effect)
    X1 = sm.add_constant(X)
    total_effect_model = sm.OLS(Y, X1).fit()

    # Model 2: Z = c + b*X (Mediator Model)
    X2 = sm.add_constant(X)
    mediator_model = sm.OLS(Z, X2).fit()

    # Model 3: Y = c + b_x*X + b_z*Z (Direct Effect)
    XZ = pd.DataFrame({"const": 1, X.name: X, Z.name: Z})
    direct_effect_model = sm.OLS(Y, XZ).fit()

    return {
        "total_effect_model": total_effect_model,
        "total_effect_summary": total_effect_model.summary(),
        "mediator_model": mediator_model,
        "mediator_summary": mediator_model.summary(),
        "direct_effect_model": direct_effect_model,
        "direct_effect_summary": direct_effect_model.summary(),
    }
