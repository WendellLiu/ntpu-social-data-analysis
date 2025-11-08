import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatter(df, x, y, figsize=(8, 6)):
    """
    Plot simple scatter plot for two variables in Jupyter Lab

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    x : str
        X-axis variable name
    y : str
        Y-axis variable name
    figsize : tuple, default (8, 6)
        Figure size (width, height)

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing scatter plot
    """
    # Remove missing values
    data_clean = df[[x, y]].dropna()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set style
    sns.set_style("whitegrid")

    # Scatter plot
    ax.scatter(
        data_clean[x],
        data_clean[y],
        alpha=0.6,
        s=50,
        color="coral",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel(x, fontsize=12, fontweight="bold")
    ax.set_ylabel(y, fontsize=12, fontweight="bold")
    ax.set_title(f"{x} vs {y}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Adjust layout
    # plt.tight_layout()

    return fig
