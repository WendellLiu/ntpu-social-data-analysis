import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def distribution_plot(
    data,
    xlabel="Value",
    ylabel="Density",
    title="Distribution",
    with_normal_curve=True,
    should_standardize=False,
):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    if should_standardize:
        data_standardized = (data - data.mean()) / data.std()
        data = data_standardized

    # Plot histogram
    ax.hist(
        data,
        bins=30,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label=xlabel,
    )

    # Fit normal distribution and plot
    if with_normal_curve:
        mu, sigma = stats.norm.fit(data)
        x = np.linspace(data.min(), data.max(), 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, "r-", linewidth=2, label=f"Normal (μ={mu:.2f}, σ={sigma:.2f})")

    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.show()
