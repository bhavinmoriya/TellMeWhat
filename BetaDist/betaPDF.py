import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from scipy.stats import beta
from ipywidgets import interact, IntSlider


def plot_beta(alpha: int = 1, beta_param: int = 1):
    # x-axis (possible probabilities)
    x = np.linspace(0.001, 0.999, 500)

    # Beta density
    y = beta.pdf(x, alpha, beta_param)

    # Store in Polars DataFrame
    df = pl.DataFrame(
        {
            "theta": x,
            "density": y,
        }
    )

    # Posterior statistics
    mean = alpha / (alpha + beta_param)
    mode = (
        (alpha - 1) / (alpha + beta_param - 2)
        if alpha > 1 and beta_param > 1
        else None
    )

    plt.figure(figsize=(9, 5))
    plt.plot(df["theta"], df["density"], linewidth=3)

    plt.axvline(
        mean,
        linestyle="--",
        label=f"Mean = {mean:.3f}",
    )

    if mode is not None:
        plt.axvline(
            mode,
            linestyle=":",
            label=f"Mode = {mode:.3f}",
        )

    plt.title(f"Beta({alpha}, {beta_param})")
    plt.xlabel("θ (possible probability)")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    print(f"Mean : {mean:.4f}")

    if mode is not None:
        print(f"Mode : {mode:.4f}")

    print(f"Variance : {alpha * beta_param / ((alpha + beta_param)**2 * (alpha + beta_param + 1)):.5f}")


interact(
    plot_beta,
    alpha=IntSlider(min=1, max=50, step=1, value=1),
    beta_param=IntSlider(min=1, max=50, step=1, value=1),
);
