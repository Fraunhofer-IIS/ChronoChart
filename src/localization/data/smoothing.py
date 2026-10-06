#%%
import numpy as np
import torch

def smooth_signal(t, lam):
    """
    Solve:
        min_x sum (t_i - x_i)^2 + lam * sum (x_i - x_{i-1})^2

    Args:
        t   : numpy array (n,)
        lam : smoothing strength (lambda)

    Returns:
        x : smoothed signal
    """
    n = len(t)

    # Diagonals of the matrix A = I + lam * D^T D
    main_diag = torch.ones(n)
    off_diag = -lam * torch.ones(n - 1)

    # Build main diagonal
    main_diag[0] += lam
    main_diag[-1] += lam
    main_diag[1:-1] += 2 * lam

    # Thomas algorithm (tridiagonal solver)

    # Forward sweep
    c = torch.zeros(n - 1)
    d = torch.zeros(n)

    c[0] = off_diag[0] / main_diag[0]
    d[0] = t[0] / main_diag[0]

    for i in range(1, n - 1):
        denom = main_diag[i] - off_diag[i - 1] * c[i - 1]
        c[i] = off_diag[i] / denom
        d[i] = (t[i] - off_diag[i - 1] * d[i - 1]) / denom

    d[n - 1] = (t[n - 1] - off_diag[n - 2] * d[n - 2]) / (
        main_diag[n - 1] - off_diag[n - 2] * c[n - 2]
    )

    # Back substitution
    x = torch.zeros(n)
    x[-1] = d[-1]

    for i in reversed(range(n - 1)):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


if __name__ == "__main__":
    from src.localization.data.loaders import DichasusLoader,FiveGLoader
    
    loader = DichasusLoader()
    dataset = loader.load_dataset()

    t = dataset.timestamps
    from matplotlib import pyplot as plt

    plt.plot(t, label="Original")
    x = smooth_signal(t, lam=10000)
    plt.plot(x, label="Smoothed")
    plt.legend()
    plt.title("Smoothed Timestamps")
    plt.xlabel("Index")
    plt.ylabel("Timestamp (s)")
    plt.show()
