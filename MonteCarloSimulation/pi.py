import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_samples = 100000

# Generate random points in the unit square
x = np.random.uniform(-1, 1, n_samples)
y = np.random.uniform(-1, 1, n_samples)

# Check if points are inside the unit circle
inside = (x**2 + y**2) <= 1

# Estimate Pi
pi_estimate = 4 * np.sum(inside) / n_samples

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(x[inside], y[inside], color='blue', s=1, alpha=0.5)
plt.scatter(x[~inside], y[~inside], color='red', s=1, alpha=0.5)
plt.title(f"Estimating Pi: π ≈ {pi_estimate:.4f}")
plt.axis('equal')
plt.show()
