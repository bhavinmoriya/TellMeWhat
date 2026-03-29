def monte_carlo_option_pricing(S0, K, T, r, sigma, n_simulations=10000):
    """
    S0: Initial stock price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    """
    np.random.seed(42)
    dt = T / 252  # Daily time step
    n_steps = int(T / dt)

    # Simulate stock price paths
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0] = S0
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_simulations)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # Calculate payoffs
    payoffs = np.maximum(paths[-1] - K, 0)

    # Discount payoffs to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)

    return option_price, std_error

# Example: Price a call option
S0 = 100    # Current stock price
K = 105     # Strike price
T = 1       # 1 year to maturity
r = 0.05    # 5% risk-free rate
sigma = 0.2 # 20% volatility

price, error = monte_carlo_option_pricing(S0, K, T, r, sigma)
print(f"Option Price: ${price:.2f} ± {error:.4f}")


K_values = np.arange(80, 120, 1)
option_prices = []

for K_val in K_values:
    price, _ = monte_carlo_option_pricing(S0, K_val, T, r, sigma)
    option_prices.append(price)

plt.figure(figsize=(10, 6))
plt.plot(K_values, option_prices, marker='o', linestyle='-', color='blue')
plt.title('Option Price vs. Strike Price (K)')
plt.xlabel('Strike Price (K)')
plt.ylabel('Option Price')
plt.grid(True)
plt.show()
