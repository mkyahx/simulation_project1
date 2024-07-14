import numpy as np
import matplotlib.pyplot as plt

# Parameter settings
n_companies = 3
n_samples = 10000  # Number of simulation samples
rho = 0.6  # Weight factor
x_thresholds = np.array([2, 3, 1])  # Thresholds for companies i

p_initial = np.array([1/n_companies] * n_companies)  # Initial investment proportions evenly distributed

# Generate random variables
V = np.random.normal(0, 1, n_samples)  # Common factor, standard normal distribution
W = np.random.exponential(1 / 0.3, n_samples)  # Market shock, exponential distribution
eta = np.random.normal(0, 1, (n_samples, n_companies))  # Company-specific risks

# Calculate the market value Xi of companies
X = (rho * V[:, np.newaxis] + np.sqrt(1 - rho**2) * eta) / (W[:, np.newaxis] + 1)

# Calculate the investment returns Yi, if Xi < xi, no profit is generated
Y = np.maximum(X - x_thresholds[np.newaxis, :], 0)

# Sharpe ratio function
def sharpe_ratio(p, Y):
    returns = np.dot(Y, p)
    return np.mean(returns) / np.std(returns)

# SPSA optimization function
def spsa_optimize(p_initial, a, c, A, alpha, gamma, iterations, Y, x_thresholds):
    p = p_initial
    sharpe_ratios_history = []
    expected_returns_history = []
    p_history = []

    for k in range(iterations):
        ak = a / (k + 1 + A)**alpha
        ck = c / (k + 1)**gamma

        delta = 2 * (np.random.rand(n_companies) - 0.5)  # Generate perturbation vector
        p_plus = p + ck * delta
        p_minus = p - ck * delta

        # Ensure non-negative investment proportions and sum to 1
        p_plus = np.clip(p_plus, 0, 1)
        p_plus /= np.sum(p_plus)
        p_minus = np.clip(p_minus, 0, 1)
        p_minus /= np.sum(p_minus)

        # Calculate perturbed Sharpe ratios
        sharpe_plus = sharpe_ratio(p_plus, Y)
        sharpe_minus = sharpe_ratio(p_minus, Y)

        # Calculate gradient estimate
        gk = (sharpe_plus - sharpe_minus) / (2 * ck * delta)

        # Update investment proportions
        p += ak * gk
        p = np.clip(p, 0, 1)  # Ensure non-negative investment proportions
        p /= np.sum(p)  # Ensure sum to 1

        # Record history
        p_history.append(p)
        sharpe_ratios_history.append(sharpe_plus)
        expected_returns_history.append(np.mean(np.dot(Y, p)) / np.std(np.dot(Y, p)))

        print(f"Iteration {k}: Sharpe Ratio = {sharpe_ratios_history[-1]:.4f}, Expected Return = {expected_returns_history[-1]:.4f}")

    return p, sharpe_ratios_history, expected_returns_history, p_history

# SPSA parameters
a = 0.5
c = 0.1
A = 1
alpha = 0.602
gamma = 0.101
iterations = 1000

# Run SPSA optimization
optimal_p, sharpe_ratios_history, expected_returns_history, p_history = spsa_optimize(p_initial, a, c, A, alpha, gamma, iterations, Y, x_thresholds)

# Plot Sharpe ratio and expected return changes
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(sharpe_ratios_history, label='Sharpe Ratio')
plt.xlabel('Iterations')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio History')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(expected_returns_history, label='Expected Return', color='orange')
plt.xlabel('Iterations')
plt.ylabel('Expected Return')
plt.title('Expected Return History')
plt.legend()

plt.tight_layout()
plt.show()

# Plot investment proportion changes
plt.figure(figsize=(8, 4))
for i in range(n_companies):
    plt.plot([p_history[j][i] for j in range(iterations)], label=f'p{i+1}')

plt.xlabel('Iterations')
plt.ylabel('Investment Proportion')
plt.title('Investment Proportion History')
plt.legend()
plt.show()

# Output optimal investment proportions
print("Optimal Investment Proportions:")
print(f"p1: {optimal_p[0]:.4f}")
print(f"p2: {optimal_p[1]:.4f}")
print(f"p3: {optimal_p[2]:.4f}")