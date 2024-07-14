import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define parameters
N = 10  # Increase the number of customers to better estimate statistical properties
z = 8  # Target 90% quantile waiting time
mu_A = 1 / 5  # Mean interarrival time
theta0 = 1.0  # Initial mean service time
K = 3000  # Increase the number of iterations

# Step size and perturbation decay functions
def epsilon(k):
    return 0.09 / (k + 1) ** 0.5

def eta(k):
    return 0.5 / (k + 1) ** 0.101

# Function to run a single trajectory
def run_trajectory(A, S):
    T = len(A)
    W = np.zeros(T)
    W[0] = S[0]
    for t in range(1, T):
        W[t] = max(W[t - 1] + S[t] - A[t], 0)
    return W

# Function to calculate mean waiting time and 90% quantile
def calculate_wait_times(theta, A):
    S = np.random.exponential(scale=1 / theta, size=len(A))
    W = run_trajectory(A, S)
    return np.mean(W), np.percentile(W, 90)

# Objective function
def J(theta, A, z):
    mean_wait, q90_wait = calculate_wait_times(theta, A)
    return (q90_wait - z) ** 2

# SPSA algorithm implementation
def SPSA(mu_A, theta0, K, epsilon, eta, z):
    theta_history = [theta0]
    for k in tqdm(range(1, K), desc="SPSA Iteration"):
        delta = np.random.choice([-1, 1], p=[0.5, 0.5])
        theta_plus = max(theta_history[-1] + eta(k) * delta, 1e-3)
        theta_minus = max(theta_history[-1] - eta(k) * delta, 1e-3)

        A_sample = np.random.exponential(scale=mu_A, size=N)
        J_plus = J(theta_plus, A_sample, z)
        J_minus = J(theta_minus, A_sample, z)

        gradient_estimate = (J_plus - J_minus) / (2 * eta(k) * delta)
        theta_history.append(max(min(theta_history[-1] - epsilon(k) * gradient_estimate, 10.0), 1e-3))

    return np.array(theta_history)

# Main program
if __name__ == '__main__':
    theta_history = SPSA(mu_A, theta0, K, epsilon, eta, z)
    A_sample = np.random.exponential(scale=mu_A, size=N)  # Used to calculate the optimal theta
    theta_opt = theta_history[np.argmin([J(theta, A_sample, z) for theta in theta_history])]

    print('Optimal service time mean:', theta_opt)

    plt.figure(figsize=(10, 6))
    plt.plot(theta_history, label='Service Time Mean')
    plt.axhline(y=theta_opt, linestyle='--', color='r', label='Optimal Mean')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Service Time')
    plt.title('Optimization of Mean Service Time using SPSA')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate the waiting time distribution using the optimal mean service time theta_opt
    S_sample = np.random.exponential(scale=1 / theta_opt, size=len(A_sample))
    W_sample = run_trajectory(A_sample, S_sample)
    
    plt.figure(figsize=(10, 6))
    plt.hist(W_sample, bins=30, alpha=0.75, color='blue', label='Waiting Time Distribution')
    plt.axvline(x=z, color='r', linestyle='--', label='90% Quantile Target')
    plt.xlabel('Waiting Time')
    plt.ylabel('Frequency')
    plt.title('Distribution of Waiting Times')
    plt.legend()
    plt.grid(True)
    plt.show()