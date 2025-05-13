#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:39:33 2025

@author: astrohiro
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


# === Risk Metric Calculation Function ===
def compute_var_cvar(mean: float, sigma: float, alpha: float):
    var = norm.ppf(alpha, loc=mean, scale=sigma)
    phi = norm.pdf(norm.ppf(alpha))
    cvar_lower = mean - sigma * (phi / alpha)
    cvar_upper = mean + sigma * (phi / (1 - alpha))
    return var, cvar_upper, cvar_lower


def compute_mean_risk(
    d_nominal: NDArray[np.float64],
    all_costs: NDArray[np.float64],
    mu: float,
    sigma: float,
    r_obs: NDArray[np.float64],
) -> np.intp:
    """
    Compute the mean risk metric for each trajectory.

    Parameters
    ----------
    d_nominal : NDArray[np.float64]
        Nominal distance for each trajectory.
        The shape is (N_trajectories, K_timestep, N_obstacles).
    all_costs : NDArray[np.float64]
        Costs for each trajectory.
        The shape is (N_trajectories,).
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution.
    r_obs : NDArray[np.float64]
        Radii of the obstacles.
        The shape is (N_obstacles,).

    Returns
    -------
    optimal_index : int
        The index of the trajectory with the minimum risk.
    """
    N_traj, K, N_obs = d_nominal.shape
    risks = np.zeros((N_traj,))
    omega = np.random.normal(loc=mu, scale=sigma, size=(N_traj, K))
    d_perturbed = d_nominal + omega[:, :, np.newaxis]  # Shape: (N_traj, K, N_obs)

    for n in range(N_traj):
        risk = 0
        for k in range(K):
            for i in range(N_obs):
                if np.mean(d_perturbed[n, k, i]) < r_obs[i]:
                    risk += 1
        risks[n] = risk + all_costs[n]
    return np.argmin(risks)


def compute_worst_case_risk(
    d_nominal: NDArray[np.float64],
    all_costs: NDArray[np.float64],
    r_obs: NDArray[np.float64],
    omega_bar: float,
) -> np.intp:
    """
    Compute the worst-case risk metric for each trajectory.

    Parameters
    ----------
    d_nominal : NDArray[np.float64]
        Nominal distance for each trajectory. Shape: (N_trajectories, K_timestep, N_obstacles)
    all_costs : NDArray[np.float64]
        Costs for each trajectory. Shape: (N_trajectories,)
    r_obs : NDArray[np.float64]
        Radii of the obstacles. Shape: (N_obstacles,)
    omega_bar : float
        Maximum worst-case disturbance.

    Returns
    -------
    optimal_index : int
        The index of the trajectory with the minimum risk.
    """
    N_traj, K, N_obs = d_nominal.shape
    risks = np.zeros(N_traj)

    for n in range(N_traj):
        risk = 0
        for k in range(K):
            for i in range(N_obs):
                if d_nominal[n, k, i] - omega_bar < r_obs[i]:
                    risk += 1
        risks[n] = risk + all_costs[n]

    return np.argmin(risks + all_costs)


def compute_cvar_risk(
    d_nominal: NDArray[np.float64],
    all_costs: NDArray[np.float64],
    mu: float,
    sigma: float,
    alpha: float,
    r_obs: NDArray[np.float64],
) -> np.intp:
    """
    Compute the CVaR (expected shortfall) risk metric for each trajectory.

    Parameters
    ----------
    d_nominal : NDArray[np.float64]
        Nominal distance for each trajectory. Shape: (N_trajectories, K_timestep, N_obstacles)
    all_costs : NDArray[np.float64]
        Costs for each trajectory. Shape: (N_trajectories,)
    mu : float
        Mean of noise (typically 0).
    sigma : float
        Standard deviation of noise.
    alpha : float
        CVaR confidence level.
    r_obs : NDArray[np.float64]
        Radii of the obstacles. Shape: (N_obstacles,)

    Returns
    -------
    risks : NDArray[np.float64]
        CVaR-based risk metric for each trajectory. Shape: (N_trajectories,)
    """
    N_traj, K, N_obs = d_nominal.shape
    risks = np.zeros(N_traj)

    for n in range(N_traj):
        risk = 0
        for k in range(K):
            for i in range(N_obs):
                var, _, cvar_lower = compute_var_cvar(
                    d_nominal[n, k, i] + mu, 0.2, alpha
                )
                # print(f"var: {var}, cvar_lower: {cvar_lower}")
                if cvar_lower < r_obs[i]:
                    risk += 1
        risks[n] = risk + all_costs[n]

    return np.argmin(risks + all_costs)


# === Load and Prepare Trajectories & Obstacles ===
all_trajectories = np.load(
    "data/all_trajectories.npy"
)  # Shape: (N_trajectories, K_timestep, state_dim)
all_costs = np.load("data/all_costs.npy")  # Shape: (N_trajectories,)
pos_obs = np.array([[5, 0], [5, 2.6], [5, -3]])  # Obstacle positions
r_obs = np.array([1, 1.3, 1.5])  # Obstacle radii
N_trajectories = all_trajectories.shape[0]
N_obstacles = pos_obs.shape[0]
theta = np.linspace(0, 2 * np.pi, 100)  # Circle plotting

# # === Plot All Sampled Trajectories and Obstacles ===
# colors = plt.cm.viridis(np.linspace(0, 1, N_trajectories))
# plt.figure(figsize=(10, 6))
# for n in range(N_trajectories):
#     trj_n = all_trajectories[n, :, :]
#     for i in range(N_obstacles):
#         ox = pos_obs[i, 0]
#         oy = pos_obs[i, 1]
#         r = r_obs[i]
#         plt.plot(
#             ox + r * np.cos(theta), oy + r * np.sin(theta), "k--"
#         )  # Dashed circle for obstacle
#     plt.plot(trj_n[:, 0], trj_n[:, 1], color=colors[n])  # Plot trajectory

# plt.gca().set_aspect("equal")
# plt.xlabel("x")
# plt.ylabel("y")
# # plt.title('Sampled Trajectories')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("data/fig_sampled_trajectories.pdf")
# plt.show()

# === Set Risk Parameters ===
eps_dmin = 0.0
dmins = r_obs + eps_dmin  # Safety distance thresholds
w_wrst = 0.6  # Worst-case uncertainty shift
w_sigm = 0.2  # Std dev of uncertainty
w_alph = 0.3  # CVaR confidence level

# === Risk Evaluation for Each Trajectory ===
K_timestep = all_trajectories.shape[1]
trj_risks = np.zeros((N_trajectories, 3))  # Each column: [mean, worst-case, CVaR]

# Calculatae the nominal distance for each of the 3 obstacles
d_nominal = np.zeros((N_trajectories, K_timestep, 3))
for i in range(N_obstacles):
    d_nominal[:, :, i] = np.linalg.norm(
        all_trajectories[:, :, :2] - pos_obs[i, :], axis=2
    )

n_star_mean = compute_mean_risk(d_nominal, all_costs, mu=0.0, sigma=w_sigm, r_obs=r_obs)
n_star_worst = compute_worst_case_risk(
    d_nominal, all_costs, r_obs=r_obs, omega_bar=w_wrst
)
n_star_cvar = compute_cvar_risk(
    d_nominal, all_costs, mu=0.0, sigma=w_sigm, alpha=w_alph, r_obs=r_obs
)

n_stars: list[np.intp] = [n_star_mean, n_star_worst, n_star_cvar]
print(
    f"Optimal trajectory indices: Mean: {n_star_mean}, Worst-case: {n_star_worst}, CVaR: {n_star_cvar}"
)

labels: list[str] = ["Mean", "Worst-case", "CVaR"]

# === Plot Plot Optimal Trajectories ===
colors = plt.cm.viridis(np.linspace(0, 1, N_trajectories))
plt.figure(figsize=(10, 6))
for i in range(N_obstacles):
    ox = pos_obs[i, 0]
    oy = pos_obs[i, 1]
    r = r_obs[i]
    plt.plot(
        ox + r * np.cos(theta), oy + r * np.sin(theta), "k--"
    )  # Dashed circle for obstacle
for n, label in zip(n_stars, labels):
    trj_n = all_trajectories[n, :, :]
    plt.plot(trj_n[:, 0], trj_n[:, 1], color=colors[n], label=label)  # Plot trajectory

plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# plt.title('Sampled Trajectories')
plt.grid(True)
plt.tight_layout()
plt.savefig("data/fig_optimal_trajectories.png", dpi=600)
# plt.show()
