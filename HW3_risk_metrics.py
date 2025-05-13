#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:39:33 2025

@author: astrohiro
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# === Risk Metric Calculation Function ===
def compute_var_cvar(mean, sigma, alpha):
    """
    Compute VaR, upper-tail CVaR, and lower-tail CVaR at level alpha for d ~ N(mean, sigma^2)
    """
    

    return var, cvar_upper, cvar_lower

# === Load and Prepare Trajectories & Obstacles ===
all_trajectories = np.load("data/all_trajectories.npy")  # Shape: (N_trajectories, K_timestep, state_dim)
all_costs = np.load("data/all_costs.npy")                # Shape: (N_trajectories,)
pos_obs = np.array([[5, 0], [5, 2.6], [5, -3]])           # Obstacle positions
r_obs = np.array([1, 1.3, 1.5])                           # Obstacle radii
N_trajectories = all_trajectories.shape[0]
N_obstacles = pos_obs.shape[0]
theta = np.linspace(0, 2 * np.pi, 100)                    # Circle plotting

# === Plot All Sampled Trajectories and Obstacles ===
colors = plt.cm.viridis(np.linspace(0, 1, N_trajectories))
plt.figure(figsize=(10, 6))
for n in range(N_trajectories):
    trj_n = all_trajectories[n, :, :]
    for i in range(N_obstacles):
        ox = pos_obs[i, 0]
        oy = pos_obs[i, 1]
        r = r_obs[i]
        plt.plot(ox + r * np.cos(theta), oy + r * np.sin(theta), 'k--')  # Dashed circle for obstacle
    plt.plot(trj_n[:, 0], trj_n[:, 1], color=colors[n])  # Plot trajectory

plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
#plt.title('Sampled Trajectories')
plt.grid(True)
plt.tight_layout()
plt.savefig("data/fig_sampled_trajectories.pdf")
plt.show()

# === Set Risk Parameters ===
eps_dmin = 0.0
dmins = r_obs + eps_dmin         # Safety distance thresholds
w_wrst = 0.6                     # Worst-case uncertainty shift
w_sigm = 0.2                     # Std dev of uncertainty
w_alph = 0.3                     # CVaR confidence level

# === Risk Evaluation for Each Trajectory ===
K_timestep = all_trajectories.shape[1]
trj_risks = np.zeros((N_trajectories, 3))  # Each column: [mean, worst-case, CVaR]