"""
Created on Sun Nov 15 2023
@author: Bilal Thonnam Thodi (btt1@nyu.edu)

Classical traffic state estimation problem using data-assimilation technique:
    Traffic flow model: Lighthill-Withams-Richards
    Transition model (process model): Discrete Godunov numerical scheme
    Assimilation method: Extended Kalman Filter
    Measurement type: Initial conditions and vehicle trajectory measurements 
    Reference paper: Localized Extended Kalman Filter for Scalable Real-Time Traffic State Estimation
        (https://ieeexplore.ieee.org/abstract/document/6105572/)

"""

# %%
# =============================================================================
# Packages and settings
# =============================================================================

import timeit
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

plt.rc("font", family="serif")
plt.rc("xtick", labelsize="large")
plt.rc("ytick", labelsize="large")
np.set_printoptions(precision=4, suppress=True)


# %%
# =============================================================================
# Some helper functions
# =============================================================================

def fd(k, *args, **kwargs):
    """Greenshield's flux function
    """
    return k * v_free * (1 - k / k_jam)

def demand_fn(k, *args, **kwargs):
    """Traffic supply function
    """
    return fd(k) if k < k_cr else q_max

def supply_fn(k, *args, **kwargs):
    """Traffic demand function
    """
    return fd(k) if k > k_cr else q_max

def boundary_flows(k_up, k_dn, *args, **kwargs):
    """Calculate flux across a cell boundary as minimum 
        of upstream demand and downstream supply
    """
    return min(demand_fn(k_up), supply_fn(k_dn))

def model_transition(u_kprev, *args, **kwargs):
    """Model transition function (process model):
        Update previous density dyanmics as Godunov numerical scheme
    """
    M = u_kprev.shape[0]
    upriori_k = np.zeros(M)

    for m in range(M):
        # get computational stencil
        k_m_n = u_kprev[m]
        if m == 0:  # upstream boundary cell
            k_nextm_n = u_kprev[m + 1]
            k_prevm_n = k_entry[m]
        elif m == M - 1:  # downstream boundary cell
            k_prevm_n = u_kprev[m - 1]
            k_nextm_n = k_exit[m]
        else:  # other cells
            k_prevm_n = u_kprev[m - 1]
            k_nextm_n = u_kprev[m + 1]

        # update density
        q_in = boundary_flows(k_prevm_n, k_m_n)
        q_ou = boundary_flows(k_m_n, k_nextm_n)
        upriori_k[m] = k_m_n - (delt / delx) * (q_ou - q_in)

    return upriori_k

def gradient_fd(k, *args, **kwargs):
    """First order derivative of fd function
    """
    return v_free - 2 * k * (v_free / k_jam)

def gradient_qin(k_up, k_dn):
    if demand_fn(k_up) < supply_fn(k_dn):
        grad = 0
    else:
        grad = gradient_fd(k_dn) if k_dn > k_cr else 0
    return grad

def gradient_qou(k_up, k_dn):
    if demand_fn(k_up) < supply_fn(k_dn):
        grad = gradient_fd(k_up) if k_up < k_cr else 0
    else:
        grad = 0
    return grad
    
def gradient_qin_prevx(k_up, k_dn):
    if demand_fn(k_up) < supply_fn(k_dn):
        grad = gradient_fd(k_up) if k_up < k_cr else 0
    else:
        grad = 0
    return grad
    
def gradient_qou_nextx(k_up, k_dn):
    if demand_fn(k_up) < supply_fn(k_dn):
        grad = 0
    else:
        grad = gradient_fd(k_dn) if k_dn > k_cr else 0
    return grad

def get_A(u):
    """Build Jacobian of model w.r. to current state
    """
    N = u.shape[0]
    A = np.zeros((N, N))
    for j in range(N):
        k_j = u[j]
        
        if j == 0:
            k_nextj = u[j+1]
            grad_qou = gradient_qou(k_j, k_nextj)
            A[j,j] = 1 + (delt / delx) * - grad_qou
            A[j,j+1] = -(delt / delx) * gradient_qou_nextx(k_j, k_nextj)
        
        elif j == N-1:
            k_prevj = u[j-1]
            grad_qin = gradient_qin(k_prevj, k_j)
            A[j,j] = 1 + (delt / delx) * grad_qin
            A[j,j-1] = +(delt / delx) * gradient_qin_prevx(k_prevj, k_j)
        
        else:
            k_nextj = u[j+1]
            k_prevj = u[j-1]
            grad_qin = gradient_qin(k_prevj, k_j)
            grad_qou = gradient_qou(k_j, k_nextj)
            A[j,j] = 1 + (delt / delx) * (grad_qin - grad_qou)
            A[j,j-1] = +(delt / delx) * gradient_qin_prevx(k_prevj, k_j)
            A[j,j+1] = -(delt / delx) * gradient_qou_nextx(k_j, k_nextj)
    
    return A

def get_H(u_meas):
    """Measurement matrix
    """
    N = u_meas.shape[0]
    yloc = np.where(u_meas != -1)[0]
    M = yloc.shape[0]
    
    y = u_meas[yloc]
    H = np.zeros((M, N))
    for i in range(M):
        H[i][yloc[i]] = 1.
    
    return y, H

def get_Q(u):
    """Process noise covariance matrix (diagonal)
    """
    N = u.shape[0]
    Q = np.zeros((N, N))
    for i in range(N):
        Q[i,i] = 0.031275 * (u[i]**2)
    return Q

def get_R(y):
    """Measurement noise covariance matrix (diagonal)
    """
    N = y.shape[0]
    R = np.zeros((N, N))
    for i in range(N):
        R[i,i] = 0.015625 * (y[i]**2)
    return R

def init_P(u0, std_dev=2.50):
    """Initial error covariance matrix 
    """
    N = u0.shape[0]
    u0_noise = u0 #+ np.random.normal(0, std_dev, size=(N))
    # u0_noise = np.where(u0_noise > k_jam, k_jam, u0_noise)
    # u0_noise = np.where(u0_noise < 0, 0, u0_noise)
    
    P0 = np.zeros((N, N))
    for i in range(N):
        P0[i,i] = 0.00390625 * u0_noise[i]**2
    return P0
