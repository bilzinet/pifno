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

# %%
# =============================================================================
# Helper functions for processing data
# =============================================================================

def generate_trajectory(x0, t0, v_map, dx=20, dt=1):
    """Generate vehicle trajectory from an initial state
    """
    time_indices = []
    space_indices = []
    x0_dist = x0*dx
    t0_dist = t0*dt
    while (x0 < v_map.shape[-1]) and (t0 < v_map.shape[-2]):
        time_indices.append(t0)
        space_indices.append(x0)
        x1_dist = x0_dist + v_map[t0, x0]*(5/18)*dt
        t1_dist = t0_dist + dt
        x1 = np.round(x1_dist/dx).astype(np.int32)
        t1 = np.round(t1_dist/dt).astype(np.int32)
        x0_dist = x1_dist
        t0_dist = t1_dist
        x0 = x1
        t0 = t1
    return np.array([time_indices, space_indices])

def create_trajectory_measurements(k_map, num_trajectories):
    """Create random vehicle trajectories as measurements
    """
    # convert to speed field
    def speed_field(k): return v_free * (1- k / k_jam)
    v_map = speed_field(k_map)
    m_map = np.full_like(k_map, -1)
    
    # initial density
    m_map[:,0,:] = k_map[:,0,:]
    # m_map[:,:,0] = k_map[:,:,0]
    # m_map[:,:,-1] = k_map[:,:,1]
    
    # density along sample trajectories
    for sam in range(v_map.shape[0]):
        for t in range(num_trajectories):
            if np.random.uniform() < 0.25:
                x0 = np.random.randint(0, M-1)
                t0 = 0
            else:
                x0 = 0
                t0 = np.random.randint(0, N-1)
            indx_rand = generate_trajectory(x0, t0, v_map[sam])
            m_map[sam,indx_rand[0,:], indx_rand[1,:]] = (k_map[sam,indx_rand[0,:],indx_rand[1,:]]).copy()
    
    return m_map

def load_data(filenames, datafold, num_samples=10, num_trajectories=50):
    """Load sample data
    """
    U_true = []
    for file in filenames:
        with open(datafold+f'{file}.pkl','rb') as f:
            data = pkl.load(f)
        # U_inps.append(data['X'][:num_samples])
        U_true.append(data['Y'][:num_samples])
    U_true = np.concatenate(U_true, axis=0)
    U_meas = create_trajectory_measurements(U_true, num_trajectories)
    return (U_true, U_meas)


# %%
# =============================================================================
# Extended Kalman Filter
# =============================================================================

def EKF_init(U_true):
    """Initialize EKF algorithm
    """
    M, N = U_true.shape[0], U_true.shape[1] 
    # k_entry = U_true[:, 0]
    # k_exit = U_true[:, -1]
    
    u0 = U_true[0, :].copy()
    P0 = init_P(u0)
    U_est = np.zeros((M, N))
    U_est[0, :] = u0 
    
    return u0, P0, U_est

def EKF_step(u_kprev, P_kprev, y_k, H_k):
    """Single recursive EKF run (single time step)
    """
    N = u_kprev.shape[0]
    M = y_k.shape[0]
    
    # Step 1: Prediction step (a-priori)
    u_kprior = model_transition(u_kprev)
    A_k = get_A(u_kprev)
    Q_k = get_Q(u_kprior)
    P_kprior = A_k.dot(P_kprev).dot(A_k.T) + Q_k
    
    # Step 2: Compute Kalman gain
    R_k = get_R(y_k)
    B_k = (H_k.dot(P_kprior)).dot(H_k.T) + R_k
    C_k = np.linalg.inv(B_k)
    K_k = (P_kprior.dot(H_k.T)).dot(C_k)
    
    # Step 3: Correction step (a-posteriori)
    I = np.eye(N)
    u_k = u_kprior + K_k.dot(y_k - H_k.dot(u_kprior))
    P_k = (I - K_k.dot(H_k)).dot(P_kprior)
        
    return u_k, P_k

def EKF_algorithm(U_true, U_meas, N=600):
    """Multiple recurivse EKF runs (multiple time steps)
    """
    # initialize EKF
    u_0, P_0, U_est = EKF_init(U_true)
    # run EKF
    u_kprev = u_0.copy()
    P_kprev = P_0.copy()
    for k in range(1, N):
        # get measurements
        u_meas = U_meas[k]
        y_k, H_k = get_H(u_meas)
        a, b = k_entry[k], k_exit[k]
        # forward step
        u_k, P_k = EKF_step(u_kprev, P_kprev, y_k, H_k)
        u_k = np.where(u_k > k_jam, k_jam, u_k)
        u_k = np.where(u_k < 0, 0, u_k)
        U_est[k] = u_k
        # update
        u_kprev = u_k
        P_kprev = P_k
    return U_est