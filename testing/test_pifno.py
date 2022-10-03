# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:02:11 2022
@author: btt1
Evaluation of PI-FNO model for LWR
"""

# %%
# =============================================================================
# Packages and Settings
# =============================================================================

# load packages
import numpy as np
from fno2d import *

import torch
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F
from torch.autograd import grad

from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# settings
np.set_printoptions(precision=4, suppress=True)
from matplotlib import rcParams
plt.rcParams["font.family"] = "Times New Roman"
rcParams['mathtext.fontset']='cm'
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')


# %%
# =============================================================================
# Helper functions
# =============================================================================

# extract vehicle trajectory
def genTraj(x0, t0, v_map, dx=20, dt=1):
    # initialize
    time_indices = []
    space_indices = []
    
    # run through speed map
    x0_dist = x0*dx
    t0_dist = t0*dt
    while (x0 < v_map.shape[-1]) and (t0 < v_map.shape[-2]):
        time_indices.append(t0)
        space_indices.append(x0)
        x1_dist = x0_dist + v_map[t0, x0]*(5/18)*dt
        t1_dist = t0_dist+dt
        x1 = np.round(x1_dist/dx).astype(np.int32)
        t1 = np.round(t1_dist/dt).astype(np.int32)
        x0_dist = x1_dist
        t0_dist = t1_dist
        x0 = x1
        t0 = t1
    return np.array([time_indices, space_indices])

# random masking of interior values
def rand_mask_interior(x_test, y_test, mp=0.05):
    # randomly sample 2D grid points
    nums = int(mp*x_test.shape[1]*x_test.shape[2])
    indx_rand = np.random.randint([1,1], [x_test.shape[1]-1,x_test.shape[2]-1], 
                                  size=(nums,2))
    # assign solutions at sampled grid points and keep rest as constant
    k_unkn = -1
    x_test[:,1:,:] = k_unkn
    x_test[:,indx_rand[:,0],indx_rand[:,1]] = (y_test[:,indx_rand[:,0],indx_rand[:,1]]).copy()
    return x_test

# random masking of interior values along vehicle trajectories
def rand_mask_interior_traj(x_test, y_test, traj_num=10):
    # convert to speed field
    v_max = 60
    k_max = 120
    k_unkn = -1
    def speed_field(k): return v_max*(1-k/k_max)
    v_map = speed_field(y_test)
    x_test[:,1:,:] = k_unkn
    
    # randomly sample trajectories
    for sam in range(v_map.shape[0]):
        for t in range(traj_num):
            if np.random.uniform() < 0.25:
                x0 = np.random.randint(0,49)
                t0 = 0
            else:
                x0 = 0
                t0 = np.random.randint(0,599)
            indx_rand = genTraj(x0, t0, v_map[sam], dx=20, dt=1)
            x_test[sam,indx_rand[0,:],indx_rand[1,:]] = (y_test[sam,indx_rand[0,:],indx_rand[1,:]]).copy()
    return x_test

def load_testdata(f_names, b_size, ntest_sc, **kwargs):
    
    # load test data
    x_test = []; y_test = []
    for f in f_names:
        with open(f'../0_Data/test/test_{f}.pkl', 'rb') as f:
            data = pkl.load(f)
        x_test.append(data['X'][:ntest_sc])
        y_test.append(data['Y'][:ntest_sc])
    x_test = np.concatenate(x_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.float32)
    x_test = rand_mask_interior_traj(x_test, y_test)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    
    # grid size params
    s1 = x_test.shape[1]
    s2 = x_test.shape[2]
    ntest = x_test.shape[0]
    
    # concat location coordinates
    grids = []
    grids.append((np.linspace(20,1000,s1)+np.linspace(0,1000-20,s1))/2)     # (np.linspace(20,1000,s1)+np.linspace(0,1000-20,s1))/2 np.linspace(0,1,s1)
    grids.append((np.linspace(1,600,s2)+np.linspace(0,600-1,s2))/2)         # np.linspace(0,1,s1) (np.linspace(1,600,s2)+np.linspace(0,600-1,s2))/2
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s1,s2,2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_test = torch.cat([x_test.reshape(ntest,s1,s2,1), 
                        grid.repeat(ntest,1,1,1)], dim=3)
    
    # pytorch loader
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), 
        batch_size=b_size, shuffle=False)
    
    return test_loader,y_test

def test(model, test_loader, y_test):
    
    index = 0
    pred = torch.zeros(y_test.shape)
    act = torch.zeros(y_test.shape)
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            pred[index] = out
            act[index] = y.squeeze(0)
            index = index + 1
    K_pred = pred.cpu().numpy()
    K_act = act.cpu().numpy()
    
    return K_act, K_pred 



# %%
# =============================================================================
# Out-of-sample performance: train-test input conditions
# =============================================================================

rin = np.random.randint
run = np.random.uniform
rno = np.random.normal

def gen_initial_condition(num_wavelets, avg_value=60, inp_stddev=10, num_points=50):
    
    i = 0
    par_length = int(num_points/num_wavelets)+5
    k_initial = np.repeat(rin(avg_value-inp_stddev,avg_value+inp_stddev), num_points)
    for t in range(num_wavelets):
        j = min(num_points-1, rin(i, i+max(5,par_length)))
        k_initial[j:] =  max(0, min(115, k_initial[j-1]+rin(-20,20)))
        i = j
    
    return k_initial

def gen_bound_condition(num_wavelets, num_points=600, avg_value=90, inp_stddev=1, par_sidewidth=10):
    
    k_bound = np.repeat(avg_value, num_points).astype(np.float64)
    k_bound += rno(0,inp_stddev,num_points)
    if num_wavelets > 0:
        par_len = int(k_bound.shape[0]/num_wavelets)-1
        for nw in range(num_wavelets):
            k_par = k_bound[nw*par_len:(nw+1)*par_len]
            k_par[rin(par_sidewidth,int(par_len)/3):rin(2*int(par_len)/3,par_len-par_sidewidth)] = 120
            k_bound[nw*par_len:(nw+1)*par_len] = k_par
    
    return k_bound
    

# visualize input conditions
fig, axs = plt.subplots(2, 8, figsize=(10,3.5))
fig.subplots_adjust(wspace=0.1, hspace=0.50)

# plot train-test initial conditions
num_trainconds = 3
num_wavelets_arr = [1,2,3,4,6,10,20,30]
x_values = np.arange(0, 50, 1)
for n, num_wavelets in enumerate(num_wavelets_arr):
    k_bound = gen_initial_condition(num_wavelets)
    if num_wavelets <= num_trainconds:
        if n == 0:
            l1, = axs[0,n].plot(x_values, k_bound, color='tab:blue')
        else:
            axs[0,n].plot(x_values, k_bound, color='tab:blue')
    else:
        if n == len(num_wavelets_arr)-1:
            l2, = axs[0,n].plot(x_values, k_bound, color='tab:orange')
        else:
            axs[0,n].plot(x_values, k_bound, color='tab:orange')
    axs[0,n].set_xticks([])
    axs[0,n].set_yticks([])
    axs[0,n].set_xlabel('$x$', fontsize=14)  
    axs[0,n].grid()
    axs[0,n].set_title(f'i$_{n+1}$', fontsize=15)
axs[0,0].set_ylabel(r'$\bar{u}_0 (x)$', fontsize=14)
    
# plot train-test boundary conditions
sam_rate = 5
num_trainconds = 2
num_wavelets_arr = [1,2,3,4,5,6,7,8]
x_values = np.arange(0,600,1)
for n,num_wavelets in enumerate(num_wavelets_arr):
    k_bound = gen_bound_condition(num_wavelets)
    if num_wavelets <= num_trainconds:
        axs[1,n].plot(x_values[::sam_rate], k_bound[::sam_rate], c='tab:blue')
    else:
        axs[1,n].plot(x_values[::sam_rate], k_bound[::sam_rate], c='tab:orange')
    axs[1,n].set_xticks([])
    axs[1,n].set_yticks([])
    axs[1,n].set_xlabel('$t$', fontsize=14)  
    axs[1,n].grid()
    axs[1,n].set_title(f'b$_{n+1}$', fontsize=15)
axs[1,0].set_ylabel(r'$\bar{u}_b (t)$', fontsize=14)

fig.legend(handles=[l1, l2], ncol=2,
           labels=['Train', 'Test'],
           loc='upper center', frameon=False,
           fontsize=14, borderaxespad=-0.2)
fig.savefig('./expt_inputconditions.png',
            bbox_inches='tight', pad_inches=0, dpi=300)



# %%
# =============================================================================
# Out-of-sample performance: forward problem
# =============================================================================

def get_error_metrics(K_mae):
    valid_mae = np.mean(np.append(K_mae.mean(axis=1)[:3], K_mae.mean(axis=1)[12:15]))
    valid_std = np.mean(np.append(K_mae.std(axis=1)[:3], K_mae.std(axis=1)[12:15]))
    test_mae = np.mean(np.append(K_mae.mean(axis=1)[3:9], K_mae.mean(axis=1)[15:22]))
    test_std = np.mean(np.append(K_mae.std(axis=1)[3:9], K_mae.std(axis=1)[15:22]))
    return valid_mae, valid_std/np.sqrt(50), test_mae, test_std/np.sqrt(50)

def piecewise_linearfn(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def fit_piecewise_linearfn(x, y, n):
    y_mod = y.copy();
    p, e = optimize.curve_fit(piecewise_linearfn, x, y_mod)
    xd = np.linspace(0,n,100); yd = piecewise_linearfn(xd, *p)
    return xd, yd, p[-2:]


# load test errors and test predictions
main_expt_fold = '../models/'
K_mae1 = np.load(main_expt_fold+'forward_kmae_results-fno.npy')
K_mae2 = np.load(main_expt_fold+'forward_kmae_results-pifno.npy')
K_comp1 = np.load(main_expt_fold+'forward_kcomp_results-fno.npy')
K_comp2 = np.load(main_expt_fold+'forward_kcomp_results-pifno.npy')
K_forw = (K_mae1,K_mae2,K_comp1,K_comp2)
fno_metrics = get_error_metrics(K_mae1)
pifno_metrics = get_error_metrics(K_mae2)

# out-of-sample performance with trendline
x1 = np.arange(0,9); y1 = K_mae1.mean(axis=1)[:9]
xd1, yd1, s1 = fit_piecewise_linearfn(x1, y1, n=9)
x2 = np.arange(0,9); y2 = K_mae2.mean(axis=1)[:9]
xd2, yd2, s2 = fit_piecewise_linearfn(x2, y2, n=9)
x3 = np.arange(0,10); y3 = K_mae1.mean(axis=1)[12:22]
xd3, yd3, s3 = fit_piecewise_linearfn(x3, y3, n=10)
x4 = np.arange(0,10); y4 = K_mae2.mean(axis=1)[12:22]
xd4, yd4, s4 = fit_piecewise_linearfn(x4, y4, n=10)
print('\n\nError rates: ', np.around([s1[-1],s2[-1],s3[-1],s4[-1]], 4))


# visualize out-of-sample error metrics
fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True, figsize=(5.25,2.5))
fig.subplots_adjust(wspace=0.05)

ax1_xlabs = ['i$_0$','i$_1$','i$_2$','i$_3$','i$_4$','i$_5$','i$_6$','i$_7$','i$_8$','i$_9$']
ax1.plot(xd3, yd3, c='tab:blue', ls='--', alpha=0.7)
ax1.errorbar(ax1_xlabs, K_mae1.mean(axis=1)[12:22], 
             K_mae1.std(axis=1)[12:22]/np.sqrt(50),
             ls='', marker='o', c='tab:blue', ms=5, mfc='w', label='FNO')
ax1.plot(xd4, yd4, c='tab:orange', ls='--', alpha=0.7)
ax1.errorbar(ax1_xlabs, K_mae2.mean(axis=1)[12:22], 
             K_mae2.std(axis=1)[12:22]/np.sqrt(50),
             ls='', marker='o', c='tab:orange', ms=5, mfc='w', label=r'$\pi$-FNO')
ax1.set_xlabel('Initial conditions', fontsize=14)
ax1.set_ylabel('MAE (vehs/km)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
# ax1.grid(True, which='major', axis='both', linestyle='-', alpha=0.4)
ax1.axvline(3.5, 0, 4.0, c='k', alpha=0.5, ls='-.')
ax1.text(0.00,3.30,r'$\leftarrow$ Train', c='k', alpha=1.0, fontsize=10)
ax1.text(3.75,3.30,r'Test $\rightarrow$', c='k', alpha=1.0, fontsize=10)
ax1.legend(fontsize=10)

ax2_xlabs = ['b$_0$','b$_1$','b$_2$','b$_3$','b$_4$','b$_5$','b$_6$','b$_7$','b$_8$']
ax2.plot(xd1, yd1, c='tab:blue', ls='--', alpha=0.7)
ax2.errorbar(ax2_xlabs, K_mae1.mean(axis=1)[:9], 
             K_mae1.std(axis=1)[:9]/np.sqrt(50),
             ls='', marker='o', c='tab:blue', ms=5, mfc='w', label='FNO')
ax2.plot(xd2, yd2, c='tab:orange', ls='--', alpha=0.7)
ax2.errorbar(ax2_xlabs, K_mae2.mean(axis=1)[:9], 
             K_mae2.std(axis=1)[:9]/np.sqrt(50),
             ls='', marker='o', c='tab:orange', ms=5, mfc='w', label=r'$\pi$-FNO')
ax2.set_xlabel('Boundary conditions', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
# ax2.grid(True, which='major', axis='both', linestyle='-', alpha=0.4)
ax2.axvline(2.5, 0, 4.0, c='k', alpha=0.5, ls='-.')
ax2.text(-0.50,3.30,r'$\leftarrow$ Train', c='k', alpha=1.0, fontsize=10)
ax2.text(2.75,3.30,r'Test $\rightarrow$', c='k', alpha=1.0, fontsize=10)
# ax2.legend(fontsize=10)

fig.suptitle('(a) Forward problem', fontsize=14)
fig.savefig('./res_outofsample_forward.png',
            bbox_inches='tight', pad_inches=0, dpi=300)



# %%
# =============================================================================
# Out-of-sample performance: inverse problem
# =============================================================================

# load test errors and test predictions
main_expt_fold = '../models/'
K_mae1 = np.load(main_expt_fold+'inverse_kmae_results-fno.npy')
K_mae2 = np.load(main_expt_fold+'inverse_kmae_results-pifno.npy')
K_comp1 = np.load(main_expt_fold+'inverse_kcomp_results-fno.npy')
K_comp2 = np.load(main_expt_fold+'inverse_kcomp_results-pifno.npy')
K_inve = (K_mae1,K_mae2,K_comp1,K_comp2)
fno_metrics = get_error_metrics(K_mae1)
pifno_metrics = get_error_metrics(K_mae2)

# compare testing errors with trendline
x1 = np.arange(0,9); y1 = K_mae1.mean(axis=1)[:9]
xd1, yd1, s1 = fit_piecewise_linearfn(x1, y1, n=9)
x2 = np.arange(0,9); y2 = K_mae2.mean(axis=1)[:9]
xd2, yd2, s2 = fit_piecewise_linearfn(x2, y2, n=9)
x3 = np.arange(0,10); y3 = K_mae1.mean(axis=1)[12:22]
xd3, yd3, s3 = fit_piecewise_linearfn(x3, y3, n=10)
x4 = np.arange(0,10); y4 = K_mae2.mean(axis=1)[12:22]
xd4, yd4, s4 = fit_piecewise_linearfn(x4, y4, n=10)
print('\n\nError rates: ', np.around([s1[-1],s2[-1],s3[-1],s4[-1]], 4))

# visualize
fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True, figsize=(5.25, 2.5))
fig.subplots_adjust(wspace=0.05)

ax1_xlabs = ['i$_0$','i$_1$','i$_2$','i$_3$','i$_4$','i$_5$','i$_6$','i$_7$','i$_8$','i$_9$']
ax1.plot(xd3, yd3, c='tab:blue', ls='--', alpha=0.7)
ax1.errorbar(ax1_xlabs, K_mae1.mean(axis=1)[12:22], 
             K_mae1.std(axis=1)[12:22]/np.sqrt(50),
             ls='', marker='o', ms=5, c='tab:blue', mfc='w', label='FNO')
ax1.plot(xd4, yd4, c='tab:orange', ls='--', alpha=0.7)
ax1.errorbar(ax1_xlabs, K_mae2.mean(axis=1)[12:22], 
             K_mae2.std(axis=1)[12:22]/np.sqrt(50),
             ls='', marker='o', ms=5, c='tab:orange', mfc='w', label=r'$\pi$-FNO')
ax1.set_xlabel('Initial conditions', fontsize=14)
ax1.set_ylabel('MAE (vehs/km)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
# ax1.grid(True, which='major', axis='both', linestyle='-', alpha=0.4)
ax1.axvline(3.5, 0, 4.0, c='k', alpha=0.5, ls='-.')
ax1.text(0.00,3.30,r'$\leftarrow$ Train', c='k', alpha=1.0, fontsize=10)
ax1.text(3.75,3.30,r'Test $\rightarrow$', c='k', alpha=1.0, fontsize=10)
ax1.legend(fontsize=10)

ax2_xlabs = ['b$_0$','b$_1$','b$_2$','b$_3$','b$_4$','b$_5$','b$_6$','b$_7$','b$_8$']
ax2.plot(xd1, yd1, c='tab:blue', ls='--', alpha=0.7)
ax2.errorbar(ax2_xlabs, K_mae1.mean(axis=1)[:9], 
             K_mae1.std(axis=1)[:9]/np.sqrt(50),
             ls='', marker='o', ms=5, c='tab:blue', mfc='w', label='FNO')
ax2.plot(xd2, yd2, c='tab:orange', ls='--', alpha=0.7)
ax2.errorbar(ax2_xlabs, K_mae2.mean(axis=1)[:9], 
             K_mae2.std(axis=1)[:9]/np.sqrt(50),
             ls='', marker='o', ms=5, c='tab:orange', mfc='w', label=r'$\pi$-FNO')
ax2.set_xlabel('Boundary conditions', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
# ax2.grid(True, which='major', axis='both', linestyle='-', alpha=0.4)
ax2.axvline(2.5, 0, 4.0, c='k', alpha=0.5, ls='-.')
ax2.text(-0.50,3.30,r'$\leftarrow$ Train', c='k', alpha=1.0, fontsize=10)
ax2.text(2.75,3.30,r'Test $\rightarrow$', c='k', alpha=1.0, fontsize=10)
# ax2.legend(fontsize=10)

fig.suptitle('(b) Inverse problem', fontsize=14)
fig.savefig('./res_outofsample_inverse.png',
            bbox_inches='tight', pad_inches=0, dpi=300)


# =============================================================================
# Sample predictions: visualize heatmaps
# =============================================================================

def plot_hmap_solutions(K_maps_arr, idx_arr=None, show_cbar=True, show_inps=False, t_max_arr=[600,600,600,600]):
    
    labels = ['True',r'$\pi$-FNO']
    sec_labels = ['(a) Forward (Train)','(b) Forward (Test)','(c) Inverse (Test)','(d) Inverse (Test)']
    x_max=1; k_jam=120; dt = 1/3600; dx = 20/1000
    
    fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(8.0,2.80))
    fig.subplots_adjust(hspace=0.25, wspace=0.00)
    
    for l, K_maps in enumerate(K_maps_arr):
        sam = idx_arr[l]
        for k, kmap in enumerate(K_maps[:2]):
            hmap = axs[k,l].imshow(kmap[sam].T, cmap='rainbow', extent=[0, t_max_arr[l], 0, x_max],
                                 origin='lower', aspect='auto', vmin=0, vmax=k_jam)
            
            if k == 1:
                axs[k,l].set_xlabel('$t$ (hrs)', fontsize=10)
            if l == 0:
                axs[k,l].set_ylabel('$x$ (km)', fontsize=10)
            axs[k,l].tick_params(axis='both', which='major', labelsize=10)
            if k == 0:
                axs[k,l].set_title(f'{sec_labels[l]} \n {labels[k]}', fontsize=10)
            else:
                axs[k,l].set_title(f'{labels[k]}', fontsize=10)
            axs[k,l].set_frame_on(False)
            
            if ((l == 2 and k==0) or (l==3 and k==0)  and show_inps):
                input_locs = np.argwhere(K_maps[-1][sam,]!=-1)
                axs[k,l].scatter(input_locs[:,0]*dt, input_locs[:,1]*dx, c='k', s=0.010)
            
            if show_cbar:
                divider = make_axes_locatable(axs[k,l])
                cax = divider.append_axes("right", size="3%", pad=0.05)
                if k==0 and l==3:
                    cbar = fig.colorbar(hmap, cax=cax)
                    cbar.set_label(r'$u$ (vehs/km)', fontsize=10, rotation=90)
                    cax.set_frame_on(False)
                else:
                    cax.axis('off')
    
    return fig, axs



sams_arr = []
K_maps_arr = []
t_max_arr = []

e=2
K_mae1, K_mae2, K_comp1, K_comp2 = K_forw
K_act, K_pred1, K_pred2 = K_comp1[e,0,...], K_comp1[e,1,...], K_comp2[e,1,...]
K_maps = (K_act, K_pred2)
sam = np.argmin(K_mae2[e])
K_maps_arr.append(K_maps)
sams_arr.append(sam)
t_max_arr.append(600)

e=5
K_mae1, K_mae2, K_comp1, K_comp2 = K_forw
K_act, K_pred1, K_pred2 = K_comp1[e,0,...], K_comp1[e,1,...], K_comp2[e,1,...]
K_maps = (K_act, K_pred2)
sam = np.argmin(K_mae2[e])
K_maps_arr.append(K_maps)
sams_arr.append(sam)
t_max_arr.append(600)

e=4
K_mae1, K_mae2, K_comp1, K_comp2 = K_inve 
K_act, K_pred1, K_pred2, K_inps = K_comp1[e,0,...], K_comp1[e,1,...], K_comp2[e,1,...], K_comp1[e,2,...]
K_maps = (K_act, K_pred2, K_inps)
sam = np.argmin(K_mae2[e])
K_maps_arr.append(K_maps)
sams_arr.append(sam)
t_max_arr.append(600)

e=7
_, K_mae2, K_comp1, K_comp2 = K_inve
K_act, K_pred1, K_pred2, K_inps = K_comp1[e,0,...], K_comp1[e,1,...], K_comp2[e,1,...], K_comp1[e,2,...]
K_maps = (K_act, K_pred2, K_inps)
sam = np.argmin(K_mae2[e])
K_maps_arr.append(K_maps)
sams_arr.append(sam)
t_max_arr.append(600)

K_maps_arr = tuple(K_maps_arr)
t_max_arr = np.array(t_max_arr)/3600
fig, ax = plot_hmap_solutions(K_maps_arr, idx_arr=sams_arr, show_cbar=True, show_inps=True, t_max_arr=t_max_arr)
fig.savefig('./res_sample_preds.png',
            bbox_inches='tight', pad_inches=0, dpi=300)



# =============================================================================
# Physics informing: Shock behavior
# =============================================================================


# plot density profiles along t-dimension
case = -4
sam = 45  # 11,25
K_mae1, K_mae2, K_comp1, K_comp2 = K_forw

t_secs = [0, 1, 2]
x_values = np.arange(0,1000,20)
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(6,2.5))
fig.subplots_adjust(wspace=0.10)
for k in range(3):
    axs[0,k].plot(x_values, K_comp1[case, 0, sam, t_secs[k], :], c='tab:blue', lw=2, label='True')
    axs[0,k].plot(x_values, K_comp1[case, 1, sam, t_secs[k], :], c='tab:orange', lw=1.5, label='FNO')
    axs[1,k].plot(x_values, K_comp2[case, 0, sam, t_secs[k], :], c='tab:blue', lw=2,  label='True')
    axs[1,k].plot(x_values, K_comp2[case, 1, sam, t_secs[k], :], c='tab:orange', lw=1.5, label=r'$\pi$-FNO')
    if k == 1:
        axs[0,k].set_title('(a) Example 1 \n $t=${} (s)'.format(t_secs[k]), fontsize=13)
    else:
        axs[0,k].set_title('$t=${} (s)'.format(t_secs[k]), fontsize=13)
    axs[1,k].set_xlabel('$x$ (m)', fontsize=13)
    axs[1,k].set_xticks([0,250,400,500,750,1000])
    axs[1,k].set_yticks([0,40,80,90,120])
    axs[0,k].grid(alpha=0.4)
    axs[1,k].grid(alpha=0.4)
    if k == 0:
        axs[0,k].legend(fontsize=8)
        axs[1,k].legend(fontsize=8)
        axs[0,k].set_ylabel(r'$u$ (vehs/km)', fontsize=13)
        axs[1,k].set_ylabel(r'$u$ (vehs/km)', fontsize=13)
    axs[0,k].tick_params(axis='both', which='major', labelsize=13)
    axs[1,k].tick_params(axis='both', which='major', labelsize=13)
    axs[0,k].set_xlim([200,600])
    axs[1,k].set_xlim([300,600])
    axs[0,k].set_ylim([70,100])
    axs[1,k].set_ylim([70,100])
fig.savefig('./res_tsec1.png',
            bbox_inches='tight', pad_inches=0, dpi=300)


case = -4
sam = 0  # 45
K_mae1, K_mae2, K_comp1, K_comp2 = K_forw

t_secs = [0, 1, 2]
x_values = np.arange(0,1000,20)
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(6,2.50))
fig.subplots_adjust(wspace=0.10)

for k in range(3):
    axs[0,k].plot(x_values, K_comp1[case, 0, sam, t_secs[k], :], c='tab:blue', lw=2, label='True')
    axs[0,k].plot(x_values, K_comp1[case, 1, sam, t_secs[k], :], c='tab:orange', lw=1.5, label='FNO')
    axs[1,k].plot(x_values, K_comp2[case, 0, sam, t_secs[k], :], c='tab:blue', lw=2,  label='True')
    axs[1,k].plot(x_values, K_comp2[case, 1, sam, t_secs[k], :], c='tab:orange', lw=1.5, label=r'$\pi$-FNO')
    if k == 1:
        axs[0,k].set_title('(b) Example 2 \n $t=${} (s)'.format(t_secs[k]), fontsize=13)
    else:
        axs[0,k].set_title('$t=${} (s)'.format(t_secs[k]), fontsize=13)
    axs[1,k].set_xlabel('$x$ (m)', fontsize=13)
    axs[1,k].set_xticks([0,250,400,500,750,1000])
    axs[1,k].set_yticks([0,40,80,120])
    axs[0,k].grid(alpha=0.4)
    axs[1,k].grid(alpha=0.4)
    if k == 0:
        axs[0,k].legend(fontsize=8)
        axs[1,k].legend(fontsize=8)
        axs[0,k].set_ylabel(r'$u$ (vehs/km)', fontsize=13)
        axs[1,k].set_ylabel(r'$u$ (vehs/km)', fontsize=13)
    axs[0,k].tick_params(axis='both', which='major', labelsize=13)
    axs[1,k].tick_params(axis='both', which='major', labelsize=13)
    axs[0,k].set_xlim([200,600])
    axs[1,k].set_xlim([300,600])
    axs[0,k].set_ylim([30,90])
    axs[1,k].set_ylim([30,90])
fig.savefig('./res_tsec2.png',
            bbox_inches='tight', pad_inches=0, dpi=300)

# =============================================================================
# 
# =============================================================================









