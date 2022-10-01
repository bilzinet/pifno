"""
Created: April 2021
@author: Bilal Thonnam Thodi (btt1@nyu.edu)

Numerical Solution for LWR model of traffic flow
Godunov/Minium Supply Denand Scheme (Finite Volume)
"""

# =============================================================================
# Import packages
# =============================================================================

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# Some useful functions
# =============================================================================

def FundDiag(k, k_max, v_max, k_cr, fd):
    '''
    Fundamenal Diagram
    '''
    if fd=='Greenshield':
        q = k*v_max*(1-k/k_max)
    elif fd=='Triangular':
        if k <= k_cr:
            q = v_max*k
        else:
            w = k_cr*v_max/(k_max-k_cr)
            q = w*(k_max-k)
    elif fd=='NewellFrank':
        q = k*v_max*(1 - np.exp((v_b/v_max)*(k_max/k - 1)))
    return q

def Demandfn(k, k_max, v_max, k_cr, q_max, fd):
    '''
    Traffic supply function
    '''
    if fd=='Greenshield':
        if k <= k_cr:
            q = k*v_max*(1-k/k_max)
        else:
            q = q_max
    elif fd=='Triangular':
        q = min(v_max*k, q_max)
    elif fd=='NewellFrank':
        if k <= k_cr:
            v = v_max*(1 - np.exp((v_b/v_max)*(k_max/k - 1)))
            q = k*v
        else:
            q = q_max
    return q

def InvDemandfn_num(q, dem_fn, k_arr):
    qb = dem_fn[dem_fn < q][-1]
    qa = dem_fn[dem_fn >= q][0]
    kb = k_arr[dem_fn < q][-1]
    ka = k_arr[dem_fn >= q][0]
    k = kb + (ka-kb)*(q-qb)/(qa-qb)
    return k

def InvDemandfn(q, k_max, v_max, k_cr, q_max, fd):
    '''
    Inverse of traffic supply function
    '''
    if fd=='Greenshield':
        q = min(q, q_max)
        k = (k_max-np.sqrt(k_max**2-4*k_max/v_free*q))/2
    elif fd=='Triangular':
        k = min(q/v_max, k_cr)
    elif fd =='NewellFrank':
        q = min(q, q_max)
        k = InvDemandfn_num(q, dem_fn, k_arr)
    return k
            
def Supplyfn(k, k_max, v_max, k_cr, q_max, fd):
    '''
    Traffic demand function
    '''
    if fd=='Greenshield':
        if k >= k_cr:    
            q = k*v_max*(1-k/k_max)
        else:
            q = q_max
    elif fd=='Triangular':
        w = k_cr*v_max/(k_max-k_cr)
        q = min(q_max, w*(k_max-k))
    elif fd=='NewellFrank':
        if k >= k_cr:
            v = v_max*(1 - np.exp((v_b/v_max)*(k_max/k - 1)))
            q = k*v
        else:
            q = q_max
    return q

def InvSupplyfn_num(q, sup_fn, k_arr):
    qb = sup_fn[sup_fn <= q][0]
    qa = sup_fn[sup_fn > q][-1]
    kb = k_arr[sup_fn <= q][0]
    ka = k_arr[sup_fn > q][-1]
    k = kb + (ka-kb)*(q-qb)/(qa-qb)
    return k

def InvSupplyfn(q, k_max, v_max, k_cr, q_max, fd):
    '''
    Inverse of traffic demand function
    '''
    if fd=='Greenshield':
        q = min(q, q_max)
        k = (k_max+np.sqrt(k_max**2-4*k_max/v_free*q))/2
    elif fd=='Triangular':
        w = k_cr*v_max/(k_max-k_cr)
        k = min(q_max, k_max-q/w)
    elif fd=='NewellFrank':
        q = min(q, q_max)
        k = InvSupplyfn_num(q, sup_fn, k_arr)
    return k

def bound_cond_entry(k_prev, q_en, k_max, v_max, k_cr, q_max, fd):
    q_en = min(q_en, q_max)
    supply = Supplyfn(k_prev, k_max, v_max, k_cr, q_max, fd)
    if q_en <= supply:
        k = InvDemandfn(q_en, k_max, v_max, k_cr, q_max, fd)
    else:
        k = InvSupplyfn(q_en, k_max, v_max, k_cr, q_max, fd)
        
    return k

def bound_cond_exit(k_prev, q_ex, k_max, v_max, k_cr, q_max, fd):
    q_ex = min(q_ex, q_max)
    demand = Demandfn(k_prev, k_max, v_max, k_cr, q_max, fd)
    if q_ex < demand:
        k = InvSupplyfn(q_ex, k_max, v_max, k_cr, q_max, fd)
    else:
        k = InvDemandfn(q_ex, k_max, v_max, k_cr, q_max, fd)
        
    return k

def flux_function(k_xup, k_xdn, k_cr, q_max, k_max, v_max, fd):
    '''
    Calculate flux across a cell boundary (Godunov scheme)
    '''
    if (k_xdn <= k_cr) and (k_xup <= k_cr):
        q_star = FundDiag(k_xup, k_max, v_max, k_cr, fd)
    elif (k_xdn <= k_cr) and (k_xup > k_cr):
        q_star = q_max
    elif (k_xdn > k_cr) and (k_xup <= k_cr):
        q_star = min(FundDiag(k_xdn, k_max, v_max, k_cr, fd), 
                     FundDiag(k_xup, k_max, v_max, k_cr, fd))
    elif (k_xdn > k_cr) and (k_xup > k_cr):
        q_star = FundDiag(k_xdn, k_max, v_max, k_cr, fd)
        
    return q_star

def density_update(k_x, k_xup, k_xdn, delt, delx, k_cr, q_max, k_max, v_max, fd):
    q_in = flux_function(k_xup, k_x, k_cr, q_max, k_max, v_max, fd)
    q_out = flux_function(k_x, k_xdn, k_cr, q_max, k_max, v_max, fd)
    k_x_nextt = k_x + (delt/delx)*(q_in - q_out)
    return k_x_nextt, q_out
    

def CFL_condition(delx, v_max):
    max_delt = delx/v_max
    return np.around(max_delt, 6)

def forward_sim(k_initial, q_entry, q_exit, t_nums, x_nums, fd, k_jam_space):
    
    # Runing the numerical scheme
    K = np.zeros((t_nums, x_nums))
    Q = np.zeros((t_nums, x_nums))
    for t in range(X_ind.shape[0]):
        # print(t)
        # Initial condition
        if t == 0:
            K[t, :] = k_initial
            continue
        # Remaining time
        for x in range(X_ind.shape[1]):
            # print(x)
            k_jam = k_jam_space[x]
            q_max = k_jam*v_free/4
            k_cr = k_jam/2
            # Get computational stencil
            k_x = K[t-1, x]
            if x == 0: # Starting Boundary condition
                q_en = q_entry[t]
                k_xup = bound_cond_entry(k_x, q_en, k_jam, v_free, k_cr, q_max, fd)
            else:
                k_xup = K[t-1, x-1]
            if x == x_nums-1: # Ending Boundary condition
                q_ex = q_exit[t]
                k_xdn = bound_cond_exit(k_x, q_ex, k_jam, v_free, k_cr, q_max, fd)
            else:
                k_xdn = K[t-1, x+1]
            
            # Calculated and update new density
            k_x_next, q_out = density_update(k_x, k_xup, k_xdn, delt, delx, 
                                             k_cr, q_max, k_jam, v_free, fd)
            K[t, x] = k_x_next
            Q[t, x] = q_out
    
    return K, Q

def train_initconds(x_nums):
    
    rin = np.random.randint
    run = np.random.uniform
    p = np.random.rand()
    
    if p < 0.20:
        i = rin(5, x_nums-5)
        k_initial = np.repeat(rin(80,100), x_nums)
        k_initial[i:] = max(0, k_initial[0] - rin(20,60))
    elif (p >= 0.20) and (p < 0.40):
        i = rin(5, x_nums-5)
        k_initial = np.repeat(rin(80,100), x_nums)
        k_initial[:i] = max(0, k_initial[0] - rin(20,60))
    elif (p >= 0.40) and (p < 0.65):
        i = rin(5, x_nums-20)
        j = max(i+rin(5, 20), rin(5, x_nums-10))
        k_initial = np.repeat(rin(0,100), x_nums)
        k_initial[:i] = max(0, k_initial[0]-rin(0,50))
        k_initial[j:] = max(0, k_initial[-1]-rin(0,50))
    elif (p >= 0.65) and (p < 0.90):
        k_initial = np.repeat(rin(0,50), x_nums)
        i1 = rin(5, int(x_nums/2))
        j1 = max(i1+rin(0,5), rin(5, int(x_nums/2)))
        k_initial[i1:j1] = min(115, k_initial[0]+rin(10,60))
        i2 = rin(int(x_nums/2), x_nums-5)
        j2 = max(i2+rin(0,5), rin(int(x_nums/2), x_nums))
        k_initial[i2:j2] = min(115, k_initial[-1]+rin(10,60))
    else:
        k_initial = run(0,100,x_nums)
        
    return k_initial
        
def test_scenarios(x_nums, t_nums, sc_type='random_w1signal'):
    
    rin = np.random.randint
    run = np.random.uniform
    
    if sc_type == 'random':
        k_initial = run(0,110,x_nums)           # 0-100
        q_entry = run(300,1200,t_nums)          # 300 - 1000
        q_exit = run(800,1500,t_nums)           # 800 - 1500
    
    elif sc_type == 'random_w1signal':
        k_initial = run(0,100,x_nums)
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
        q_exit[rin(50,150):rin(150,300)] = 0
    
    elif sc_type == 'random_w2signal':
        k_initial = run(0,100,x_nums)
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
        q_exit[rin(50,100):rin(100,200)] = 0
        q_exit[rin(250,350):rin(350,450)] = 0
    
    elif sc_type == 'random_w3signal':
        k_initial = run(0,100,x_nums)
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
        q_exit[rin(50,80):rin(80,150)] = 0
        q_exit[rin(200,230):rin(230,300)] = 0
        q_exit[rin(350,380):rin(380,450)] = 0
        
    elif sc_type == 'random_w4signal':
        i = rin(20, x_nums-10)
        k_initial = run(0,100,x_nums)
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
        q_exit[rin(50,100):rin(100,150)] = 0
        q_exit[rin(160,210):rin(210,260)] = 0
        q_exit[rin(270,320):rin(320,370)] = 0
        q_exit[rin(380,430):rin(430,480)] = 0
    
    elif sc_type == 'random_w5signal':
        i = rin(20, x_nums-10)
        k_initial = run(0,100,x_nums)
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
        q_exit[rin(40,80):rin(80,120)] = 0
        q_exit[rin(130,170):rin(170,210)] = 0
        q_exit[rin(220,260):rin(260,300)] = 0
        q_exit[rin(310,350):rin(350,390)] = 0
        q_exit[rin(400,440):rin(440,480)] = 0
        
        
    elif sc_type == 'random_w6signal':
        i = rin(20, x_nums-10)
        k_initial = run(0,100,x_nums)
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
        q_exit[rin(40,75):rin(75,110)] = 0
        q_exit[rin(120,155):rin(155,190)] = 0
        q_exit[rin(200,235):rin(235,270)] = 0
        q_exit[rin(280,315):rin(315,350)] = 0
        q_exit[rin(360,395):rin(395,430)] = 0
        q_exit[rin(440,475):rin(475,510)] = 0
        
    elif sc_type == 'random_w7signal':
        i = rin(20, x_nums-10)
        k_initial = run(0,100,x_nums)
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
        q_exit[rin(40,70):rin(70,100)] = 0
        q_exit[rin(110,140):rin(140,170)] = 0
        q_exit[rin(180,210):rin(210,240)] = 0
        q_exit[rin(250,280):rin(280,310)] = 0
        q_exit[rin(320,350):rin(350,380)] = 0
        q_exit[rin(390,420):rin(420,450)] = 0
        q_exit[rin(460,490):rin(490,520)] = 0
        
    elif sc_type == 'random_w8signal':
        i = rin(20, x_nums-10)
        k_initial = run(0,100,x_nums)
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
        q_exit[rin(40,67):rin(67,95)] = 0
        q_exit[rin(105,132):rin(132,160)] = 0
        q_exit[rin(170,197):rin(197,225)] = 0
        q_exit[rin(235,262):rin(262,290)] = 0
        q_exit[rin(300,327):rin(327,355)] = 0
        q_exit[rin(365,392):rin(392,420)] = 0
        q_exit[rin(430,457):rin(457,485)] = 0
        q_exit[rin(495,522):rin(522,550)] = 0
    
    elif sc_type == 'wavelet-1':
        i = rin(5, x_nums-20)
        j = max(i+rin(5, 20), rin(5, x_nums-10))
        k_initial = np.repeat(rin(0,100), x_nums)
        k_initial[:i] = max(0, k_initial[0]-rin(0,50))
        k_initial[j:] = max(0, k_initial[-1]-rin(0,50))
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
        
    elif sc_type == 'wavelet-2':
        
        k_initial = np.repeat(rin(0,50), x_nums)
        i1 = rin(5, int(x_nums/2))
        j1 = max(i1+rin(0,5), rin(5, int(x_nums/2)))
        k_initial[i1:j1] = min(115, k_initial[0]+rin(10,60))
        i2 = rin(int(x_nums/2), x_nums-5)
        j2 = max(i2+rin(0,5), rin(int(x_nums/2), x_nums))
        k_initial[i2:j2] = min(115, k_initial[-1]+rin(10,60))
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
    
    elif sc_type == 'wavelet-3':
        
        k_initial = np.repeat(rin(0,50), x_nums)
        i1 = rin(5, int(x_nums/3))
        j1 = max(i1+rin(5,10), rin(5, int(x_nums/3)))
        k_initial[i1:j1] = min(115, k_initial[0]+rin(10,60))
        i2 = rin(int(x_nums/3), int(2*x_nums/3))
        j2 = max(i2+rin(5,10), rin(int(x_nums/3), int(2*x_nums/3)))
        k_initial[i2:j2] = min(115, k_initial[-1]+rin(10,60))
        i3 = rin(int(2*x_nums/3), x_nums-5)
        j3 = max(i3+rin(5,10), rin(int(x_nums/2), x_nums))
        k_initial[i3:j3] = min(115, k_initial[-1]+rin(10,60))
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
    
    elif sc_type == 'wavelet-4':
        
        k_initial = np.repeat(rin(0,50), x_nums)
        i1 = rin(5, int(x_nums/4))
        j1 = max(i1+rin(0,5), rin(5, int(x_nums/4)))
        k_initial[i1:j1] = min(115, k_initial[0]+rin(10,60))
        i2 = rin(int(x_nums/4), int(x_nums/2))
        j2 = max(i2+rin(0,5), rin(int(x_nums/4), int(x_nums/2)))
        k_initial[i2:j2] = min(115, k_initial[-1]+rin(10,60))
        i3 = rin(int(x_nums/2), int(3*x_nums/4))
        j3 = max(i3+rin(0,5), rin(int(x_nums/2), int(3*x_nums/4)))
        k_initial[i3:j3] = min(115, k_initial[-1]+rin(10,60))
        i4 = rin(int(3*x_nums/4), x_nums-5)
        j4 = max(i4+rin(0,5), rin(int(3*x_nums/4), x_nums))
        k_initial[i4:j4] = min(115, k_initial[-1]+rin(10,60))
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
    
    elif sc_type == 'wavelet-5':
        
        k_initial = np.repeat(rin(0,50), x_nums)
        i1 = rin(0, int(x_nums/5))
        j1 = max(i1+rin(0,5), rin(5, int(x_nums/5)))
        k_initial[i1:j1] = min(115, k_initial[0]+rin(10,60))
        i2 = rin(int(x_nums/5), int(2*x_nums/5))
        j2 = max(i2+rin(0,5), rin(int(x_nums/5)+5, int(2*x_nums/5)))
        k_initial[i2:j2] = min(115, k_initial[-1]+rin(10,60))
        i3 = rin(int(2*x_nums/5), int(3*x_nums/5))
        j3 = max(i3+rin(0,5), rin(int(2*x_nums/5)+5, int(3*x_nums/5)))
        k_initial[i3:j3] = min(115, k_initial[-1]+rin(10,60))
        i4 = rin(int(3*x_nums/5), int(4*x_nums/5))
        j4 = max(i4+rin(0,5), rin(int(3*x_nums/5)+5, int(4*x_nums/5)))
        k_initial[i4:j4] = min(115, k_initial[-1]+rin(10,60))
        i5 = rin(int(4*x_nums/5), x_nums)
        j5 = max(i5+rin(0,5), rin(int(4*x_nums/5)+5, x_nums))
        k_initial[i5:j5] = min(115, k_initial[-1]+rin(10,60))
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
    
    elif sc_type == 'wavelet-6':
        
        k_initial = np.repeat(rin(0,50), x_nums)
        i1 = rin(0, int(x_nums/6))
        j1 = max(i1+rin(0,5), rin(5, int(x_nums/6)))
        k_initial[i1:j1] = min(115, k_initial[0]+rin(10,60))
        i2 = rin(int(x_nums/6), int(2*x_nums/6))
        j2 = max(i2+rin(0,5), rin(int(x_nums/6)+5, int(2*x_nums/6)))
        k_initial[i2:j2] = min(115, k_initial[-1]+rin(10,60))
        i3 = rin(int(2*x_nums/6), int(3*x_nums/6))
        j3 = max(i3+rin(0,5), rin(int(2*x_nums/6)+5, int(3*x_nums/6)))
        k_initial[i3:j3] = min(115, k_initial[-1]+rin(10,60))
        i4 = rin(int(3*x_nums/6), int(4*x_nums/6))
        j4 = max(i4+rin(0,5), rin(int(3*x_nums/6)+5, int(4*x_nums/6)))
        k_initial[i4:j4] = min(115, k_initial[-1]+rin(10,60))
        i5 = rin(int(4*x_nums/6), int(5*x_nums/6))
        j5 = max(i5+rin(0,5), rin(int(4*x_nums/6)+5, int(5*x_nums/6)))
        k_initial[i5:j5] = min(115, k_initial[-1]+rin(10,60))
        i6 = rin(int(5*x_nums/6), x_nums)
        j6 = max(i6+rin(0,5), rin(int(5*x_nums/6)+5, x_nums))
        k_initial[i6:j6] = min(115, k_initial[-1]+rin(10,60))
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
    
    elif sc_type == 'sic-10':
        
        i = 0
        k_initial = np.repeat(rin(40,80), x_nums)
        for t in range(10):
            j = min(50, rin(i, i+10))
            k_initial[j:] =  max(0, min(115, k_initial[j-1]+rin(-20,20)))
            i = j
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
    
    elif sc_type == 'sic-20':
        
        i = 0
        k_initial = np.repeat(rin(40,80), x_nums)
        for t in range(20):
            j = min(50, rin(i, i+5))
            k_initial[j:] =  max(0, min(115, k_initial[j-1]+rin(-20,20)))
            i = j
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
    
    elif sc_type == 'sic-30':
        
        i = 0
        k_initial = np.repeat(rin(40,80), x_nums)
        for t in range(30):
            j = min(50, rin(i, i+5))
            k_initial[j:] =  max(0, min(115, k_initial[j-1]+rin(-20,20)))
            i = j
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
    
    elif sc_type == 'sic-40':
        
        i = 0
        k_initial = np.repeat(rin(40,80), x_nums)
        for t in range(40):
            j = min(50, rin(i, i+4))
            k_initial[j:] =  max(0, min(115, k_initial[j-1]+rin(-20,20)))
            i = j
        q_entry = run(300,1000,t_nums)
        q_exit = run(800,1500,t_nums)
    
    else:
        print('No scenario selected...?')   
    
    return k_initial, q_entry, q_exit 


def plot_densities(K):
    
    fig, ax = plt.subplots()
    ax.axis('off')
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(top=0.95, bottom=0.05, left=0.1, right=0.9, wspace=0)
    ax = plt.subplot(gs0[:, :])
    h = ax.imshow(K.T, cmap='rainbow', extent=[0, t_max, 0, x_max], 
                  origin='lower', aspect='auto', vmin=0, vmax=k_jam)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax) 
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$k(t, x)$', fontsize=12)
    
    return fig, ax
    

# =============================================================================
# Simulation parameters
# =============================================================================

# parameters
k_jam = 120                         # in vehicles/km
v_free = 60                         # in km/hr
v_b = -15
fd = 'Greenshield'

if fd == 'Greenshield':
    q_max = k_jam*v_free/4          # in vehicles/hr
    k_cr = k_jam/2                  # for Greenshield's model
elif fd=='Triangular':
    k_cr = 0.40*k_jam
    q_max = v_free*k_cr
elif fd == 'NewellFrank':
    k_cr = 40
    k_arr = np.arange(0, k_jam+0.1, 0.1)
    q_arr = FundDiag(k_arr, k_jam, v_free, k_cr, fd)
    q_max = q_arr.max()
    q_argmax = np.argmax(q_arr)
    k_cr = k_arr[q_argmax]
    dem_fn = q_arr.copy()
    dem_fn[q_argmax:] = q_max
    sup_fn = q_arr.copy()
    sup_fn[:q_argmax] = q_max

# Cell discretization
x_max = 1000/1000                   # road length in kilometres
t_max = 600/3600                    # time period of simulation in hours
delx = 20/1000                      # cell length in kilometres
delt = 1/3600                       # time discretization in hours
cfl = (v_free)/(delx/delt)
x_nums = round(x_max/delx)
t_nums = round(t_max/delt)
x_space = np.linspace(0,x_max,x_nums)
t_time = np.linspace(0,t_max,t_nums)

# Time-space mesh creation
x_ind = np.arange(0, x_nums)
t_ind = np.arange(0, t_nums)
X_ind, T_ind = np.meshgrid(x_ind, t_ind)

# variable jam densities
k_jam_space = np.repeat(k_jam, x_nums)

# =============================================================================
# Numerically solve a single input condition
# =============================================================================

# # Single simulation
# k_initial = np.random.uniform(0,100,x_nums)
# k_initial = savgol_filter(k_initial, 15, 5)
# q_entry = np.random.uniform(300,1000,t_nums)
# q_exit = np.random.uniform(800,1500,t_nums)
# q_exit[np.random.randint(50,150):np.random.randint(150,300)] = 0
# k_initial, q_entry, q_exit = test_scenarios(x_nums, t_nums, sc_type='random-1')
# K, Q = forward_sim(k_initial, q_entry, q_exit, t_nums, x_nums, fd, k_jam_space)
# _ = plot_densities(K,Q)


# =============================================================================
# Generate training data
# =============================================================================

run = np.random.uniform
rin = np.random.randint
train_cases = ['sic1', 'sic2', 'sic3']

num_runs = 50
for sc in train_cases:
    print(f'Training input conditions: {sc}')
    
    K_arr = []
    Q_arr = []
    num_runs = 2750
    for n in range(num_runs):
        print(f'\t Run = {n}')
        
        # Initial condition
        k_initial = train_initconds(x_nums)
        
        # Boundary condition (add two wavelets probabilistically)
        q_entry = run(300, 1500, t_nums)
        q_exit = run(800, 1500, t_nums)
        p = np.random.rand()
        if (p <= 0.40):
            q_exit[rin(50,150):rin(150,400)] = 0
        elif (p > 0.40) and (p <= 0.80):
            q_exit[rin(50,100):rin(100,200)] = 0
            q_exit[rin(250,350):rin(350,450)] = 0
        
        # Run Godunov scheme
        K, Q = forward_sim(k_initial, q_entry, q_exit, t_nums, x_nums, fd, k_jam_space)
        K_arr.append(K)
        Q_arr.append(Q)
    
    # Extract input-output pairs
    num_train_runs = 2500
    TrainX = []; TrainY = []
    TestX = []; TestY = []
    for n in range(num_runs):
        print(n)
        K, Q = K_arr[n], Q_arr[n]
        K_out = K.copy().astype('float32')
        K_inp = K.copy().astype('float32')
        K_inp[1:,1:-1] = k_jam
        if n <= num_train_runs-1:
            TrainX.append(K_inp)
            TrainY.append(K_out)
        else:
            TestX.append(K_inp)
            TestY.append(K_out)
    
    # Save data offline    
    data = {'X':TrainX, 'Y':TrainY}
    with open('train/train_lwrg20x1gs-{}.pkl'.format(sc), 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
    data = {'X':TestX, 'Y':TestY}
    with open('train/vald_lwrg20x1gs-{}.pkl'.format(sc), 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

    
# =============================================================================
# Generate testing data
# =============================================================================

test_cases = ['random','random_w1signal','random_w2signal',
              'random_w3signal','random_w4signal',
              'random_w5signal','random_w6signal',
              'random_w7signal','random_w8signal',
              'wavelet-1','wavelet-2','wavelet-3',
              'wavelet-4','wavelet-5','wavelet-6',
              'sic-10','sic-20','sic-30','sic-40']

num_runs = 50
for sc in test_cases:
    print(f'Testing input conditions: {sc}')
    
    TestX = []; TestY = []
    for n in range(num_runs):
        print(f'\tRun = {n}')
        
        # generate conditions
        k_initial, q_entry, q_exit = test_scenarios(x_nums, t_nums, sc)    
        
        # density solution
        K,_ = forward_sim(k_initial, q_entry, q_exit, t_nums, x_nums, fd, k_jam_space)
        
        # create input-output pair
        K_out = K.copy().astype('float32')
        K_inp = K.copy().astype('float32')
        K_inp[1:,1:-1] = k_jam
        TestX.append(K_inp)
        TestY.append(K_out)
        
    # Save data offline    
    data = {'X':TestX, 'Y':TestY}
    with open('test/test_lwrg20x1gs-{}.pkl'.format(sc), 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

