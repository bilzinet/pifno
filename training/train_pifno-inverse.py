"""
Created: Jan 2022
Author: Bilal Thonnam Thodi (btt1@nyu.edu)

Physics Informed Fourier Neural Operator Model
Learning Inverse Solutions for LWR Model 
Params: 20m x 1sec grid resolution, Greenshield, homogeneous road section

NN Model: Original FNO model with added physics loss function (Godunov equations)

"""

# =============================================================================
# Packages and settings
# =============================================================================

# load packages
import sys
import numpy as np
import torch
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial
from timeit import default_timer

# some settings
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Fourier Neural Operator (Reference: https://arxiv.org/pdf/2010.08895.pdf)
# =============================================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer: FFT -> Linear Transform -> Inverse FFT  
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        # Perform Fourier transform
        batchsize = x.shape[0]
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        # Multiply top Fourier modes with Fourier weights
        out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Perform Inverse Fourier transform
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=( x.size(-2), x.size(-1)))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        
        # Projection P
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # FNO Layer 1
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # FNO Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # FNO Layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # FNO Layer 4
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # Projection Q
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        """
        A wrapper function
        """
        self.conv1 = SimpleBlock2d(modes1, modes2,  width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

# =============================================================================
# Some useful functions
# =============================================================================
'''
Godunov-based physics loss function

k(x,t+1) = k(x,t) + (delt/delx) * (q_in(x,t) - q_out(x,t))

where: 
    q_in(x,t) = min ( dem(x-1,t), supp(x,t) )
    q_out(x,t) = min( dem(x,t), supp(x+1,t) )

    where:
        dem(x-1,t) = Q(k(x-1,t)) if k(x-1) < k_cr else q_max
        sup(x,t) = Q(k(x-1,t)) if k(x-1) > k_cr else q_max

'''

# Relative l-2 norm loss function
class DataLoss(object):
    def __init__(self, p=2):
        super(DataLoss, self).__init__()
        self.p = p

    def data_loss(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        return torch.mean(diff_norms/y_norms)
    
    def __call__(self, x, y):
        return self.data_loss(x, y)

# Relative physics loss function
class PhysLoss(object):
    def __init__(self, p=2):
        super(PhysLoss, self).__init__()
        self.p = p
    
    def godunov_res(self, y):
        # parameters
        k_max = 120
        v_max = 60
        k_cr = k_max/2
        q_max = (k_max*v_max)/4
        def flux(k): return k*v_max*(1-k/k_max)
        alpha = ((1/3600)/(20/1000))
        
        # compute flow in
        dem_prev = (y[:,:-1,:-2] <= k_cr)*flux(y[:,:-1,:-2]) + (y[:,:-1,:-2] > k_cr)*q_max
        sup_curr = (y[:,:-1,1:-1] > k_cr)*flux(y[:,:-1,1:-1]) + (y[:,:-1,1:-1] <= k_cr)*q_max
        q_p_c = torch.stack([dem_prev, sup_curr], dim=-1)
        q_in = torch.min(q_p_c, dim=-1, keepdim=False)[0]
        
        # compute flow out
        dem_curr = (y[:,:-1,1:-1] <= k_cr)*flux(y[:,:-1,1:-1]) + (y[:,:-1,1:-1] > k_cr)*q_max
        sup_next = (y[:,:-1,2:] > k_cr)*flux(y[:,:-1,2:]) + (y[:,:-1,2:] <= k_cr)*q_max
        q_c_n = torch.stack([dem_curr, sup_next], dim=-1)
        q_out = torch.min(q_c_n, dim=-1, keepdim=False)[0]
        
        # compute godunov_loss
        num_examples = y.size()[0]
        q_in, q_out = q_in.to(device), q_out.to(device)
        phys_loss_num = y[:,1:,1:-1]-y[:,:-1,1:-1]-(alpha)*(q_in-q_out)
        phys_loss_num = torch.norm(phys_loss_num.reshape(num_examples,-1), self.p, 1)
        phys_loss_den = y[:,1:,1:-1]
        phys_loss_den = torch.norm(phys_loss_den.reshape(num_examples,-1), self.p, 1)
        phys_loss = torch.mean(phys_loss_num/phys_loss_den)
        return phys_loss
    
    def __call__(self, y):
        return self.godunov_res(y)

# Complex multiplication
def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

# initiale weights
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # torch.nn.init.xavier_uniform_(m.bias)
            torch.nn.init.zeros_(m.bias)
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

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

# random masking of boundary values
def rand_mask_boundary(x_test, mp=0):
    if mp==0:
        return x_test
    else:
        mk=int(mp*x_test.shape[1])
        mind=np.random.randint(0,x_test.shape[1],size=(x_test.shape[0],mk))
        for i in range(mind.shape[0]):
            x_test[...,0][i,mind[i,:]]=-1
            x_test[...,-1][i,mind[i,:]]=-1
        return x_test

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
            if np.random.uniform() < 0.10:
                x0 = np.random.randint(0,49)
                t0 = 0
            else:
                x0 = 0
                t0 = np.random.randint(0,599)
            indx_rand = genTraj(x0, t0, v_map[sam], dx=20, dt=1)
            x_test[sam,indx_rand[0,:],indx_rand[1,:]] = (y_test[sam,indx_rand[0,:],indx_rand[1,:]]).copy()
    return x_test

def load_data(f_names_train, f_names_test, data_fold_train, data_fold_test, b_size, ntest_sc, ntest_max, train=True, normalize=True, **kwargs):
    
    # load test data
    x_test = []; y_test = []
    for f in f_names_test:
        with open(data_fold_test+f'test_{f}.pkl','rb') as f:
            data = pkl.load(f)
        x_test.append(data['X'][:ntest_sc])
        y_test.append(data['Y'][:ntest_sc])
    x_test = np.concatenate(x_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.float32)
    x_test = x_test[:ntest_max,:,:]
    y_test = y_test[:ntest_max,:,:]
    x_test = rand_mask_interior_traj(x_test, y_test)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    
    # grid size params
    s1 = x_test.shape[1]
    s2 = x_test.shape[2]
    ntest = x_test.shape[0]
    
    # load train data if required
    ntrain = 0
    if train:
        x_train = []; y_train = []
        for f in f_names_train:
            with open(data_fold_train+f'train_{f}.pkl','rb') as f:
                data = pkl.load(f)
            x_train.append(data['X'][:ntrain_sc])
            y_train.append(data['Y'][:ntrain_sc])
        x_train = np.concatenate(x_train, axis=0).astype(np.float32)
        y_train = np.concatenate(y_train, axis=0).astype(np.float32)
        x_train = x_train[:ntrain_max,:,:]
        y_train = y_train[:ntrain_max,:,:]
        x_train = rand_mask_interior_traj(x_train, y_train)
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)    
        ntrain = x_train.shape[0]
    
    # concat location coordinates
    grids = []
    grids.append((np.linspace(1,600,s1)+np.linspace(0,600-1,s1))/2)                                   # np.linspace(0,1,s1) (np.linspace(20,1000,s1)+np.linspace(0,1000-20,s1))/2
    grids.append((np.linspace(20,1000,s2)+np.linspace(0,1000-20,s2))/2)             # np.linspace(0,1,s2) (np.linspace(1,600,s2)+np.linspace(0,600-1,s2))/2
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s1,s2,2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_test = torch.cat([x_test.reshape(ntest,s1,s2,1), 
                        grid.repeat(ntest,1,1,1)], dim=3)
    
    # pytorch loader
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), 
        batch_size=b_size, shuffle=False)
    train_loader = None
    if train:
        x_train = torch.cat([x_train.reshape(ntrain,s1,s2,1), 
                             grid.repeat(ntrain,1,1,1)], dim=3)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train), 
            batch_size=b_size, shuffle=True)
    
    return train_loader, test_loader, ntrain, ntest, s1, s2

def train(train_loader):
    
    model.train()
    train_dataloss = 0
    train_physloss = 0
    for x, y in train_loader:
        # initialize
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # forward pass
        out = model(x)    
        # backward pass
        d_loss = data_loss(y, out)
        p_loss = phys_loss(out)
        loss = d_loss + lam*p_loss
        loss.backward()
        # update
        optimizer.step()
        train_dataloss += d_loss.item()
        train_physloss += p_loss.item()
    train_dataloss /= len(train_loader)
    train_physloss /= len(train_loader)
    # update learning rate
    scheduler.step()
    
    return train_dataloss,train_physloss

def test(test_loader):
    
    model.eval()
    test_dataloss = 0.0
    test_physloss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_dataloss += data_loss(out, y).item()
            test_physloss += phys_loss(out).item()
    test_dataloss /= len(test_loader)
    test_physloss /= len(test_loader)
    
    return test_dataloss,test_physloss

def eval_test(eval_test_loader, eval_ntest, eval_s1, eval_s2):
    
    index = 0
    act = torch.zeros((eval_ntest, eval_s1, eval_s2))
    pred = torch.zeros((eval_ntest, eval_s1, eval_s2))
    inps = torch.zeros((eval_ntest, eval_s1, eval_s2, 3))
    with torch.no_grad():
        for x, y in eval_test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred[index] = out
            act[index] = y.squeeze(0)
            inps[index] = x
            index = index + 1
    K_pred = pred.cpu().numpy()
    K_act = act.cpu().numpy()
    K_rmse = np.sqrt(np.mean(np.power(K_pred-K_act, 2), axis=(1,2)))
    K_mae = np.mean(np.abs(K_pred-K_act), axis=(1,2))
    K_inps = inps.cpu().numpy()
    
    return K_act, K_pred, K_inps, K_rmse, K_mae

# =============================================================================
# Parameters
# =============================================================================

mainexpt = 'inverse'
expt = str(sys.argv[1])
lam_dict = {'fno':0.0, 'pifno':2.0}
train_res = '20x1gs'
load_model = True

batch_size = 128
learning_rate = 1e-3
epochs = 500
step_size = 100
gamma = 0.5
lam = lam_dict[expt]

modes1 = 128
modes2 = 24
width = 64

ntrain_sc = 1300
ntest_sc = 100
ntrain_max = 10000
ntest_max = 500

# =============================================================================
# Build model
# =============================================================================

# fourier neural operator model
if load_model:
    model = torch.load('{}_kmodel-{}.pt'.format(mainexpt,expt), 
                       map_location=torch.device('cuda'))
else:
    model = FNO2d(modes1, modes2, width)
    model.apply(init_weights)
    model.to(device)

# optimizer definition
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# =============================================================================
# Load data
# =============================================================================

print('\n ------ Loading dataset -------')

# data files
train_req = True
df_test = '../data/'
df_train = '../data/'
f_names_train = ['lwrg20x1gs_sic-1','lwrg20x1gs_sic-2','lwrg20x1gs_sic-3', 'lwrg20x1gs_rps-1']
f_names_test =  ['lwrg20x1gs-random_w2signal','lwrg20x1gs-random_w3signal']

# train batch loader
train_loader, test_loader, ntrain, ntest, s1, s2 = load_data(
    f_names_train, f_names_test, df_train, df_test, batch_size, ntest_sc, ntest_max, train_req, ntrain_sc=ntrain_sc, ntrian_max=ntrain_max)

print('number of train samples: ', ntrain)
print('number of test samples: ', ntest)
print('grid resolution of train: ', s1, ' x ', s2)

# =============================================================================
# Training FNO-2D model
# =============================================================================

if not(load_model):
    print('Training the FNO model...\n')
    
    # initialize
    data_loss = DataLoss()
    phys_loss = PhysLoss()
    train_dataarr=[]; train_physarr=[]
    test_dataarr=[]; test_physarr=[]
    
    # training loop
    for ep in range(1,epochs+1):
        
        # training and validation
        t1 = default_timer()
        train_dloss, train_ploss = train(train_loader)
        test_dloss, test_ploss = test(test_loader)
        train_dataarr.append(train_dloss)
        train_physarr.append(train_ploss)
        test_dataarr.append(test_dloss)
        test_physarr.append(test_ploss)
        t2 = default_timer()
        print(f'{ep}, {t2-t1:.03f}, {train_dloss:.03f}, {train_ploss:.03f}, {test_dloss:.03f}, {test_ploss:.03f}')
            
    # save offline
    print('\n ------ Saving model offline -------')
    print('train mse: ', train_dloss)
    print('test mse: ', test_dloss)
    np.save('../model/{}_train_results-{}.npy'.format(mainexpt,expt),[train_dataarr,train_physarr,test_dataarr,test_physarr])
    torch.save(model, '../model/{}_kmodel-{}.pt'.format(mainexpt,expt))

else:
    print('No training! Testing the FNO model...\n')
    
# =============================================================================
# Testing results
# =============================================================================

print('\n ------ Evaluatng on test dataset -------')

# data params
i = 0
eval_batch_size = 1
eval_ntest_sc = 50
eval_ntest_max = 1000
eval_ntrain_sc = 0
eval_ntrain_max = 0
eval_train_req = False
eval_f_names = ['lwrg20x1gs-random',
                'lwrg20x1gs-random_w1signal',
                'lwrg20x1gs-random_w2signal',
                'lwrg20x1gs-random_w3signal',
                'lwrg20x1gs-random_w4signal',
                'lwrg20x1gs-random_w5signal',
                'lwrg20x1gs-random_w6signal',
                'lwrg20x1gs-random_w7signal',
                'lwrg20x1gs-random_w8signal',
                'lwrg20x1gs-wavelet-1',
                'lwrg20x1gs-wavelet-2',
                'lwrg20x1gs-wavelet-3',
                'lwrg20x1gs-wavelet-4',
                'lwrg20x1gs-wavelet-5',
                'lwrg20x1gs-wavelet-6',
                'lwrg20x1gs-sic-10',
                'lwrg20x1gs-sic-20',
                'lwrg20x1gs-sic-30',
                'lwrg20x1gs-sic-40']
K_mae_arr = np.zeros((len(eval_f_names), eval_ntest_sc))
K_comp_arr = np.zeros((len(eval_f_names), 3, eval_ntest_sc, s1, s2))

for f_name in eval_f_names:
    
    test_res = f_name.split("-")[0][4:]

    # define batch loader
    _,eval_test_loader,_,eval_ntest,eval_s1,eval_s2 = load_data(
        [f_name],[f_name],df_train,df_test,eval_batch_size,eval_ntest_sc,eval_ntest_max,eval_train_req,ntrain_sc=eval_ntrain_sc,ntrain_max=eval_ntrain_max, train_res=test_res)

    # Predicting over all test 
    K_act,K_pred,K_inps,K_rmse,K_mae = eval_test(eval_test_loader,eval_ntest,eval_s1,eval_s2)
    
    # save results offline
    K_mae_arr[i,:] = K_mae
    K_comp_arr[i,0,:,:,:] = K_act
    K_comp_arr[i,1,:,:,:] = K_pred
    K_comp_arr[i,2,:,:,:] = K_inps[:,:,:,0]
    
    # print results
    print('\n File name: ', f_name)
    print('Number of evaluation samples: ', eval_ntest)
    print(f'MAE: {K_mae.mean():0.02f} +- {K_mae.std():0.02f} vehs/km')
    i += 1
    
np.save('../model/{}_kmae_results-{}.npy'.format(mainexpt, expt), K_mae_arr)
np.save('../model/{}_kcomp_results-{}.npy'.format(mainexpt, expt), K_comp_arr)

print('\n ------ Training and Evalution done -------')
