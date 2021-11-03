import torch
import torch.nn as nn
import torch.optim as optim
# from collections import OrderedDict 
import numpy as np
import warnings
import time

import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs

from tqdm.auto import tqdm, trange

from utils import Misc

import os
ROOT= os.getcwd()

ACTIVATIONS = [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.LogSigmoid(), nn.Tanh(), nn.Tanhshrink(),
               nn.Hardtanh(), nn.ELU()]


def pinn_ext_binary_decoder(X):
    
    if X.ndim == 2:
        X = np.squeeze(X)
    #
  
    binary_code = X[:]
    binary_code = (binary_code>=0.5).astype(int)
    
    bits_activation = 3
    bits_hidden_depth = 4
    bits_hidden_width = 8
    
    #print(binary_code)

    curr = 0
    
    binary_activation = ''.join([str(x) for x in binary_code[curr: curr+bits_activation].tolist()])
    act_fn = ACTIVATIONS[int(binary_activation, 2)]
    curr += bits_activation
    
    binary_hidden_depth = ''.join([str(x) for x in binary_code[curr: curr+bits_hidden_depth].tolist()])
    hidden_depth_offset = 1
    hidden_depth = int(binary_hidden_depth, 2)+hidden_depth_offset
    curr += bits_hidden_depth
    
    binary_hidden_width = ''.join([str(x) for x in binary_code[curr: curr+bits_hidden_width].tolist()])
    hidden_depth_offset = 1
    hidden_width = int(binary_hidden_width,2)+hidden_depth_offset
    curr += bits_hidden_width
    
    pinn_config = {}
    pinn_config['act_fn'] = act_fn
    pinn_config['hidden_depth'] = hidden_depth
    pinn_config['hidden_width'] = hidden_width
    
    return pinn_config

# def pinn_binary_decoder_v2(X):
    
#     if X.ndim == 2:
#         X = np.squeeze(X)
#     #
  
#     binary_code = X[:]
#     binary_code = (binary_code>=0.5).astype(int)
    
#     bits_activation = 3
#     bits_hidden_depth = 3
#     bits_hidden_width = 6
    
#     #print(binary_code)

#     curr = 0
    
#     binary_activation = ''.join([str(x) for x in binary_code[curr: curr+bits_activation].tolist()])
#     act_fn = ACTIVATIONS[int(binary_activation, 2)]
#     curr += bits_activation
    
#     binary_hidden_depth = ''.join([str(x) for x in binary_code[curr: curr+bits_hidden_depth].tolist()])
#     hidden_depth_offset = 1
#     hidden_depth = int(binary_hidden_depth, 2)+hidden_depth_offset
#     curr += bits_hidden_depth
    
#     binary_hidden_width = ''.join([str(x) for x in binary_code[curr: curr+bits_hidden_width].tolist()])
#     hidden_depth_offset = 1
#     hidden_width = int(binary_hidden_width,2)+hidden_depth_offset
#     curr += bits_hidden_width
    
#     pinn_config = {}
#     pinn_config['act_fn'] = act_fn
#     pinn_config['hidden_depth'] = hidden_depth
#     pinn_config['hidden_width'] = hidden_width
    
#     return pinn_config
    
class Net(nn.Module):
    def __init__(self, layers, lb, ub, act=nn.Tanh()):
        super(Net, self).__init__()
        self.act = act
        self.fc = nn.ModuleList()
#         self.lb = torch.tensor(lb, dtype=torch.float32)
#         self.ub = torch.tensor(ub, dtype=torch.float32)
        self.lb = lb
        self.ub = ub
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight)
        
    def forward(self, x):
        x = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x)
        return x


class PINN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, activation, device):
        self.lb = lb
        self.ub = ub
        self.nu = nu
        self.device = device
    
        #boundary points --- training
        self.xt_u = torch.tensor(X_u, dtype=torch.float32).to(self.device)
        self.u = torch.tensor(u, dtype=torch.float32).to(self.device)
        
        #collocation points --- residual
        self.x_f = torch.tensor(X_f[:,0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        self.t_f = torch.tensor(X_f[:,1:2], dtype=torch.float32, requires_grad=True).to(self.device)
        
        lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        ub = torch.tensor(ub, dtype=torch.float32).to(self.device)
        
        self.u_net = Net(layers, lb, ub, activation).to(self.device)
        self.nepoch = 50000

    def get_loss(self):
        mse_u = (self.u - self.u_net(self.xt_u)).square().mean()
        u = self.u_net(torch.cat((self.x_f, self.t_f), 1))
        u_sum = u.sum()
        u_t = torch.autograd.grad(u_sum, self.t_f, create_graph=True)[0]
        u_x = torch.autograd.grad(u_sum, self.x_f, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), self.x_f, create_graph=True)[0]
        mse_f = (u_t +u*u_x - self.nu*u_xx).square().mean()
        return mse_u + mse_f

#     def train_sgd(self, X_star, u_star):
#         optimizer = torch.optim.Adam(self.u_net.parameters(), lr=1e-3)
#         for n in range(self.nepoch):
#             loss = self.get_loss()
#             if n%100==0:
#                 with torch.no_grad():
#                     u_pred = self.u_net(torch.from_numpy(X_star).float()).numpy()
#                     error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
#                     print('epoch %d, loss: %g, Error u: %g'%(n, loss.item(), error_u))

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         #test
#         with torch.no_grad():
#             u_pred = self.u_net(torch.from_numpy(X_star).float()).numpy()
#             error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
#             print('Error u: %g'%error_u)

#     def train(self, X_star, u_star, max_iter):
#         #history_size = 100 will crash
#         optimizer = torch.optim.LBFGS(self.u_net.parameters(), lr=0.1, max_iter=max_iter, max_eval=None,
#             tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50, line_search_fn='strong_wolfe')
#         def closure():
#             optimizer.zero_grad()
#             loss = self.get_loss()
#             loss.backward(retain_graph=True)
#             #print('loss:%g'%loss.item())
#             return loss
#         #
#         optimizer.step(closure)
#         #test
#         with torch.no_grad():
#             u_pred = self.u_net(torch.from_numpy(X_star).float()).numpy()
#             error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
# #             return error_u
#             return np.log(error_u)
#             #print('Error u: %g'%error_u)

    def train(self, X_star, u_star, max_iter):
        #history_size = 100 will crash
        optimizer = torch.optim.LBFGS(self.u_net.parameters(), lr=0.1, max_iter=max_iter, max_eval=None,
            tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward(retain_graph=True)
            #print('loss:%g'%loss.item())
            return loss
        #
        optimizer.step(closure)
        #test
        
        torch_X_star = torch.from_numpy(X_star).float().to(self.device)
        with torch.no_grad():
            
#             u_pred = self.u_net(torch.from_numpy(X_star).float()).numpy()
            u_pred = self.u_net(torch_X_star).data.cpu().numpy()
            error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
            return np.log(error_u)
#             print('Error u: %g'%error_u)
    
    
def eval_pinn_ext_performace(domain, binary_config, max_iters, mode, device):
    
    if binary_config.ndim == 2:
        binary_config = binary_config.squeeze()
        
    pinn_config = pinn_ext_binary_decoder(binary_config)
    
    #print(pinn_config)
    
    nu = 0.01/np.pi #viscosity
    noise = 0.0        

    N_u = 200 #no. of boundary points
    N_f = 10000 #no. of collocation points
    
#     data_home=os.path.join('functionals/cache')
#     data = scipy.io.loadmat('./functionals/cache/burgers_shock.mat')
    data = scipy.io.loadmat(os.path.join(ROOT, 'functionals/cache/burgers_shock.mat'))
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    #u(0,x)
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    #u(t, -1)
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    #u(t,1)
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
    
    
    layers = [2] + [pinn_config['hidden_width']]*pinn_config['hidden_depth'] + [1]
    
    fidelity_list = [10,100,50000]
    #fidelity_list = [5,10,50]
    
    
    if mode == 'query':
        model = PINN(X_u_train, u_train, X_f_train, layers, lb, ub, nu, pinn_config['act_fn'], device)
        test_error = model.train(X_star, u_star, max_iters)
        return -test_error
    elif mode == 'generate':
        mf_test_error = []
        for fid in fidelity_list:
            model = PINN(X_u_train, u_train, X_f_train, layers, lb, ub, nu, pinn_config['act_fn'], device)
            test_error = model.train(X_star, u_star, fid)
            mf_test_error.append(test_error)
        #
        return -np.array(mf_test_error)
    
    
    