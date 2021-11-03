import torch
import torch.nn as nn
from torch.optim import LBFGS

import hamiltorch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm, trange

from multiprocessing import Pool

class FullAutoRegressive:
    def __init__(self, mf_layers_list, activation, device):
        
        self.M = len(mf_layers_list)
        
        print('layers:',mf_layers_list)
        
        self.device = device
        self.mf_layers_list = mf_layers_list
        self.actfn = {'tanh': torch.nn.Tanh(), 'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid()}[activation]
        
        self.mf_weights = []
        self.mf_biases = []
        self.mf_log_tau = []

        for m in range(self.M):
            weights, biases = self._init_nn_weights(m)
            log_tau = torch.tensor([0.0], requires_grad=True).to(self.device)
            self.mf_weights.append(weights)
            self.mf_biases.append(biases)
            self.mf_log_tau.append(log_tau)

    def _xavier_init(self, out_d, in_d):
        xavier_stddev = np.sqrt(2.0/(in_d + out_d))
        W = torch.normal(size=(out_d,in_d), mean=0.0, std=xavier_stddev, requires_grad=True).to(self.device)
        b = torch.normal(size=(out_d,), mean=0.0, std=xavier_stddev, requires_grad=True).to(self.device)
        return W, b
        
    def _init_nn_weights(self, m):
        layers_config = self.mf_layers_list[m]
        num_layers = len(layers_config)
        
        weights = []
        biases = []
        
        for l in range(num_layers-1):
            in_d = layers_config[l]
            out_d = layers_config[l+1]
            W,b = self._xavier_init(out_d, in_d)
            weights.append(W)
            biases.append(b)
        #
        
        return weights, biases
    
    def _fidelity_forward_by_params(self, X, weights, biases):
        H = X
        num_layers = len(weights)
      
        for l in range(num_layers-1):
            in_d = weights[l].shape[1]
            H = torch.add(torch.matmul(H, weights[l].T), biases[l])
            H = self.actfn(H)
            
        #
        
        in_d = weights[-2].shape[1]
        Y = torch.add(torch.matmul(H, weights[-1].T), biases[-1])

        return Y
    
    def forward_by_params(self, X, m, params_mf_weights, params_mf_biases):
        prev_outputs_list = []
        Ym = self._fidelity_forward_by_params(X, params_mf_weights[0], params_mf_biases[0])
        prev_outputs_list.append(Ym)
        
        for im in range(1, m+1):
            #print(im)
            X_cat = torch.cat(prev_outputs_list + [X], dim=1)
            Ym = self._fidelity_forward_by_params(X_cat, params_mf_weights[im], params_mf_biases[im])
            prev_outputs_list.append(Ym)
        #
        
        return Ym
    

    def forward(self, X, m):
        Ym = self.forward_by_params(X, m, self.mf_weights, self.mf_biases)
        return Ym
    
class SeqAutoRegressive:
    def __init__(self, mf_layers_list, activation, device):
        
        self.M = len(mf_layers_list)
        
        print('layers:',mf_layers_list)
        
        self.device = device
        self.mf_layers_list = mf_layers_list
        self.actfn = {'tanh': torch.nn.Tanh(), 'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid()}[activation]
        
        self.mf_weights = []
        self.mf_biases = []
        self.mf_log_tau = []

        for m in range(self.M):
            weights, biases = self._init_nn_weights(m)
            log_tau = torch.tensor([0.0], requires_grad=True).to(self.device)
            self.mf_weights.append(weights)
            self.mf_biases.append(biases)
            self.mf_log_tau.append(log_tau)

    def _xavier_init(self, out_d, in_d):
        xavier_stddev = np.sqrt(2.0/(in_d + out_d))
        W = torch.normal(size=(out_d,in_d), mean=0.0, std=xavier_stddev, requires_grad=True).to(self.device)
        b = torch.normal(size=(out_d,), mean=0.0, std=xavier_stddev, requires_grad=True).to(self.device)
        return W, b
        
    def _init_nn_weights(self, m):
        layers_config = self.mf_layers_list[m]
        num_layers = len(layers_config)
        
        weights = []
        biases = []
        
        for l in range(num_layers-1):
            in_d = layers_config[l]
            out_d = layers_config[l+1]
            W,b = self._xavier_init(out_d, in_d)
            weights.append(W)
            biases.append(b)
        #
        
        return weights, biases
    
    def _fidelity_forward_by_params(self, X, weights, biases):
        H = X
        # print(H.shape)
        num_layers = len(weights)
      
        for l in range(num_layers-1):
            in_d = weights[l].shape[1]
            H = torch.add(torch.matmul(H, weights[l].T), biases[l])
            H = self.actfn(H)
            
        #
        
        in_d = weights[-2].shape[1]
        Y = torch.add(torch.matmul(H, weights[-1].T), biases[-1])

        return Y
    
    def forward_by_params(self, X, m, params_mf_weights, params_mf_biases):
        Ym = self._fidelity_forward_by_params(X, params_mf_weights[0], params_mf_biases[0])
        
        for im in range(1, m+1):
            #print(im)
            X_cat = torch.cat([Ym, X], dim=1)
            Ym = self._fidelity_forward_by_params(X_cat, params_mf_weights[im], params_mf_biases[im])
        #
        
        return Ym


    def forward(self, X, m):
        Ym = self.forward_by_params(X, m, self.mf_weights, self.mf_biases)
        return Ym
    