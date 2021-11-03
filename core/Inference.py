import torch
import torch.nn as nn
from torch.optim import LBFGS

import hamiltorch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm, trange

from multiprocessing import Pool


# import Dataset as Dataset
# import Hamilton as hamilton
  
class MFHMC:
    """ multi-fidelity inference client with HMC """
    
    def __init__(self, model, sampling, device):
        self.model = model
        self.sampling = sampling
        self.device = device
        
        self.flat_tau = False

        self.prior_params = torch.distributions.Normal(torch.tensor(0.0).to(device),torch.tensor(1.0).to(device))
        self.prior_outputs = torch.distributions.Gamma(torch.tensor(100.0).to(device),torch.tensor(1.0).to(device)) # prior for tau

    def _flatten_net_params(self, params, flat_tau):
        mf_weights = params['mf_weights']
        mf_biases = params['mf_biases']
        mf_log_tau = params['mf_log_tau']
        
        flatten_params_buffer = []

        for m in range(self.model.M):
            weights = mf_weights[m]
            biases = mf_biases[m]
            for W, b in zip(weights, biases):
                flatten_params_buffer.append(torch.flatten(W))
                flatten_params_buffer.append(torch.flatten(b))
            #
        #
        
        if flat_tau:
            for log_tau in mf_log_tau:
                flatten_params_buffer.append(log_tau)
            
        flat_params = torch.cat(flatten_params_buffer)
        #print(flat_params.shape)  
        
        return flat_params
    
    
    def _unflatten_net_params(self, flat_params, flat_tau):
        mf_weights = []
        mf_biases = []
        mf_log_tau = []
        
        i_param = 0
        
        for m in range(self.model.M):
            layers_config = self.model.mf_layers_list[m]
            num_layers = len(layers_config)
            weights = []
            biases = []
            for l in range(num_layers-1):
                in_d = layers_config[l]
                out_d = layers_config[l+1]
                
                size_W = out_d*in_d
                size_b = out_d
                
                W_flat = flat_params[i_param:i_param+size_W]
                b_flat = flat_params[i_param+size_W:i_param+size_W+size_b]
                
                W = W_flat.view([out_d, in_d])
                b = b_flat.view([out_d,])
                
                weights.append(W)
                biases.append(b)
                
                i_param += size_W+size_b
            #
            mf_weights.append(weights)
            mf_biases.append(biases)
        #
        
        if flat_tau:
            mf_log_tau_flat = flat_params[i_param:]
            for m in range(self.model.M):
                mf_log_tau.append(mf_log_tau_flat[m])
            #
            
        #
        
        params = {}
        params['mf_weights'] = mf_weights
        params['mf_biases'] = mf_biases
        params['mf_log_tau'] = mf_log_tau
        
        return params
    
    def _functional_log_prob(self, mf_X_list, mf_y_list, constraint):
        """ retrun a function of mf params given mf data """
        def log_prob_func(flat_params):
            unflatten_params = self._unflatten_net_params(flat_params, flat_tau=self.flat_tau)
            
            llh_list = []

            for m in range(self.model.M):
                X = mf_X_list[m]
                y = mf_y_list[m]
                
                if self.flat_tau:
                    tau = torch.exp(unflatten_params['mf_log_tau'][m])
                else:
                    tau = torch.exp(self.model.mf_log_tau[m])
                #
                
                pred = self.model.forward_by_params(X, m, unflatten_params['mf_weights'], unflatten_params['mf_biases'])
                llh = -0.5*tau*((pred-y)**2).sum()
                #print(llh)
                llh_list.append(llh)
            #
            
#             re_flatten_params = self._flatten_net_params(unflatten_params, flat_tau=False)
            l_prior = self.prior_params.log_prob(flat_params).sum()
    
            if constraint:
                #print('*** with sigmoid constraint ***')
                mf_constraint_list = []
                for m in range(self.model.M-1):
                    y_curr = mf_y_list[m]
                    y_next = mf_y_list[m+1]
                    log_constraint = torch.sum(torch.log(torch.sigmoid(y_next - y_curr.T)))
                    mf_constraint_list.append(log_constraint)
                #
                return sum(llh_list) + l_prior + sum(mf_constraint_list)
            else:
                #print('*** without sigmoid constraint ***')
                return sum(llh_list) + l_prior

        return log_prob_func
    
    def posterior(self, mf_X_list, mf_y_list, constraint):
        mf_X_list_device = []
        mf_y_list_device = []
        
        for m in range(self.model.M):
            mf_X_list_device.append(mf_X_list[m].to(self.device))
            mf_y_list_device.append(mf_y_list[m].to(self.device))
        #

        params = {}
        params['mf_weights'] = self.model.mf_weights
        params['mf_biases'] = self.model.mf_biases
        params['mf_log_tau'] = self.model.mf_log_tau
        
        param_init = self._flatten_net_params(params, flat_tau=self.flat_tau).clone()
        #print(param_init.shape)
        
        log_prob_func = self._functional_log_prob(mf_X_list_device, mf_y_list_device, constraint)
        
        post_samples = hamiltorch.sample(log_prob_func, 
                                    param_init, 
                                    burn=self.sampling['burn'],
                                    num_samples=self.sampling['burn']+self.sampling['Ns'], 
                                    num_steps_per_sample=self.sampling['L'], 
                                    step_size=self.sampling['step_size'],)
        
        return post_samples
    
    def predict(self, mf_X_list, hmc_samples):
        mf_X_list_device = []
        for m in range(self.model.M):
            mf_X_list_device.append(mf_X_list[m].to(self.device))
        #   
        
        mf_pred_list = []
        
        for m in range(self.model.M):
            X = mf_X_list_device[m]
            pred_list = []
            for s in hmc_samples:
                params = self._unflatten_net_params(s, flat_tau=self.flat_tau)
                pred = self.model.forward_by_params(X, m, params['mf_weights'], params['mf_biases'])
                pred_list.append(pred)
            #
            mf_pred_list.append(torch.stack(pred_list))
        #
        
        return mf_pred_list
    
    
    