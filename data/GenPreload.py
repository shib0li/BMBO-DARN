import torch
import numpy as np

import time
from datetime import datetime
import os
import sobol_seq
from tqdm.auto import trange 
import fire
import pickle

from utils import GenConfigs
from utils import Misc
from functionals import Functions

FCNN_APPLICATIONS = ['Boston', 'California', 'Sonar']
COVNET_APPPLICATIONS = ['Cifar10']
LDA_APPLICATIONS = ['NewsGroup']
PINN_APPLICATIONS = ['BurgersShock']
PINN_EXT_APPLICATIONS = ['BurgersShockExt']
XGBR_APPLICATIONS = ['Diabetes']

def generate(**kwargs):
    
    gen_config = GenConfigs.GenerateFcnnConfig()
    gen_config._parse(kwargs)

    if gen_config.domain_name in FCNN_APPLICATIONS:
        bounds=tuple([(0,1)]*11 + [(-3,0)])
        dim = 12
    elif gen_config.domain_name in COVNET_APPPLICATIONS:
        bounds=tuple([(0,1)]*24 + [(-3,0)])
        dim = 25
    elif gen_config.domain_name in LDA_APPLICATIONS:
        continuous_bounds = list(((1e-3,1.0),(1e-3,1.0),(0.51,1.0),(-5,-1)))
        binary_bounds = [(0,1)]*12
        bounds = continuous_bounds+binary_bounds
        dim = 16
    #
    elif gen_config.domain_name in PINN_APPLICATIONS:
        bounds = [(0,1)]*12
        dim = 12
    #
    elif gen_config.domain_name in PINN_EXT_APPLICATIONS:
        bounds = [(0,1)]*15
        dim = 15
    elif gen_config.domain_name  in XGBR_APPLICATIONS:
        continuous_bounds = list(((-2,0),(-2,2),(-1,0),(-2,0)))
        binary_bounds = [(0,1)]*8
        bounds = continuous_bounds+binary_bounds
        dim = 12
    else:
        raise Exception('Unrecognized domains group!')
    
    lb = np.array(bounds, ndmin=2)[:, 0]
    ub = np.array(bounds, ndmin=2)[:, 1]
    
    device = torch.device(gen_config.placement)
    if gen_config.sobol:
        X = Misc.generate_sobol_inputs(gen_config.N, dim, lb, ub)
        prefix = 'sobol_raw_'
    else:
        X = Misc.generate_random_inputs(gen_config.N, dim, lb, ub, gen_config.seed)
        prefix = 'uniform_raw_'
    #

    if gen_config.domain_name == 'Boston':
        fcnn_fn = Functions.BostonFunctional(device, gen_config.horizon, mode='generate')
    elif gen_config.domain_name == 'California':
        fcnn_fn = Functions.CaliforniaFunctional(device, gen_config.horizon, mode='generate')
    elif gen_config.domain_name == 'Sonar':
        fcnn_fn = Functions.SonarFunctional(device, gen_config.horizon, mode='generate')
    elif gen_config.domain_name == 'Cifar10':
        fcnn_fn = Functions.Cifar10Functional(device, gen_config.horizon, mode='generate')
    elif gen_config.domain_name == 'NewsGroup':
        fcnn_fn = Functions.NewsGroupFunctional(device, gen_config.horizon, mode='generate')
    elif gen_config.domain_name == 'BurgersShock':
        fcnn_fn = Functions.BurgersShockFunctional(device, gen_config.horizon, mode='generate')
    elif gen_config.domain_name == 'BurgersShockExt':
        fcnn_fn = Functions.BurgersShockExtFunctional(device, gen_config.horizon, mode='generate')
    elif gen_config.domain_name == 'Diabetes':
        fcnn_fn = Functions.DiabetesFunctional(device, gen_config.horizon, mode='generate')
    else:
        raise Exception('Error: unimplemented domain '+ gen_config.domain_name)
    
    dump_path = os.path.join('data', 'buff', gen_config.domain_name)
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    
    pickle_name = prefix+gen_config.domain_name+'.pickle'
    
    Y = []
    for n in trange(gen_config.N):
        hist_scores = fcnn_fn(X[n,:])
        Y.append(hist_scores)
        
        np_Y = np.array(Y)
        #print(np_Y.shape)
        raw = {}
        raw['X'] = X
        raw['Y'] = np_Y
        with open(os.path.join(dump_path, pickle_name), 'wb') as handle:
            pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #    
    
 
if __name__=='__main__':
    fire.Fire(generate)
    
    