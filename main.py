import fire
import numpy as np
import torch
import os 
import pickle5 as pickle
import subprocess
from datetime import datetime
from time import time

import hamiltorch
# import logging
# import ExpConfigs as exp_configs
# import Misc as misc
# import data.Dataset as Dataset
# import data.Functions as functions


from utils import ExpConfigs as exp_configs
from utils import Misc as misc
from data import Dataset as Dataset
from functionals import Functions as functions
# from baselines import SMAC3
# from baselines import SMAC4
# from baselines import Hyperband
# from baselines import BOHB
# from baselines import SHPO

# import Hamilton as hamilton

from core import Model
from core import Inference
from core import BayesOpt

from tqdm.auto import trange, tqdm


MF_DNN_APPROACH = ['dnn_mf_bo']
SINGLE_BASED_APPROACH = ['mf_hmc_cs', 'mf_hmc_ucs', 'mf_hmc_fix_low', 'mf_hmc_fix_high']
PAR_HMC_BASED_APPROACH = ['par_hmc_cs', 'par_hmc_ucs']
BATCH_HMC_BASED_APPROACH = ['ratio_batch_hmc_cs', 'ratio_batch_hmc_ucs', 'bound_ratio_batch_hmc_cs']
AO_HMC_BASED_APPROACH = ['ao_batch_hmc_cs', 'ao_batch_hmc_ucs']
RANDOM_APPROACH = ['full_random']
MF_GP_BASED_APPROACH = ['mf_gp_ucb', 'mf_mes', 'par_mf_mes']
SMAC_APPROACH = ['smac', 'gp_kernel', 'hyperband', 'bohb']
MT_APPROACH = ['mtbo']
GP_TS_APPROACH = ['gp_ts']

def create_path(path): 
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        #
        print("Directory '%s' created successfully" % (path))
    except OSError as error:
        print("Directory '%s' can not be created" % (path))
    #

def parse_exp_configs(kwargs):

    configs = {}
   
    opt_config = exp_configs.default_optimization_config()
    opt_config._parse(kwargs)

    domain_config = exp_configs.default_domain_config()
    domain_config._parse(kwargs)
    
    configs['opt_config'] = opt_config
    configs['domain_config'] = domain_config
    
    all_methods = MF_DNN_APPROACH + SINGLE_BASED_APPROACH + PAR_HMC_BASED_APPROACH + BATCH_HMC_BASED_APPROACH +\
                  RANDOM_APPROACH + MF_GP_BASED_APPROACH + SMAC_APPROACH + MT_APPROACH + GP_TS_APPROACH +\
                  AO_HMC_BASED_APPROACH
    
    sampling_methods = MF_DNN_APPROACH + SINGLE_BASED_APPROACH + PAR_HMC_BASED_APPROACH + BATCH_HMC_BASED_APPROACH +\
                  RANDOM_APPROACH + AO_HMC_BASED_APPROACH
    
    if opt_config.algorithm_name not in all_methods:
        raise Exception("ERROR: "+opt_config.algorithm_name+" NOT implemented.")
        
    
    if opt_config.algorithm_name in sampling_methods:
        hmc_sampler_config = exp_configs.default_hmc_sampler_config()
        hmc_sampler_config._parse(kwargs)
        configs['hmc_config'] = hmc_sampler_config
    #
    
    method_config = None
    mf_nn_surrogate_config = None
    if opt_config.algorithm_name in SINGLE_BASED_APPROACH:
        #
        method_config = exp_configs.default_mf_hmc_single_config()
        mf_nn_surrogate_config = exp_configs.default_mf_nn_surrogate_config()
        method_config._parse(kwargs)
        mf_nn_surrogate_config._parse(kwargs)
        #
    elif opt_config.algorithm_name in RANDOM_APPROACH:
        #
        method_config = exp_configs.default_mf_hmc_single_config()
        mf_nn_surrogate_config = exp_configs.default_mf_nn_surrogate_config()
        method_config._parse(kwargs)
        mf_nn_surrogate_config._parse(kwargs)
        #
    if opt_config.algorithm_name in MF_DNN_APPROACH:
        #
        method_config = exp_configs.default_mf_hmc_single_config()
        mf_nn_surrogate_config = exp_configs.default_mf_nn_surrogate_config()
        method_config._parse(kwargs)
        mf_nn_surrogate_config._parse(kwargs)
        #
    elif opt_config.algorithm_name in PAR_HMC_BASED_APPROACH:
        #
        method_config = exp_configs.default_mf_hmc_parallel_config()
        mf_nn_surrogate_config = exp_configs.default_mf_nn_surrogate_config()
        method_config._parse(kwargs)
        mf_nn_surrogate_config._parse(kwargs)
        #
    elif opt_config.algorithm_name in BATCH_HMC_BASED_APPROACH:
        #
        method_config = exp_configs.default_mf_hmc_batch_config()
        mf_nn_surrogate_config = exp_configs.default_mf_nn_surrogate_config()
        method_config._parse(kwargs)
        mf_nn_surrogate_config._parse(kwargs)
        #
    elif opt_config.algorithm_name in AO_HMC_BASED_APPROACH:
        #
        method_config = exp_configs.default_mf_hmc_batch_config()
        mf_nn_surrogate_config = exp_configs.default_mf_nn_surrogate_config()
        method_config._parse(kwargs)
        mf_nn_surrogate_config._parse(kwargs)
        #
#     elif opt_config.algorithm_name in RAND_HMC_BASED_APPROACH:
#         #
#         method_config = exp_configs.default_mf_hmc_random_config()
#         mf_nn_surrogate_config = exp_configs.default_mf_nn_surrogate_config()
#         method_config._parse(kwargs)
#         mf_nn_surrogate_config._parse(kwargs)
#         #
    elif opt_config.algorithm_name in MT_APPROACH:
        method_config = exp_configs.default_mtbo_config()
        method_config._parse(kwargs)
        mf_nn_surrogate_config = exp_configs.default_mf_nn_surrogate_config()
    #
    
    configs['method_config'] = method_config
    configs['mf_nn_surrogate_config'] = mf_nn_surrogate_config
    
    return configs

def experiment_mf_dnn(dataset, method_config, mf_nn_surrogate_config, hmc_sampler_config, horizon, res_path, trial_id):
    
    layers = misc.seq_auto_regressive_layers(
        dataset.in_dim, dataset.out_dim, 
        mf_nn_surrogate_config.hidden_depths,
        mf_nn_surrogate_config.hidden_widths,
    )
    sampling = {
        'step_size':hmc_sampler_config.step_size, 
        'L':hmc_sampler_config.L, 
        'burn':hmc_sampler_config.burn, 
        'Ns':hmc_sampler_config.Ns
    }
    
    res = {}
    res['hist_argm'] = []
    res['hist_argx'] = []
    res['hist_yq'] = []
    res['hist_y_ground'] = []
    res['hist_config'] = []
    res['hist_t_fit'] = []
    res['hist_t_acq'] = []
    res['hist_t_query_m'] = []
    res['hist_t_query_h'] = []
    
    pickle_name = os.path.join(res_path, 'trial'+str(trial_id)+'.pickle')
    
    log_file_name = os.path.join(res_path, 'logs_trial'+str(trial_id)+'.txt')
    logger = open(log_file_name, 'w+')     
    logger.write('===============================================\n')
    logger.write('            Experiment with DNN-MFBO           \n')
    logger.write('===============================================\n')
    logger.write('Experiment start at: '+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
    logger.flush()
    
    exp_t_start = time()
    
    for t in trange(horizon, desc='experiment', leave=True):
        
        t_trial_start = time()
        
        model = Model.SeqAutoRegressive(layers, mf_nn_surrogate_config.activation, 
                               torch.device(mf_nn_surrogate_config.surrogate_placement))
        hamil_opt = BayesOpt.HamilBayesOpt(model, dataset, sampling)
        hmc_samples = hamil_opt.fit(constraint=False)
        
        t_fit = time()

        argm, argx = hamil_opt.info_gain_step(hmc_samples)
        
        t_acq = time()
        
        np_argm = argm.data.cpu().numpy()
        np_argx = argx.data.cpu().numpy()

        #yq, y_ground, success, t_query_m, t_query_h = dataset.add(np_argx, np_argm, scaled_input=True)
        yq, y_ground, success, config, t_query_m, t_query_h = dataset.add_interpret(np_argx, np_argm, scaled_input=True)
        
        t_query = time()
        
        if success:
            res['hist_argm'].append(np_argm)
            res['hist_argx'].append(np_argx)
            res['hist_yq'].append(yq)
            res['hist_y_ground'].append(y_ground)
            res['hist_config'].append(config)
            res['hist_t_fit'].append(t_fit-t_trial_start)
            res['hist_t_acq'].append(t_acq-t_fit)
            res['hist_t_query_m'].append(t_query_m)
            res['hist_t_query_h'].append(t_query_h)   

            logger.write('* Optimization step'+str(t+1)+' finished at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
            logger.write('    - argm = '+str(np_argm)+'\n')
            logger.write('    - argx = '+np.array2string(np_argx)+'\n')
            logger.write('    - t_fit_surrogate = '+str(t_fit-t_trial_start)+' secs\n')
            
            logger.write('    - t_acq = '+str(t_acq-t_fit)+' secs\n')
            logger.write('    - t_query_m = '+str(t_query_m)+' secs\n')
            logger.write('    - t_query_h = '+str(t_query_h)+' secs\n')
            
            logger.write('    - t_query = '+str(t_query-t_acq)+' secs\n')
            logger.write('    - total_elapsed = '+str(t_query-exp_t_start)+' secs\n')
            logger.flush()
        else:
            logger.write('* Optimization step'+str(t+1)+' FAILED at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
            logger.write('    - argm = '+str(np_argm)+'\n')
            logger.write('    - argx = '+np.array2string(np_argx)+'\n')
            logger.flush()
        
        with open(pickle_name, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #

    #
    
    logger.close()


def experiment_single_hmc(dataset, method_config, mf_nn_surrogate_config, hmc_sampler_config, horizon, res_path, trial_id):
    
    layers = misc.full_auto_regressive_layers(
        dataset.in_dim, dataset.out_dim, 
        mf_nn_surrogate_config.hidden_depths,
        mf_nn_surrogate_config.hidden_widths,
    )
    sampling = {
        'step_size':hmc_sampler_config.step_size, 
        'L':hmc_sampler_config.L, 
        'burn':hmc_sampler_config.burn, 
        'Ns':hmc_sampler_config.Ns
    }
    
    res = {}
    res['hist_argm'] = []
    res['hist_argx'] = []
    res['hist_yq'] = []
    res['hist_y_ground'] = []
    res['hist_config'] = []
    res['hist_t_fit'] = []
    res['hist_t_acq'] = []
    res['hist_t_query_m'] = []
    res['hist_t_query_h'] = []
    
    pickle_name = os.path.join(res_path, 'trial'+str(trial_id)+'.pickle')
    
    log_file_name = os.path.join(res_path, 'logs_trial'+str(trial_id)+'.txt')
    logger = open(log_file_name, 'w+')     
    logger.write('===============================================\n')
    logger.write('        Experiment with Single-Constrained     \n')
    logger.write('===============================================\n')
    logger.write('Experiment start at: '+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
    logger.flush()
    
    exp_t_start = time()
    
    for t in trange(horizon, desc='experiment', leave=True):
        
        t_trial_start = time()
        
        model = Model.FullAutoRegressive(layers, mf_nn_surrogate_config.activation, 
                               torch.device(mf_nn_surrogate_config.surrogate_placement))
        hamil_opt = BayesOpt.HamilBayesOpt(model, dataset, sampling)
        hmc_samples = hamil_opt.fit(constraint=method_config.constraint)
        
        t_fit = time()
        
        if method_config.fixed_fidelity is not None:
            argm, argx = hamil_opt.fixed_fidelity_step(hmc_samples, method_config.fixed_fidelity)
            np_argm = argm
        else:
            argm, argx = hamil_opt.info_gain_step(hmc_samples)
            np_argm = argm.data.cpu().numpy()
        #
        np_argx = argx.data.cpu().numpy()
        
        t_acq = time()
 
        #yq, y_ground, success = dataset.add(np_argx, np_argm, scaled_input=True)
        yq, y_ground, success, config, t_query_m, t_query_h = dataset.add_interpret(np_argx, np_argm, scaled_input=True)
        
        t_query = time()
        
        if success:
            res['hist_argm'].append(np_argm)
            res['hist_argx'].append(np_argx)
            res['hist_yq'].append(yq)
            res['hist_y_ground'].append(y_ground)
            res['hist_config'].append(config)
            res['hist_t_fit'].append(t_fit-t_trial_start)
            res['hist_t_acq'].append(t_acq-t_fit)
            res['hist_t_query_m'].append(t_query_m)
            res['hist_t_query_h'].append(t_query_h) 
            
            logger.write('* Optimization step'+str(t+1)+' finished at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
            logger.write('    - argm = '+str(np_argm)+'\n')
            logger.write('    - argx = '+np.array2string(np_argx)+'\n')
            logger.write('    - t_fit_surrogate = '+str(t_fit-t_trial_start)+' secs\n')
            
            logger.write('    - t_acq = '+str(t_acq-t_fit)+' secs\n')
            logger.write('    - t_query_m = '+str(t_query_m)+' secs\n')
            logger.write('    - t_query_h = '+str(t_query_h)+' secs\n')
            
            logger.write('    - t_query = '+str(t_query-t_acq)+' secs\n')
            logger.write('    - total_elapsed = '+str(t_query-exp_t_start)+' secs\n')
            logger.flush()
        else:
            logger.write('* Optimization step'+str(t+1)+' FAILED at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
            logger.write('    - argm = '+str(np_argm)+'\n')
            logger.write('    - argx = '+np.array2string(np_argx)+'\n')
            logger.flush()
            
        
        with open(pickle_name, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
    #
    
    logger.close()
    
    
    
def experiment_random(dataset, method_config, mf_nn_surrogate_config, hmc_sampler_config, horizon, res_path, trial_id):
    
    #print('Random searching heuristics')
    layers = misc.full_auto_regressive_layers(
        dataset.in_dim, dataset.out_dim, 
        mf_nn_surrogate_config.hidden_depths,
        mf_nn_surrogate_config.hidden_widths,
    )
    sampling = {
        'step_size':hmc_sampler_config.step_size, 
        'L':hmc_sampler_config.L, 
        'burn':hmc_sampler_config.burn, 
        'Ns':hmc_sampler_config.Ns
    }
    
    res = {}
    res['hist_argm'] = []
    res['hist_argx'] = []
    res['hist_yq'] = []
    res['hist_y_ground'] = []
    res['hist_config'] = []
    res['hist_t_fit'] = []
    res['hist_t_acq'] = []
    res['hist_t_query_m'] = []
    res['hist_t_query_h'] = []
    
    pickle_name = os.path.join(res_path, 'trial'+str(trial_id)+'.pickle')
    
    log_file_name = os.path.join(res_path, 'logs_trial'+str(trial_id)+'.txt')
    logger = open(log_file_name, 'w+')     
    logger.write('===============================================\n')
    logger.write('          Experiment with Full-Random          \n')
    logger.write('===============================================\n')
    logger.write('Experiment start at: '+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
    logger.flush()
    
    exp_t_start = time()
    
    for t in trange(horizon, desc='experiment', leave=True):
        
        t_trial_start = time()
        
        model = Model.FullAutoRegressive(layers, mf_nn_surrogate_config.activation, 
                               torch.device(mf_nn_surrogate_config.surrogate_placement))
        hamil_opt = BayesOpt.HamilBayesOpt(model, dataset, sampling)
        hmc_samples = hamil_opt.fit(constraint=method_config.constraint)
        
        t_fit = time()
        
        np_argm = np.random.randint(0, model.M)
        np_argx = misc.generate_random_inputs(1, dataset.in_dim, dataset.lb, dataset.ub, seed=np.random.randint(0,100000))
        
        t_acq = time()
        
        #yq, y_ground, success, t_query_m, t_query_h = dataset.add(np_argx, np_argm, scaled_input=False)
        yq, y_ground, success, config, t_query_m, t_query_h = dataset.add_interpret(np_argx, np_argm, scaled_input=False)
        
        t_query = time()
        
        if success:
            res['hist_argm'].append(np_argm)
            res['hist_argx'].append(np_argx)
            res['hist_yq'].append(yq)
            res['hist_y_ground'].append(y_ground)
            res['hist_config'].append(config)
            res['hist_t_fit'].append(t_fit-t_trial_start)
            res['hist_t_acq'].append(t_acq-t_fit)
            res['hist_t_query_m'].append(t_query_m)
            res['hist_t_query_h'].append(t_query_h) 

            logger.write('* Optimization step'+str(t+1)+' finished at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
            logger.write('    - argm = '+str(np_argm)+'\n')
            logger.write('    - argx = '+np.array2string(np_argx)+'\n')
            logger.write('    - t_fit_surrogate = '+str(t_fit-t_trial_start)+' secs\n')
            
            logger.write('    - t_acq = '+str(t_acq-t_fit)+' secs\n')
            logger.write('    - t_query_m = '+str(t_query_m)+' secs\n')
            logger.write('    - t_query_h = '+str(t_query_h)+' secs\n')
            
            logger.write('    - t_query = '+str(t_query-t_acq)+' secs\n')
            logger.write('    - total_elapsed = '+str(t_query-exp_t_start)+' secs\n')
            logger.flush()
        else:
            logger.write('* Optimization step'+str(t+1)+' FAILED at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
            logger.write('    - argm = '+str(np_argm)+'\n')
            logger.write('    - argx = '+np.array2string(np_argx)+'\n')
            logger.flush()
        
        with open(pickle_name, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
    #
    logger.close()
    
    
def experiment_par_hmc(dataset, method_config, mf_nn_surrogate_config, hmc_sampler_config, horizon, res_path, trial_id):
    
    layers = misc.full_auto_regressive_layers(
        dataset.in_dim, dataset.out_dim, 
        mf_nn_surrogate_config.hidden_depths,
        mf_nn_surrogate_config.hidden_widths,
    )
    sampling = {
        'step_size':hmc_sampler_config.step_size, 
        'L':hmc_sampler_config.L, 
        'burn':hmc_sampler_config.burn, 
        'Ns':hmc_sampler_config.Ns
    }
    
    res = {}
    res['hist_argm'] = []
    res['hist_argx'] = []
    res['hist_yq'] = []
    res['hist_y_ground'] = []
    res['hist_config'] = []
    res['hist_t_fit'] = []
    res['hist_t_acq'] = []
    res['hist_t_query_m'] = []
    res['hist_t_query_h'] = []
    
    pickle_name = os.path.join(res_path, 'trial'+str(trial_id)+'.pickle')
    
    log_file_name = os.path.join(res_path, 'logs_trial'+str(trial_id)+'.txt')
    logger = open(log_file_name, 'w+')     
    logger.write('===============================================\n')
    logger.write('          Experiment with Parallel HMC         \n')
    logger.write('===============================================\n')
    logger.write('Experiment start at: '+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
    logger.flush()
    
    exp_t_start = time()
    
    for t in trange(horizon, desc='experiment', leave=True):
        
        t_trial_start = time()
        
        model = Model.FullAutoRegressive(layers, mf_nn_surrogate_config.activation, 
                               torch.device(mf_nn_surrogate_config.surrogate_placement))
        hamil_opt = BayesOpt.HamilBayesOpt(model, dataset, sampling)
        hmc_samples = hamil_opt.fit(constraint=method_config.constraint)
        
        t_fit = time()
      
        pool_argm, pool_argx = hamil_opt.pseudo_par_step(hmc_samples, method_config.n_threads)
        
        t_acq = time()
        
        np_pool_argx = []
        np_pool_argm = []
        for argm, argx in zip(pool_argm, pool_argx):
            np_pool_argx.append(argx.data.cpu().numpy())
            np_pool_argm.append(argm.data.cpu().numpy())
        #
        #pool_yq, pool_y_ground, pool_success = dataset.add_pool(np_pool_argx, np_pool_argm, scaled_input=True)
        pool_yq, pool_y_ground, pool_success, pool_config, t_query_m_pool, t_query_h_pool =\
            dataset.add_pool_interpret(np_pool_argx, np_pool_argm, scaled_input=True)
        
        #print(pool_config)
        
        t_query = time()
        
        logger.write('* Optimization step'+str(t+1)+' finished at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
        
        pool_size = len(pool_success)
        
        res['hist_t_fit'].append(t_fit-t_trial_start)
        res['hist_t_acq'].append(t_acq-t_fit)
        for i in range(pool_size):
            success = pool_success[i]
            if success:
                res['hist_argm'].append(np_pool_argm[i])
                res['hist_argx'].append(np_pool_argx[i])
                res['hist_yq'].append(pool_yq[i])
                res['hist_y_ground'].append(pool_y_ground[i])
                res['hist_config'].append(pool_config[i])
                res['hist_t_query_m'].append(t_query_m_pool[i])
                res['hist_t_query_h'].append(t_query_h_pool[i])
                logger.write('    - Success add argm = '+np.array2string(np.array(np_pool_argm[i]))+'\n')
                logger.write('    - Success add argx = '+np.array2string(np.array(np_pool_argx[i]))+'\n')
                logger.write('    * t_query_m = '+str(t_query_m_pool[i])+' secs\n')
                logger.write('    * t_query_h = '+str(t_query_h_pool[i])+' secs\n')
            else:
                logger.write('    - Failed add argm = '+np.array2string(np.array(np_pool_argm[i]))+'\n')
                logger.write('    - Failed add argx = '+np.array2string(np.array(np_pool_argx[i]))+'\n')
            #
        #

        logger.write('    - t_fit_surrogate = '+str(t_fit-t_trial_start)+' secs\n')
        logger.write('    - t_acq = '+str(t_acq-t_fit)+' secs\n')
        logger.write('    - t_query = '+str(t_query-t_acq)+' secs\n')
        logger.write('    - total_elapsed = '+str(t_query-exp_t_start)+' secs\n')
        logger.flush()

        with open(pickle_name, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
    #
    logger.close()
        
    
def experiment_batch_hmc(dataset, method_config, mf_nn_surrogate_config, hmc_sampler_config, horizon, res_path, trial_id):
    
    layers = misc.full_auto_regressive_layers(
        dataset.in_dim, dataset.out_dim, 
        mf_nn_surrogate_config.hidden_depths,
        mf_nn_surrogate_config.hidden_widths,
    )
    sampling = {
        'step_size':hmc_sampler_config.step_size, 
        'L':hmc_sampler_config.L, 
        'burn':hmc_sampler_config.burn, 
        'Ns':hmc_sampler_config.Ns
    }
    
    res = {}
    res['hist_argm'] = []
    res['hist_argx'] = []
    res['hist_yq'] = []
    res['hist_y_ground'] = []
    res['hist_config'] = []
    res['hist_t_fit'] = []
    res['hist_t_acq'] = []
    res['hist_t_query_m'] = []
    res['hist_t_query_h'] = []
    
    pickle_name = os.path.join(res_path, 'trial'+str(trial_id)+'.pickle')
    
    log_file_name = os.path.join(res_path, 'logs_trial'+str(trial_id)+'.txt')
    logger = open(log_file_name, 'w+')     
    logger.write('===============================================\n')
    logger.write('           Experiment with Ratio Batch         \n')
    logger.write('===============================================\n')
    logger.write('Experiment start at: '+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
    logger.flush()
    
    exp_t_start = time()
    

    for t in trange(horizon, desc='experiment', leave=True):
        
        t_trial_start = time()
        
        model = Model.FullAutoRegressive(layers, mf_nn_surrogate_config.activation, 
                               torch.device(mf_nn_surrogate_config.surrogate_placement))
        hamil_opt = BayesOpt.HamilBayesOpt(model, dataset, sampling)
        hmc_samples = hamil_opt.fit(constraint=method_config.constraint)
        
        t_fit = time()
        
        if method_config.batch_mode == 'ratio':
            pool_argm, pool_argx, _ = hamil_opt.ratio_batch_step(hmc_samples, method_config.batch_size)
#         elif method_config.batch_mode == 'bound_ratio':
#             #print('***** New bounded query *****')
#             pool_argm, pool_argx, _ = hamil_opt.bound_ratio_batch_step(hmc_samples, method_config.batch_size)
        elif method_config.batch_mode == 'linear':
            pool_argm, pool_argx, _ = hamil_opt.linear_batch_step(hmc_samples, method_config.batch_size, method_config.beta)
        #
        
        t_acq = time()

        np_pool_argx = []
        np_pool_argm = []
        for argm, argx in zip(pool_argm, pool_argx):
            np_pool_argx.append(argx.data.cpu().numpy())
            np_pool_argm.append(argm.data.cpu().numpy())
        #
        #pool_yq, pool_y_ground, pool_success = dataset.add_pool(np_pool_argx, np_pool_argm, scaled_input=True)
        pool_yq, pool_y_ground, pool_success, pool_config, t_query_m_pool, t_query_h_pool =\
            dataset.add_pool_interpret(np_pool_argx, np_pool_argm, scaled_input=True)
        
        t_query = time()
        
        logger.write('* Optimization step'+str(t+1)+' finished at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
        
        pool_size = len(pool_success)
        
        res['hist_t_fit'].append(t_fit-t_trial_start)
        res['hist_t_acq'].append(t_acq-t_fit)
        
        for i in range(pool_size):
            success = pool_success[i]
            if success:
                res['hist_argm'].append(np_pool_argm[i])
                res['hist_argx'].append(np_pool_argx[i])
                res['hist_yq'].append(pool_yq[i])
                res['hist_y_ground'].append(pool_y_ground[i])
                res['hist_config'].append(pool_config[i])
                res['hist_t_query_m'].append(t_query_m_pool[i])
                res['hist_t_query_h'].append(t_query_h_pool[i])
                logger.write('    - Success add argm = '+np.array2string(np.array(np_pool_argm[i]))+'\n')
                logger.write('    - Success add argx = '+np.array2string(np.array(np_pool_argx[i]))+'\n')
                logger.write('    * t_query_m = '+str(t_query_m_pool[i])+' secs\n')
                logger.write('    * t_query_h = '+str(t_query_h_pool[i])+' secs\n')
            else:
                logger.write('    - Failed add argm = '+np.array2string(np.array(np_pool_argm[i]))+'\n')
                logger.write('    - Failed add argx = '+np.array2string(np.array(np_pool_argx[i]))+'\n')
            #
        #


        logger.write('    - t_fit_surrogate = '+str(t_fit-t_trial_start)+' secs\n')
        logger.write('    - t_acq = '+str(t_acq-t_fit)+' secs\n')
        logger.write('    - t_query = '+str(t_query-t_acq)+' secs\n')
        logger.write('    - total_elapsed = '+str(t_query-exp_t_start)+' secs\n')
        logger.flush()

        with open(pickle_name, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
    #
    logger.close()
    
def experiment_ao_hmc(dataset, method_config, mf_nn_surrogate_config, hmc_sampler_config, horizon, res_path, trial_id):
    
    layers = misc.full_auto_regressive_layers(
        dataset.in_dim, dataset.out_dim, 
        mf_nn_surrogate_config.hidden_depths,
        mf_nn_surrogate_config.hidden_widths,
    )
    sampling = {
        'step_size':hmc_sampler_config.step_size, 
        'L':hmc_sampler_config.L, 
        'burn':hmc_sampler_config.burn, 
        'Ns':hmc_sampler_config.Ns
    }
    
    res = {}
    res['hist_argm'] = []
    res['hist_argx'] = []
    res['hist_yq'] = []
    res['hist_y_ground'] = []
    res['hist_config'] = []
    res['hist_t_fit'] = []
    res['hist_t_acq'] = []
    res['hist_t_query_m'] = []
    res['hist_t_query_h'] = []
    
    pickle_name = os.path.join(res_path, 'trial'+str(trial_id)+'.pickle')
    
    log_file_name = os.path.join(res_path, 'logs_trial'+str(trial_id)+'.txt')
    logger = open(log_file_name, 'w+')     
    logger.write('===============================================\n')
    logger.write('           Experiment with AO Batch         \n')
    logger.write('===============================================\n')
    logger.write('Experiment start at: '+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
    logger.flush()
    
    exp_t_start = time()
    
    
    for t in trange(horizon, desc='experiment', leave=True):
        
        t_trial_start = time()
        
        model = Model.FullAutoRegressive(layers, mf_nn_surrogate_config.activation, 
                               torch.device(mf_nn_surrogate_config.surrogate_placement))
        hamil_opt = BayesOpt.HamilBayesOpt(model, dataset, sampling)
        hmc_samples = hamil_opt.fit(constraint=method_config.constraint)
        
        t_fit = time()

        pool_argm, pool_argx, _ = hamil_opt.ao_ratio_batch_step(hmc_samples, method_config.batch_size, alters=5)

        t_acq = time()

        np_pool_argx = []
        np_pool_argm = []
        for argm, argx in zip(pool_argm, pool_argx):
            np_pool_argx.append(argx.data.cpu().numpy())
            np_pool_argm.append(argm.data.cpu().numpy())
        #
        #pool_yq, pool_y_ground, pool_success = dataset.add_pool(np_pool_argx, np_pool_argm, scaled_input=True)
        pool_yq, pool_y_ground, pool_success, pool_config, t_query_m_pool, t_query_h_pool =\
            dataset.add_pool_interpret(np_pool_argx, np_pool_argm, scaled_input=True)
        
        t_query = time()
        
        logger.write('* Optimization step'+str(t+1)+' finished at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
        
        pool_size = len(pool_success)
        
        res['hist_t_fit'].append(t_fit-t_trial_start)
        res['hist_t_acq'].append(t_acq-t_fit)
        
        for i in range(pool_size):
            success = pool_success[i]
            if success:
                res['hist_argm'].append(np_pool_argm[i])
                res['hist_argx'].append(np_pool_argx[i])
                res['hist_yq'].append(pool_yq[i])
                res['hist_y_ground'].append(pool_y_ground[i])
                res['hist_config'].append(pool_config[i])
                res['hist_t_query_m'].append(t_query_m_pool[i])
                res['hist_t_query_h'].append(t_query_h_pool[i])
                logger.write('    - Success add argm = '+np.array2string(np.array(np_pool_argm[i]))+'\n')
                logger.write('    - Success add argx = '+np.array2string(np.array(np_pool_argx[i]))+'\n')
                logger.write('    * t_query_m = '+str(t_query_m_pool[i])+' secs\n')
                logger.write('    * t_query_h = '+str(t_query_h_pool[i])+' secs\n')
            else:
                logger.write('    - Failed add argm = '+np.array2string(np.array(np_pool_argm[i]))+'\n')
                logger.write('    - Failed add argx = '+np.array2string(np.array(np_pool_argx[i]))+'\n')
            #
        #


        logger.write('    - t_fit_surrogate = '+str(t_fit-t_trial_start)+' secs\n')
        logger.write('    - t_acq = '+str(t_acq-t_fit)+' secs\n')
        logger.write('    - t_query = '+str(t_query-t_acq)+' secs\n')
        logger.write('    - total_elapsed = '+str(t_query-exp_t_start)+' secs\n')
        logger.flush()

        with open(pickle_name, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
    #
    logger.close()
    
        
def experiment_mf_mes(domain_name, horizon, num_trials, num_inits, init_i_trial, penalty):
    
    Ninit="".join([str(e)+',' for e in num_inits])[:-1]
    costs = "".join([str(e)+',' for e in penalty])[:-1]
    
    os.chdir('baselines/MF-MES/experiments')
    
    subprocess.run(["python", "customized_bo_runner.py", "-m", "MFMES_RFM", "-d", domain_name, "-t", str(horizon),
                    "-c", "2000000", "-i", Ninit, "-s", costs, "-T", str(num_trials), "-f", str(init_i_trial)])
    os.chdir('../../')
    
def experiment_par_mf_mes(domain_name, horizon, num_trials, num_inits, init_i_trial, penalty):
    
    Ninit="".join([str(e)+',' for e in num_inits])[:-1]
    costs = "".join([str(e)+',' for e in penalty])[:-1]
    
    os.chdir('baselines/MF-MES/experiments')
    
    subprocess.run(["python", "customized_parallel_bayesopt_exp.py", "-m", "Parallel_MFMES_RFM", "-d", domain_name, "-t", str(horizon),
                    "-c", "2000000", "-i", Ninit, "-s", costs, "-T", str(num_trials), "-f", str(init_i_trial)])
    os.chdir('../../')
    
def experiment_mf_gp_ucb(domain_name, horizon, num_trials, num_inits, init_i_trial, penalty):
    
    Ninit="".join([str(e)+',' for e in num_inits])[:-1]
    costs = "".join([str(e)+',' for e in penalty])[:-1]
    
    os.chdir('baselines/MF-MES/experiments')
    
    subprocess.run(["python", "customized_bo_runner.py", "-m", "BOCA", "-d", domain_name, "-t", str(horizon), 
                    "-f", str(init_i_trial),
                    "-c", "2000000", "-i", Ninit, "-s", costs, "-T", str(num_trials), "-f", str(init_i_trial)])
    os.chdir('../../')
    
def experiment_smac(domain_name, horizon, num_trials, init_i_trial, penalty, placement):
    for t in range(num_trials):
        trial = t + init_i_trial
        
        res_path = os.path.join('results', domain_name, 'smac', 'trial'+str(trial))
        try:
            if not os.path.exists(res_path):
                os.makedirs(res_path, exist_ok=True)
            #
            print("Directory '%s' created successfully" % (res_path))
        except OSError as error:
            print("Directory '%s' can not be created" % (res_path))
        #

        client = SMAC3.Client(domain_name, penalty[-1], horizon, res_path, torch.device(placement))
        client.minimize()
        
def experiment_gp_kernel(domain_name, horizon, num_trials, init_i_trial, penalty, placement):
    for t in range(num_trials):
        trial = t + init_i_trial
        
        res_path = os.path.join('results', domain_name, 'gp_kernel', 'trial'+str(trial))
        try:
            if not os.path.exists(res_path):
                os.makedirs(res_path, exist_ok=True)
            #
            print("Directory '%s' created successfully" % (res_path))
        except OSError as error:
            print("Directory '%s' can not be created" % (res_path))
        #

        client = SMAC4.Client(domain_name, penalty[-1], horizon, res_path, torch.device(placement))
        client.minimize()

def experiment_hyperband(domain_name, horizon, num_trials, init_i_trial, penalty, placement):
    for t in range(num_trials):
        trial = t + init_i_trial
        
        res_path = os.path.join('results', domain_name, 'hyperband', 'trial'+str(trial))
        try:
            if not os.path.exists(res_path):
                os.makedirs(res_path, exist_ok=True)
            #
            print("Directory '%s' created successfully" % (res_path))
        except OSError as error:
            print("Directory '%s' can not be created" % (res_path))
        #

        client = Hyperband.Client(domain_name, penalty[-1], horizon, res_path, torch.device(placement))
        client.minimize()

        
def experiment_bohb(domain_name, horizon, num_trials, init_i_trial, penalty, placement):
    for t in range(num_trials):
        trial = t + init_i_trial
        
        res_path = os.path.join('results', domain_name, 'bohb', 'trial'+str(trial))
        try:
            if not os.path.exists(res_path):
                os.makedirs(res_path, exist_ok=True)
            #
            print("Directory '%s' created successfully" % (res_path))
        except OSError as error:
            print("Directory '%s' can not be created" % (res_path))
        #

        client = BOHB.Client(domain_name, penalty[-1], horizon, res_path, torch.device(placement))
        client.minimize()
    
    
def experiment_multitask_bo(dataset, method_config, horizon, res_path, trial_id):
    
    res = {}
    res['hist_argm'] = []
    res['hist_argx'] = []
    res['hist_yq'] = []
    res['hist_y_ground'] = []
    res['hist_config'] = []
    res['hist_t_fit'] = []
    res['hist_t_acq'] = []
    res['hist_t_query_m'] = []
    res['hist_t_query_h'] = []
    
    pickle_name = os.path.join(res_path, 'trial'+str(trial_id)+'.pickle')
    
    log_file_name = os.path.join(res_path, 'logs_trial'+str(trial_id)+'.txt')
    logger = open(log_file_name, 'w+')     
    logger.write('===============================================\n')
    logger.write('          Experiment with Multitask BO         \n')
    logger.write('===============================================\n')
    logger.write('Experiment start at: '+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
    logger.flush()

    base_dim = method_config.base_dim
    base_hidden_depth = method_config.base_hidden_depth
    base_hidden_width = method_config.base_hidden_width
    surrogate_device = torch.device(method_config.surrogate_placement)
    
    exp_t_start = time()

    for t in trange(horizon, desc='experiment', leave=True):
        
        t_trial_start = time()
        
        hpo = SHPO.HPO(dataset, base_dim, base_hidden_depth, base_hidden_width, surrogate_device)
        
        t_fit = time()
        
        np_argm, np_argx = hpo.step(dataset.penalty)
        
        t_acq = time()

        #yq, y_ground, success = dataset.add(np_argx, np_argm, scaled_input=True)
        yq, y_ground, success, config, t_query_m, t_query_h = dataset.add_interpret(np_argx, np_argm, scaled_input=False)
        
        t_query = time()
        
        if success:
            res['hist_argm'].append(np_argm)
            res['hist_argx'].append(np_argx)
            res['hist_yq'].append(yq)
            res['hist_y_ground'].append(y_ground)
            res['hist_config'].append(config)
            res['hist_t_fit'].append(t_fit-t_trial_start)
            res['hist_t_acq'].append(t_acq-t_fit)
            res['hist_t_query_m'].append(t_query_m)
            res['hist_t_query_h'].append(t_query_h) 

            logger.write('* Optimization step'+str(t+1)+' finished at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
            logger.write('    - argm = '+str(np_argm)+'\n')
            logger.write('    - argx = '+np.array2string(np_argx)+'\n')
            logger.write('    - t_fit_surrogate = '+str(t_fit-t_trial_start)+' secs\n')
            logger.write('    - t_acq = '+str(t_acq-t_fit)+' secs\n')
            
            logger.write('    - t_query_m = '+str(t_query_m)+' secs\n')
            logger.write('    - t_query_h = '+str(t_query_h)+' secs\n')
            
            logger.write('    - t_query = '+str(t_query-t_acq)+' secs\n')
            logger.write('    - total_elapsed = '+str(t_query-exp_t_start)+' secs\n')
            logger.flush()
        else:
            logger.write('* Optimization step'+str(t+1)+' FAILED at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+'\n')
            logger.write('    - argm = '+str(np_argm)+'\n')
            logger.write('    - argx = '+np.array2string(np_argx)+'\n')
            logger.flush()

        with open(pickle_name, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
    #
    
    logger.close()
    
def experiment_gp_ts(domain_name):
    
    os.chdir('baselines/gp-parallel-ts')
#     print('run_gp_tes')
    
    runner_name = 'run_gpts_'+domain_name+'.py'
    
#     print(runner_name)
    
#     if domain_name == 'Diabetes':
    subprocess.run(["python", runner_name])
    #
    os.chdir('../../')
    

def evaluation(**kwargs):
    configs = parse_exp_configs(kwargs)
    
    domain_name = configs['domain_config'].domain_name
    domain_penalty = configs['domain_config'].penalty
    domain_placement = configs['domain_config'].domain_placement
    domain_Ninits = configs['domain_config'].num_inits
    
    preload = os.path.join('data','preload',domain_name+'.pickle')

    mf_func = functions.MfFunc(domain_name, domain_penalty, torch.device(domain_placement))
    
    #dataset = Dataset.MfData(mf_func, preload)

    if configs['opt_config'].algorithm_name in MF_DNN_APPROACH:
        
        init_i_trial = configs['opt_config'].init_i_trial

        for t in range(configs['opt_config'].num_trials):
            dataset = Dataset.MfData(mf_func, preload)
            tid = t + init_i_trial
            res_path = os.path.join('results', domain_name, configs['opt_config'].algorithm_name, 'trial'+str(tid))
            create_path(res_path)
            experiment_mf_dnn(
                dataset, 
                configs['method_config'],
                configs['mf_nn_surrogate_config'], 
                configs['hmc_config'],
                configs['opt_config'].horizon,
                res_path,
                tid,
            )
        #

    elif configs['opt_config'].algorithm_name in SINGLE_BASED_APPROACH:


        init_i_trial = configs['opt_config'].init_i_trial
        
        for t in range(configs['opt_config'].num_trials):
            dataset = Dataset.MfData(mf_func, preload)
            tid = t + init_i_trial
            res_path = os.path.join('results', domain_name, configs['opt_config'].algorithm_name, 'trial'+str(tid))
            create_path(res_path)
            experiment_single_hmc(
                dataset, 
                configs['method_config'],
                configs['mf_nn_surrogate_config'], 
                configs['hmc_config'],
                configs['opt_config'].horizon,
                res_path,
                tid,
            )
        #
        
    elif configs['opt_config'].algorithm_name in RANDOM_APPROACH:


        init_i_trial = configs['opt_config'].init_i_trial
        
        for t in range(configs['opt_config'].num_trials):
            dataset = Dataset.MfData(mf_func, preload)
            tid = t + init_i_trial
            res_path = os.path.join('results', domain_name, configs['opt_config'].algorithm_name, 'trial'+str(tid))
            create_path(res_path)
            experiment_random(
                dataset, 
                configs['method_config'],
                configs['mf_nn_surrogate_config'], 
                configs['hmc_config'],
                configs['opt_config'].horizon,
                res_path,
                tid,
            )
        #
        
    elif configs['opt_config'].algorithm_name in PAR_HMC_BASED_APPROACH:

        init_i_trial = configs['opt_config'].init_i_trial
        
        for t in range(configs['opt_config'].num_trials):
            dataset = Dataset.MfData(mf_func, preload)
            tid = t + init_i_trial
            res_path = os.path.join('results', domain_name, configs['opt_config'].algorithm_name, 'trial'+str(tid))
            create_path(res_path)
            experiment_par_hmc(
                dataset, 
                configs['method_config'],
                configs['mf_nn_surrogate_config'], 
                configs['hmc_config'],
                configs['opt_config'].horizon,
                res_path,
                tid,
            )
        #
        
    elif configs['opt_config'].algorithm_name in BATCH_HMC_BASED_APPROACH:


        init_i_trial = configs['opt_config'].init_i_trial
    
#         if configs['method_config'].batch_mode=='linear':
#             beta = configs['method_config'].beta
#             res_path = os.path.join('results', domain_name, configs['opt_config'].algorithm_name+'_'+str(beta), 'init'+str(init_i_trial))
#         elif configs['method_config'].batch_mode=='ratio':
#             res_path = os.path.join('results', domain_name, configs['opt_config'].algorithm_name, 'init'+str(init_i_trial))
#         else:
#             raise Exception('Error: invalid batch mode')
#         #
        
#         try:
#             if not os.path.exists(res_path):
#                 os.makedirs(res_path, exist_ok=True)
#             #
#             print("Directory '%s' created successfully" % (res_path))
#         except OSError as error:
#             print("Directory '%s' can not be created" % (res_path))
#         #
        
        for t in range(configs['opt_config'].num_trials):
            dataset = Dataset.MfData(mf_func, preload)
            tid = t + init_i_trial
            res_path = os.path.join('results', domain_name, configs['opt_config'].algorithm_name, 'trial'+str(tid))
            create_path(res_path)
            experiment_batch_hmc(
                dataset, 
                configs['method_config'],
                configs['mf_nn_surrogate_config'], 
                configs['hmc_config'],
                configs['opt_config'].horizon,
                res_path,
                tid,
            )
         #
        
    elif configs['opt_config'].algorithm_name in AO_HMC_BASED_APPROACH:


        init_i_trial = configs['opt_config'].init_i_trial

        for t in range(configs['opt_config'].num_trials):
            dataset = Dataset.MfData(mf_func, preload)
            tid = t + init_i_trial
            res_path = os.path.join('results', domain_name, configs['opt_config'].algorithm_name, 'trial'+str(tid))
            create_path(res_path)
            experiment_ao_hmc(
                dataset, 
                configs['method_config'],
                configs['mf_nn_surrogate_config'], 
                configs['hmc_config'],
                configs['opt_config'].horizon,
                res_path,
                tid,
            )
         #
        
    elif configs['opt_config'].algorithm_name == 'mf_gp_ucb':
        experiment_mf_gp_ucb(
            domain_name, 
            configs['opt_config'].horizon, 
            configs['opt_config'].num_trials, 
            domain_Ninits, 
            configs['opt_config'].init_i_trial,
            domain_penalty
        )
    elif configs['opt_config'].algorithm_name == 'mf_mes':
        experiment_mf_mes(
            domain_name, 
            configs['opt_config'].horizon, 
            configs['opt_config'].num_trials, 
            domain_Ninits, 
            configs['opt_config'].init_i_trial,
            domain_penalty
        )
    elif configs['opt_config'].algorithm_name == 'par_mf_mes':
        #print('********** New implemnted par mfmes **********')
        experiment_par_mf_mes(
            domain_name, 
            configs['opt_config'].horizon, 
            configs['opt_config'].num_trials, 
            domain_Ninits, 
            configs['opt_config'].init_i_trial,
            domain_penalty
        )
    elif configs['opt_config'].algorithm_name == 'smac':
        experiment_smac(
            domain_name, 
            configs['opt_config'].horizon, 
            configs['opt_config'].num_trials, 
            configs['opt_config'].init_i_trial, 
            domain_penalty,
            domain_placement
        )
    elif configs['opt_config'].algorithm_name == 'gp_kernel':
        experiment_gp_kernel(
            domain_name, 
            configs['opt_config'].horizon, 
            configs['opt_config'].num_trials, 
            configs['opt_config'].init_i_trial, 
            domain_penalty,
            domain_placement
        )
    elif configs['opt_config'].algorithm_name == 'hyperband':
        experiment_hyperband(
            domain_name, 
            configs['opt_config'].horizon, 
            configs['opt_config'].num_trials, 
            configs['opt_config'].init_i_trial, 
            domain_penalty,
            domain_placement
        )
    elif configs['opt_config'].algorithm_name == 'bohb':
        experiment_bohb(
            domain_name, 
            configs['opt_config'].horizon, 
            configs['opt_config'].num_trials, 
            configs['opt_config'].init_i_trial, 
            domain_penalty,
            domain_placement
        )
    elif configs['opt_config'].algorithm_name in MT_APPROACH:


        init_i_trial = configs['opt_config'].init_i_trial
        
        for t in range(configs['opt_config'].num_trials):
            dataset = Dataset.MfData(mf_func, preload)
            tid = t + init_i_trial
            res_path = os.path.join('results', domain_name, configs['opt_config'].algorithm_name, 'trial'+str(tid))
            create_path(res_path)
            experiment_multitask_bo(
                dataset, 
                configs['method_config'], 
                configs['opt_config'].horizon,
                res_path,
                tid,
            )
        #
    elif configs['opt_config'].algorithm_name == 'gp_ts':
        experiment_gp_ts(
            domain_name
        )
        
        
    

if __name__=='__main__':
    fire.Fire(evaluation)