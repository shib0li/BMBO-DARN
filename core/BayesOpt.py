import torch
import torch.nn as nn
from torch.optim import LBFGS

import hamiltorch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm, trange

# from multiprocessing import Pool
from core import Inference


def init_random_inputs_w_bounds(np_lb, np_ub, N=1, torch_tensor=True, seed=None, device=torch.device('cpu')):
    dim = np_lb.size
    scale = (np_ub - np_lb).reshape([1,-1])

    rand_state = np.random.get_state()
    try:
        if seed is None:
            from datetime import datetime
            seed=int(datetime.utcnow().timestamp()*10000%(2**32))
        #
        np.random.seed(seed)
        uniform_noise = np.random.uniform(0,1,size=[N,dim])
    except:
        print('Errors occured when generating random noise...')
    finally:
        np.random.set_state(rand_state)

    X_init = uniform_noise*scale + np_lb

    if torch_tensor:
        X_init = torch.tensor(X_init, requires_grad=True, dtype=torch.float).to(device)

    return X_init
    
class HamilBayesOpt:
    def __init__(self, model, dataset, sampling):
        self.model = model
        self.dataset = dataset
        self.sampling = sampling
        self.device = model.device
        
        print('Initialize the HMC inference module...')
        self.inference = Inference.MFHMC(self.model, self.sampling, self.device)
        
    def fit(self, constraint):
        print('Evaluating the posterior samples given training data...')
        nXtr_list, nYtr_list = self.dataset.get_data(train=True, norm=True, torch_tensor=True, device=self.device)
        hmc_samples = self.inference.posterior(nXtr_list, nYtr_list, constraint)
        
        return hmc_samples


    def predict(self, hmc_samples):
        nXte_list, nYte_list = self.dataset.get_data(train=False, norm=True, torch_tensor=True, device=self.device)
        mf_nPred_list = self.inference.predict(nXte_list, hmc_samples)
        return mf_nPred_list

    def _functional_forward_(self, m, sample):
        
        params = self.inference._unflatten_net_params(sample, flat_tau=self.inference.flat_tau)
        
        def func_forward(nX):
            Ym = self.model.forward_by_params(nX, m, params['mf_weights'], params['mf_biases'])
            return Ym
        
        return func_forward

    def _eval_sample_f_star_(self, m, sample):
        n_np_lb, n_np_ub = self.dataset.get_scaled_bounds(m)
        #print(n_np_lb, n_np_ub)
        bounds = torch.tensor(np.vstack((n_np_lb, n_np_ub))).to(self.device)
        
        nXq = init_random_inputs_w_bounds(n_np_lb, n_np_ub, device=self.device)
        
        optimizer_f_star = LBFGS(
            [nXq], lr=1e-1, max_iter=20, max_eval=None, 
            tolerance_grad=1e-8, tolerance_change=1e-12, history_size=100)

        objective = self._functional_forward_(m, sample)

        def closure():
            optimizer_f_star.zero_grad() 

            loss = -torch.squeeze(objective(nXq))
            loss.backward(retain_graph=True)

            with torch.no_grad():
                for j, (lb, ub) in enumerate(zip(*bounds)):
                    nXq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
                #
            #
            return loss
        #
        
        optimizer_f_star.step(closure)
        
        for j, (lb, ub) in enumerate(zip(*bounds)):
            nXq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
        #
        
        nfstar = objective(nXq)

        return nfstar, nXq

    
    def fidelity_samples_extremes(self, m, samples):
        restart = 5
        
        samples_fstar = []
        samples_Xq = []
        
        for s in samples:
            
            fstar_buffer = []
            Xq_buffer = []
            
            for r in range(restart):
                nfstar, nXq = self._eval_sample_f_star_(m, s)
                fstar_buffer.append(nfstar)
                Xq_buffer.append(nXq)
            #
            
            # argmax the restart trials
            argi = torch.argmax(torch.tensor(fstar_buffer))

            samples_fstar.append(fstar_buffer[argi])
            samples_Xq.append(Xq_buffer[argi])
        #
        
        samples_nfstar = torch.stack(samples_fstar).squeeze()
        samples_nXq = torch.stack(samples_Xq)
        
        return samples_nfstar, samples_nXq

    def _functional_info_gain_(self, m, samples, Fstar):
        
        def info_gain_function(nX): 

            pred_list = []
            for s in samples:
                params = self.inference._unflatten_net_params(s, flat_tau=self.inference.flat_tau)
                pred_s = self.model.forward_by_params(nX, m, params['mf_weights'], params['mf_biases'])
                pred_list.append(pred_s)
            #

            Ns = len(samples)
            pred_samples = torch.stack(pred_list).squeeze()

            H_pred_samples = 0.5*torch.log(2*np.pi*np.e*torch.square(torch.std(pred_samples)))
            H_Fstar = 0.5*torch.log(2*np.pi*np.e*torch.square(torch.std(Fstar)))

            joint = torch.cat([Fstar, pred_samples])
            H_joint = 0.5*torch.log(2*np.pi*np.e*torch.square(torch.std(joint)))
            
            gain = H_pred_samples + H_Fstar - H_joint

            return gain
        #
        
        return info_gain_function

    def _batch_functional_info_gain_(self, m, samples, Fstar, batch_queried):
        
        def info_gain_function(nX): 

            pred_list = []
            for s in samples:
                params = self.inference._unflatten_net_params(s, flat_tau=self.inference.flat_tau)
                pred_s = self.model.forward_by_params(nX, m, params['mf_weights'], params['mf_biases'])
                pred_list.append(pred_s)
            #

            Ns = len(samples)
            pred_samples = torch.stack(pred_list).squeeze()

#             prev_batch = torch.stack(batch_queried).flatten()
            prev_batch = torch.stack(batch_queried).flatten().repeat(Ns)
            #print(prev_batch.shape)
            
            curr_batch = torch.cat([pred_samples, prev_batch])
            
            H_curr_batch = 0.5*torch.log(2*np.pi*np.e*torch.square(torch.std(curr_batch)))
            H_Fstar = 0.5*torch.log(2*np.pi*np.e*torch.square(torch.std(Fstar)))
            
            joint = torch.cat([Fstar, curr_batch])
            H_joint = 0.5*torch.log(2*np.pi*np.e*torch.square(torch.std(joint)))
            
            gain = H_curr_batch + H_Fstar - H_joint
            
            return gain
        
        return info_gain_function

    def _eval_info_gain_(self, m, samples, Fstar, batch_queried=[]):
        n_np_lb, n_np_ub = self.dataset.get_scaled_bounds(m)
        bounds = torch.tensor(np.vstack((n_np_lb, n_np_ub))).to(self.device)
        
        nXq = init_random_inputs_w_bounds(n_np_lb, n_np_ub, device=self.device)
        
        optimizer_f_star = LBFGS(
            [nXq], lr=1e-1, max_iter=20, max_eval=None, 
            tolerance_grad=1e-8, tolerance_change=1e-12, history_size=100)
        
        if not batch_queried:
            #print('no batch found, SIGNLE acquisition function used.')
            objective = self._functional_info_gain_(m, samples, Fstar)
        else:
            #print('found previous batch, BATCH acquisition function used.')
            objective = self._batch_functional_info_gain_(m, samples, Fstar, batch_queried)
        #

        def closure():
            optimizer_f_star.zero_grad() 

            loss = -torch.squeeze(objective(nXq))
            loss.backward(retain_graph=True)

            with torch.no_grad():
                for j, (lb, ub) in enumerate(zip(*bounds)):
                    nXq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
                #
            #
            return loss
        #
        
        optimizer_f_star.step(closure)
        
        for j, (lb, ub) in enumerate(zip(*bounds)):
            nXq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
        #
        
        eval_acq = objective(nXq)

        return eval_acq, nXq

    def fidelity_info_gain_by_samples(self, m, samples, Fstar, batch_queried=[]):

        restart = 5

        acq_buffer = []
        Xq_buffer = []
            
        for r in range(restart):
            eval_acq, nXq = self._eval_info_gain_(m, samples, Fstar, batch_queried)
            acq_buffer.append(eval_acq)
            Xq_buffer.append(nXq)
        #
            
        # argmax the restart trials
        argi = torch.argmax(torch.tensor(acq_buffer))
        #print(acq_buffer)
        
        arg_acq_eval = acq_buffer[argi]
        arg_nXq = Xq_buffer[argi]

        return arg_acq_eval, arg_nXq

    def info_gain_by_samples(self, samples, batch_queried=[]):
        
        evals_info_gain = []
        evals_nXq = []
        
        Fstar, _ = self.fidelity_samples_extremes(self.model.M-1, samples)
        
        for m in range(self.model.M):
            fidelity_info_gain, fidelity_nXq = self.fidelity_info_gain_by_samples(m, samples, Fstar, batch_queried)
            evals_info_gain.append(fidelity_info_gain)
            evals_nXq.append(fidelity_nXq)
        #
        return evals_info_gain, evals_nXq
    

    def info_gain_step(self, samples):

        evals_info_gain, evals_nXq = self.info_gain_by_samples(samples)

        penalty = torch.from_numpy(np.array(self.dataset.penalty)).float().to(self.device)

        reg_evals_info_gain = torch.stack(evals_info_gain)/penalty

        argm = torch.argmax(reg_evals_info_gain)
        argx = evals_nXq[argm]
        
        return argm, argx
    
    def fixed_fidelity_step(self, samples, fixed_fidelty):
        evals_info_gain, evals_nXq = self.info_gain_by_samples(samples)
        argm = fixed_fidelty
        argx = evals_nXq[argm] 
        
        return argm, argx
    
    def ratio_batch_step(self, samples, batch_size):

        batch_argm = []
        batch_argx = []
        batch_queried = []
        batch_costs = []
        
        penalty = np.array(self.dataset.penalty)

        for b in range(batch_size):
            evals_info_gain, evals_nXq = self.info_gain_by_samples(samples, batch_queried)
            
            buffer_acquisition = []
            for m in range(self.model.M):
                eval_acquisition = evals_info_gain[m]/(sum(batch_costs)+penalty[m])
                buffer_acquisition.append(eval_acquisition)
            #
            
            argm = torch.argmax(torch.stack(buffer_acquisition))
            argx = evals_nXq[argm]
#             print(argm)
#             print(argx)
            batch_argm.append(argm)
            batch_argx.append(argx)
            
#             print(buffer_acquisition )
#             print(argm, argx)
            
            batch_costs.append(penalty[argm])
            nyq = self.dataset.query_n_sample(argx, argm, torch_tensor=True, device=self.device)
            
            batch_queried.append(nyq)
#             print(batch_costs)
#             print(batch_queried)
            
        #
            
        return batch_argm, batch_argx, batch_queried
    
    def ao_ratio_batch_step(self, samples, batch_size, alters):
        

        batch_argm = []
        batch_argx = []
        batch_queried = []
        batch_costs = []
        
        penalty = np.array(self.dataset.penalty)
        
        if self.model.M == 2:
            p_init = [0.8, 0.2]
        elif self.model.M == 3:
            p_init = [0.8, 0.15, 0.05]
        #

        np_batch_argm = np.random.choice(self.model.M, size=batch_size, p=p_init).tolist()
        batch_argm = [torch.tensor(x).to(self.device) for x in np_batch_argm]
        batch_argx = []
        batch_queried = []
        batch_costs = []
        
        for i in range(batch_size):
            argm = batch_argm[i]
            n_np_lb, n_np_ub = self.dataset.get_scaled_bounds(argm)
            nXq = init_random_inputs_w_bounds(n_np_lb, n_np_ub, device=self.device)
            nyq = self.dataset.query_n_sample(nXq,  argm, torch_tensor=True, device=self.device)
            batch_argx.append(nXq)
            batch_queried.append(nyq)
            batch_costs.append(penalty[argm])
        #
        
        for k in range(alters):
            
            for b in range(batch_size):
                fixed_batch_argm = batch_argm[:b]+batch_argm[b+1:]
                fixed_batch_argx = batch_argx[:b]+batch_argx[b+1:]
                fixed_batch_queried = batch_queried[:b]+batch_queried[b+1:]
                fixed_batch_costs = batch_costs[:b]+batch_costs[b+1:]

                evals_info_gain, evals_nXq = self.info_gain_by_samples(samples, fixed_batch_queried)

                buffer_acquisition = []
                for m in range(self.model.M):
                    eval_acquisition = evals_info_gain[m]/(sum(fixed_batch_costs)+penalty[m])
                    buffer_acquisition.append(eval_acquisition)
                #

                argm = torch.argmax(torch.stack(buffer_acquisition))
                argx = evals_nXq[argm]

                nyq = self.dataset.query_n_sample(argx, argm, torch_tensor=True, device=self.device)

                batch_argm = batch_argm[:b]+ [argm] + batch_argm[b+1:]
                batch_argx = batch_argx[:b]+ [argx] + batch_argx[b+1:]
                batch_queried = batch_queried[:b]+ [nyq] + batch_queried[b+1:]
                batch_costs = batch_costs[:b]+ [penalty[argm]] + batch_costs[b+1:]


            #
        #
            
        return batch_argm, batch_argx, batch_queried
    
        
    def pseudo_par_step(self, samples, n_threads):
        # n_threads
        Ns = len(samples)
        perm = np.random.permutation(Ns)
        ns_per_thread = int(Ns/n_threads)
        
        pool_samples = []
        for i in range(0, Ns, ns_per_thread):
            i_subset = perm[i:i+ns_per_thread].astype(int).tolist()
            subset_samples = []
            for j in i_subset:
                subset_samples.append(samples[j])
            #
            pool_samples.append(subset_samples)
        #
        
        pool_argm = []
        pool_argx = []
        
        for subset_samples in pool_samples:
            argm, argx = self.info_gain_step(subset_samples)
            print(argm, argx)
            pool_argm.append(argm)
            pool_argx.append(argx)
        #
        
        return pool_argm, pool_argx
    
    
    
    
    
    
    
    