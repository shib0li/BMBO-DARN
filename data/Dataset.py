import torch
import numpy as np
from sklearn import preprocessing
import time

import pickle5 as pickle
import warnings


class MfData:
    def __init__(self, mf_func, preload):
        
        self.mf_func = mf_func
        self.mf_functional = mf_func.mf_functional
        self.in_dim = self.mf_func.in_dim
        self.out_dim = self.mf_func.out_dim
        self.M = self.mf_func.Nfid
        
        self.penalty = self.mf_func.penalty
        
        self.lb = np.array(self.mf_func.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.mf_func.bounds, ndmin=2)[:, 1]
        
        self.mf_Xtr_list = []
        self.mf_ytr_list = []
        self.mf_Xte_list = []
        self.mf_yte_list = []

        with open(preload, 'rb') as handle:
            print('loaded saved samples')
            mfdata = pickle.load(handle)

        self.mf_Xtr_list = mfdata['mf_Xtr_list']
        self.mf_ytr_list = mfdata['mf_ytr_list']
        self.mf_Xte_list = mfdata['mf_Xte_list']
        self.mf_yte_list = mfdata['mf_yte_list']

    
    def sequential_query_output(self, X, m):
        if X.ndim == 1:
            X = np.reshape(X, [-1, self.in_dim])
        
        N = X.shape[0]

        y_list = []
        
        for n in range(N):
            Xn = X[n].reshape([1,-1])
            #print(Xn)
            yn = self.mf_functional[m](Xn)
            y_list.append(yn)
            
        Y = np.stack(y_list).reshape([N, self.out_dim])

        return Y

    def _dim_sanity_check_input(self, X):
        if X.ndim == 1 or X.ndim == 0:
            X = np.reshape(X, [-1, self.in_dim])
        #
        return X
    
    def _dim_sanity_check_output(self, y):
        if y.ndim == 1 or y.ndim == 0:
            y = np.reshape(y, [-1, self.out_dim])
        #
        return y
    
    def _get_normalize_scalers(self, m):

        Xtr = self.mf_Xtr_list[m]
        ytr = self.mf_ytr_list[m]
        
        scaler_X = preprocessing.StandardScaler().fit(Xtr)
        scaler_y = preprocessing.StandardScaler().fit(ytr)
        
        return scaler_X, scaler_y
    
    def scale_input(self, X, m, inverse):
        X = self._dim_sanity_check_input(X)
        scaler_X, _ = self._get_normalize_scalers(m)
        
        if inverse:
            X_scaled = scaler_X.inverse_transform(X)
        else:
            X_scaled = scaler_X.transform(X)
        #
        return X_scaled
    
    def scale_output(self, y, m, inverse):
        y = self._dim_sanity_check_output(y)
        _, scaler_y = self._get_normalize_scalers(m)
        
        if inverse:
            y_scaled = scaler_y.inverse_transform(y)
        else:
            y_scaled = scaler_y.transform(y)
        #
        return y_scaled

    def get_scaled_bounds(self, m):
        scaled_lb = self.scale_input(self.lb, m, inverse=False).squeeze(axis=0)
        scaled_ub = self.scale_input(self.ub, m, inverse=False).squeeze(axis=0)
        return scaled_lb, scaled_ub
    

    def _query(self, X, m, decode=False):
        X = self._dim_sanity_check_input(X) 
        y = self._dim_sanity_check_output(self.mf_functional[m](X))
        
        if decode:
            config = self.mf_func.config_decoder(X)
            return y, config
        else:
            return y
    
    def query_n_sample(self, nX, m, torch_tensor=False, device=torch.device('cpu')):
        
        if torch_tensor:
            nX = nX.data.cpu().numpy()
        #
        
        nX = self._dim_sanity_check_input(nX) 
        X = self.scale_input(nX, m, inverse=True)
        
        y = self._dim_sanity_check_output(self.mf_functional[m](X))
        ny = self.scale_output(y, m, inverse=False)
        
        if torch_tensor:
            ny = torch.from_numpy(ny).float().to(device)
        #
            
        
        return ny

#     def _add_by_one(self, X, m):
        
#         y = self._query(X, m)
        
#         self.mf_Xtr_list[m] = np.vstack([self.mf_Xtr_list[m], X])
#         self.mf_ytr_list[m] = np.vstack([self.mf_ytr_list[m], y])
        
#         print('*** Added sample:', X)
#         print('New sample added...the training dataset size is', self.mf_Xtr_list[m].shape[0], 'at fidelity', m)
        
#         if m == self.M-1:
#             y_hf = y
#         else:
#             y_hf = self._query(X, self.M-1)
#         #
#         return y, y_hf

    def _add_by_one(self, X, m):
        
        success = True
        t_query_m = 0.0
        t_query_h = 0.0
        
        # first check if the configuration has nan elements
        if np.isnan(np.sum(X)):
            success = False
        else:
            # second check the query success
            try:
                
                t_start = time.time()
                y = self._query(X, m)
                t_query_m = time.time()-t_start
                
                if np.isnan(np.sum(y)):
                    warnings.warn("WARNING: Queried NaN sample! Skip the sample")
                    success = False
            except:
                warnings.warn("WARNING: Query is not successful! Skip the sample")
                success = False
            #
        #
        
        if success:
            self.mf_Xtr_list[m] = np.vstack([self.mf_Xtr_list[m], X])
            self.mf_ytr_list[m] = np.vstack([self.mf_ytr_list[m], y])
            print('*** Added sample:', X)
            print('New sample added...the training dataset size is', self.mf_Xtr_list[m].shape[0], 'at fidelity', m)
        
            if m == self.M-1:
                t_query_h = t_query_m
                y_hf = y
            else:
                t_start = time.time()
                y_hf = self._query(X, self.M-1)
                t_query_h = time.time()-t_start
            #
            
            if np.isnan(np.sum(y_hf)):
                warnings.warn("WARNING: Queried HF resulted in NaN sample! Skip the sample")
                success = False
            #
            
            return y, y_hf, success, t_query_m, t_query_h
        else:
            return np.nan, np.nan, success, t_query_m, t_query_h
        
    def _add_by_one_interpret(self, X, m):
        
        success = True
        t_query_m = 0.0
        t_query_h = 0.0
        
        interpreted = None
        
        # first check if the configuration has nan elements
        if np.isnan(np.sum(X)):
            success = False
        else:
            # second check the query success
            try:
                t_start = time.time()
                y, interpreted = self._query(X, m, decode=True)
                t_query_m = time.time()-t_start
                
                if np.isnan(np.sum(y)):
                    warnings.warn("WARNING: Queried NaN sample! Skip the sample")
                    success = False
            except:
                warnings.warn("WARNING: Query is not successful! Skip the sample")
                success = False
            #
        #
        
        if success:
            self.mf_Xtr_list[m] = np.vstack([self.mf_Xtr_list[m], X])
            self.mf_ytr_list[m] = np.vstack([self.mf_ytr_list[m], y])
            print('*** Added sample in INTERPRET mode:', X)
            print('New sample added...the training dataset size is', self.mf_Xtr_list[m].shape[0], 'at fidelity', m)
            print(interpreted)
            
            if m == self.M-1:
                t_query_h = t_query_m
                y_hf = y
            else:
                t_start = time.time()
                y_hf = self._query(X, self.M-1)
                t_query_h = time.time()-t_start
            #
            
            if np.isnan(np.sum(y_hf)):
                warnings.warn("WARNING: Queried HF resulted in NaN sample! Skip the sample")
                success = False
            #
            
            return y, y_hf, success, interpreted, t_query_m, t_query_h
        else:
            return np.nan, np.nan, success, interpreted, t_query_m, t_query_h
    
    def add(self, X, m, scaled_input=False):
        
        X = self._dim_sanity_check_input(X)
        if scaled_input:
            X = self.scale_input(X, m, inverse=True)
        #
        
        y, y_hf, success, t_query_m, t_query_h = self._add_by_one(X, m)

        return y, y_hf, success, t_query_m, t_query_h

    def add_interpret(self, X, m, scaled_input=False):
        
        X = self._dim_sanity_check_input(X)
        if scaled_input:
            X = self.scale_input(X, m, inverse=True)
        #
        
        y, y_hf, success, config, t_query_m, t_query_h = self._add_by_one_interpret(X, m)

        return y, y_hf, success, config, t_query_m, t_query_h
        

    def add_pool(self, X_pool, m_pool, scaled_input=False):
        if scaled_input:
            scaled_X_pool = []
            for X, m in zip(X_pool, m_pool):
                X = self._dim_sanity_check_input(X)
                scaled_X = self.scale_input(X, m, inverse=True)
                scaled_X_pool.append(scaled_X)
            #
            X_pool = scaled_X_pool
        #
        
        y_pool = []
        y_hf_pool = []
        success_pool = []
        t_query_m_pool = []
        t_query_h_pool = []
        
        for X, m in zip(X_pool, m_pool):
            y, y_hf, success, t_query_m, t_query_h = self._add_by_one(X, m)
            y_pool.append(y)
            y_hf_pool.append(y_hf)
            success_pool.append(success)
            t_query_m_pool.append(t_query_m)
            t_query_h_pool.append(t_query_h)
        #
        
        return y_pool, y_hf_pool, success_pool, t_query_m_pool, t_query_h_pool
    
    def add_pool_interpret(self, X_pool, m_pool, scaled_input=False):
        if scaled_input:
            scaled_X_pool = []
            for X, m in zip(X_pool, m_pool):
                X = self._dim_sanity_check_input(X)
                scaled_X = self.scale_input(X, m, inverse=True)
                scaled_X_pool.append(scaled_X)
            #
            X_pool = scaled_X_pool
        #
        
        y_pool = []
        y_hf_pool = []
        success_pool = []
        config_pool = []
        t_query_m_pool = []
        t_query_h_pool = []
        
        for X, m in zip(X_pool, m_pool):
            y, y_hf, success, config, t_query_m, t_query_h = self._add_by_one_interpret(X, m)
            y_pool.append(y)
            y_hf_pool.append(y_hf)
            success_pool.append(success)
            config_pool.append(config)
            t_query_m_pool.append(t_query_m)
            t_query_h_pool.append(t_query_h)
        #
        
        return y_pool, y_hf_pool, success_pool, config_pool, t_query_m_pool, t_query_h_pool
    

    def get_data(self, train=True, norm=True, torch_tensor=True, device=torch.device('cpu')):
        if train:
            mf_X_list = self.mf_Xtr_list.copy()
            mf_y_list = self.mf_ytr_list.copy()
        else:
            mf_X_list = self.mf_Xte_list.copy()
            mf_y_list = self.mf_yte_list.copy()
        #

        if norm:
            for m in range(self.M):
                scaler_X, scaler_y = self._get_normalize_scalers(m)
                mf_X_list[m] = scaler_X.transform(mf_X_list[m])
                mf_y_list[m] = scaler_y.transform(mf_y_list[m])
            #
        #
        
        
        if torch_tensor:
            for m in range(self.M):
                mf_X_list[m] = torch.from_numpy(mf_X_list[m]).float().to(device)
                mf_y_list[m] = torch.from_numpy(mf_y_list[m]).float().to(device)
            #
        #
        
        return mf_X_list, mf_y_list
    
    