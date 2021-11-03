import numpy as np
# from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn import ensemble
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_squared_log_error
# from sklearn import preprocessing


# from utils import Misc
# import functionals.Domains as domains

import warnings


def xgboost_regressor_binary_decoder(X):
    
    CRITERIA = ['friedman_mse','mse']
    
    log_alpha = X[0]
    log_ccpa = X[1]
    log_subsample = X[2]
    log_max_features = X[3]
    
    alpha = np.power(10, log_alpha)
    if alpha >= 1.0:
        warnings.warn("subsample larger than 1.0, val = "+str(alpha))
        alpha=1.0-1e-3
    
    ccpa = np.power(10, log_ccpa)
    
    subsample = np.power(10, log_subsample)
    if subsample > 1.0:
        warnings.warn("subsample larger than 1.0, val = "+str(subsample))
        subsample=1.0
    
    max_features = np.power(10, log_max_features)
    if max_features > 1.0:
        warnings.warn("max_features larger than 1.0, val = "+str(max_features))
        max_features=1.0
    
    binary_code = X[4:]
    binary_code = (binary_code>=0.5).astype(int)
    #print(binary_code)
    
    
    bits_criteria = 1
    bits_sample_splitting = 3
    bits_max_depth = 4
    
    curr = 0
    
    binary_criteria = ''.join([str(x) for x in binary_code[curr: curr+bits_criteria].tolist()])
    criteria = CRITERIA[int(binary_criteria, 2)]
    curr += bits_criteria
    #print(criteria)
    
    binary_sample_splitting = ''.join([str(x) for x in binary_code[curr: curr+bits_sample_splitting].tolist()])
    sample_splitting = int(binary_sample_splitting, 2)+2
    curr += bits_sample_splitting
    #print(sample_splitting)
    
    binary_max_depth = ''.join([str(x) for x in binary_code[curr: curr+bits_max_depth].tolist()])
    max_depth = int(binary_max_depth, 2)+1
    curr += bits_max_depth
    #print(max_depth)
    
    
    xgb_config = {}
    xgb_config['alpha'] = alpha
    xgb_config['ccpa'] = ccpa
    xgb_config['subsample'] = subsample
    xgb_config['max_features'] = max_features
    xgb_config['criteria'] = criteria
    xgb_config['sample_splitting'] = sample_splitting
    xgb_config['max_depth'] = max_depth
    
    return xgb_config


def xgboost_regressor_binary_decoder_v2(X):
    
    if X.ndim == 2:
        X = np.squeeze(X)
    #
    
    CRITERIA = ['friedman_mse','mse']
    
    log_alpha = X[0]
    log_ccpa = X[1]
    log_subsample = X[2]
    log_max_features = X[3]
    
    alpha = np.power(10, log_alpha)
    if alpha >= 1.0:
        warnings.warn("subsample larger than 1.0, val = "+str(alpha))
        alpha=1.0-1e-3
    
    ccpa = np.power(10, log_ccpa)
    
    subsample = np.power(10, log_subsample)
    if subsample > 1.0:
        warnings.warn("subsample larger than 1.0, val = "+str(subsample))
        subsample=1.0
    
    max_features = np.power(10, log_max_features)
    if max_features > 1.0:
        warnings.warn("max_features larger than 1.0, val = "+str(max_features))
        max_features=1.0
    
    binary_code = X[4:]
    binary_code = (binary_code>=0.5).astype(int)
    #print(binary_code)
    
    
    bits_criteria = 1
    bits_sample_splitting = 3
    bits_max_depth = 4
    
    curr = 0
    
    binary_criteria = ''.join([str(x) for x in binary_code[curr: curr+bits_criteria].tolist()])
    criteria = CRITERIA[int(binary_criteria, 2)]
    curr += bits_criteria
    #print(criteria)
    
    binary_sample_splitting = ''.join([str(x) for x in binary_code[curr: curr+bits_sample_splitting].tolist()])
    sample_splitting = int(binary_sample_splitting, 2)+2
    curr += bits_sample_splitting
    #print(sample_splitting)
    
    binary_max_depth = ''.join([str(x) for x in binary_code[curr: curr+bits_max_depth].tolist()])
    max_depth = int(binary_max_depth, 2)+1
    curr += bits_max_depth
    #print(max_depth)
    
    
    xgb_config = {}
    xgb_config['alpha'] = alpha
    xgb_config['ccpa'] = ccpa
    xgb_config['subsample'] = subsample
    xgb_config['max_features'] = max_features
    xgb_config['criteria'] = criteria
    xgb_config['sample_splitting'] = sample_splitting
    xgb_config['max_depth'] = max_depth
    
    return xgb_config

def wrap_xgb_regressor_params(xgb_config, n_boosters):
    
    params = {'n_estimators': n_boosters,
              'loss': 'huber',
              'alpha': xgb_config['alpha'],
              'ccp_alpha': xgb_config['ccpa'],
              'subsample': xgb_config['subsample'],
              'max_features': xgb_config['max_features'],
              'criterion': xgb_config['criteria'],
              'min_samples_split': xgb_config['sample_splitting'],
              'max_depth': xgb_config['max_depth'],
              'learning_rate': 1e-1,
              'n_iter_no_change':10}
    
    return params


def eval_xgb_regressor_performace(domain, binary_config, max_boosters, mode, device):
    if binary_config.ndim == 2:
        binary_config = binary_config.squeeze()
        
    Xtr, ytr = domain.get_data(train=True, normalize=True)
    Xte, yte = domain.get_data(train=False, normalize=True)
    
    eval_set = [(Xte, yte.squeeze())]
        
    xgb_config = xgboost_regressor_binary_decoder(binary_config)
    #print(xgb_config)
    
    fidelity_list = [2,50,100]

    if mode == 'query':
        params = wrap_xgb_regressor_params(xgb_config, max_boosters)
        
        xgb_regressor = ensemble.GradientBoostingRegressor(**params)
        xgb_regressor.fit(Xtr, ytr.squeeze())
        
        pred = xgb_regressor.predict(Xte)
        score = domain.metric(pred)
        
        return score
    
    elif mode == 'generate':
        hist_scores = []
        for n_boosters in fidelity_list:
            params = wrap_xgb_regressor_params(xgb_config, n_boosters)
            xgb_regressor = ensemble.GradientBoostingRegressor(**params)
            xgb_regressor.fit(Xtr, ytr.squeeze())

            pred = xgb_regressor.predict(Xte)
            score = domain.metric(pred)
            
            hist_scores.append(score)
        #
        return np.array(hist_scores)
    
    
    