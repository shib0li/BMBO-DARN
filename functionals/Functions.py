import torch
import numpy as np
import functionals.Domains as domains
import functionals.FcnnUtils as fcnn_utils
import functionals.ConvNetUtils as conv_net_utils
import functionals.LdaUtils as lda_utils
import functionals.PinnUtils as pinn_utils
import functionals.PinnExtUtils as pinn_ext_utils
import functionals.XGBRegressorUtils as xgbr_utils

FCNN_APPLICATIONS = ['Boston', 'California', 'Sonar']
COVNET_APPPLICATIONS = ['Cifar10']
LDA_APPPLICATIONS = ['NewsGroup']
PINN_APPLICATIONS = ['BurgersShock']
PINN_EXT_APPLICATIONS = ['BurgersShockExt']
XGBR_APPLICATIONS = ['Diabetes']


def BostonFunctional(device, max_epoch, mode='query'):
    domain = domains.BostonDomain(partition_ratio=0.8, partition_seed=1)
    def func(binary_code):
        score = fcnn_utils.eval_fcnn_performace(domain, binary_code, max_epoch, mode, device)
        return score
    #
    return func

def CaliforniaFunctional(device, max_epoch, mode='query'):
    domain = domains.CaliforniaDomain(partition_ratio=0.5, partition_seed=1)
    def func(binary_code):
        score = fcnn_utils.eval_fcnn_performace(domain, binary_code, max_epoch, mode, device)
        return score
    #
    return func

def SonarFunctional(device, max_epoch, mode='query'):
    domain = domains.SonarDomain(partition_ratio=0.5, partition_seed=1)
    def func(binary_code):
        score = fcnn_utils.eval_fcnn_performace(domain, binary_code, max_epoch, mode, device)
        return score
    #
    return func

def Cifar10Functional(device, max_epoch, mode='query'):
    domain = domains.Cifar10Domain()
    def func(binary_code):
        score = conv_net_utils.eval_conv_net_performance(domain, binary_code, max_epoch, mode, device)
        return score
    #
    return func

def NewsGroupFunctional(device, max_epoch, mode='query'):
    domain = domains.NewsGroupDomain()
    def func(binary_code):
        score = lda_utils.eval_lda_performance(domain, binary_code, max_epoch, mode, device=None)
        return score
    #
    return func

def BurgersShockFunctional(device, max_iters, mode='query'):
    domain = domains.BurgersShock()
    def func(binary_code):
        score = pinn_utils.eval_pinn_performace(domain, binary_code, max_iters, mode, device)
        return score
    #
    return func

def BurgersShockExtFunctional(device, max_iters, mode='query'):
    domain = domains.BurgersShockExt()
    def func(binary_code):
        score = pinn_ext_utils.eval_pinn_ext_performace(domain, binary_code, max_iters, mode, device)
        return score
    #
    return func

def DiabetesFunctional(device, max_iters, mode='query'):
    domain = domains.DiabetesDomain(partition_ratio=0.33, partition_seed=27)
    def func(binary_code):
        score = xgbr_utils.eval_xgb_regressor_performace(domain, binary_code, max_iters, mode, device)
        return score
    #
    return func

class MfFunc:
    def __init__(self, domain_name, penalty, device):
        self.penalty = penalty
        self.max_epochs_list = penalty
        self.Nfid = len(penalty)
        self.device = device
        
        
        if domain_name in FCNN_APPLICATIONS:
            self.bounds=tuple([(0,1)]*11 + [(-3,0)])
            self.in_dim = 12
            self.out_dim = 1
        elif domain_name in COVNET_APPPLICATIONS:
            self.bounds=tuple([(0,1)]*24 + [(-3,0)])
            self.in_dim = 25
            self.out_dim = 1
            self.config_decoder = conv_net_utils.conv_binary_decoder_v2
        elif domain_name in LDA_APPPLICATIONS:
            continuous_bounds = list(((1e-3,1.0),(1e-3,1.0),(0.51,1.0),(-5,-1)))
            binary_bounds = [(0,1)]*12
            self.bounds = continuous_bounds+binary_bounds
            self.in_dim = 16
            self.out_dim = 1
            self.config_decoder = lda_utils.lda_binary_decoder_v2
        elif domain_name in PINN_APPLICATIONS:
            self.bounds = [(0,1)]*12
            self.in_dim = 12
            self.out_dim = 1
            self.config_decoder = pinn_utils.pinn_binary_decoder_v2
        elif domain_name in PINN_EXT_APPLICATIONS:
            self.bounds = [(0,1)]*15
            self.in_dim = 15
            self.out_dim = 1
            self.config_decoder = pinn_ext_utils.pinn_ext_binary_decoder
        elif domain_name in XGBR_APPLICATIONS:
            continuous_bounds = list(((-2,0),(-2,2),(-1,0),(-2,0)))
            binary_bounds = [(0,1)]*8
            self.bounds = continuous_bounds+binary_bounds
            self.in_dim = 12
            self.out_dim = 1
            self.config_decoder = xgbr_utils.xgboost_regressor_binary_decoder_v2
        else:
            raise Exception('Error! Unrecognoized domains!')
        
        self.mf_functional = []

        for m in range(self.Nfid):
            max_epoch = self.max_epochs_list[m]
            if domain_name == 'Boston':
                func = BostonFunctional(self.device, max_epoch)
            elif domain_name == 'California':
                func = CaliforniaFunctional(self.device, max_epoch)
            elif domain_name == 'Sonar':
                func = SonarFunctional(self.device, max_epoch)
            elif domain_name == 'Cifar10':
                func = Cifar10Functional(self.device, max_epoch)
            elif domain_name == 'NewsGroup':
                func = NewsGroupFunctional(self.device, max_epoch)
            elif domain_name == 'BurgersShock':
                func = BurgersShockFunctional(self.device, max_epoch)
            elif domain_name == 'BurgersShockExt':
                func = BurgersShockExtFunctional(self.device, max_epoch)
            elif domain_name == 'Diabetes':
                func = DiabetesFunctional(self.device, max_epoch)
            else:
                raise Exception("ERROR: domain not implemented! Change to other domains...")
            #
            self.mf_functional.append(func)
        #


