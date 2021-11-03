
class general_config(object):
    def __init__(self, config_name):
        self.config_name = config_name
        
    def _parse(self, kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        #
        
        print('=================================')
        print('*', self.config_name)
        print('---------------------------------')
        
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('-', k, ':', getattr(self, k))
        print('=================================')

        
class default_optimization_config(general_config):
    algorithm_name = None
    horizon = None
    num_trials = None
    init_i_trial = None
 
    def __init__(self,):
        super().__init__('Opt_Config')

class default_domain_config(general_config):

    domain_name = None
    domain_placement = None

    num_inits = None
    penalty = None

    def __init__(self,):
        super().__init__('Domain_Config')

        
class default_hmc_sampler_config(general_config):
    step_size = None
    L = None
    burn = None
    Ns = None
    
    def __init__(self,):
        super().__init__('HMC_Sampler_Config')
        
class default_mf_nn_surrogate_config(general_config):
    hidden_widths = None
    hidden_depths = None
    activation = None
    surrogate_placement = None
    
    def __init__(self,):
        super().__init__('MF_NN_Config')
        
class default_mf_gp_ucb_config(general_config):
    def __init__(self,):
        super().__init__('MF_GP_UCB_Config')
        
class default_mf_mes_config(general_config):
    def __init__(self,):
        super().__init__('MF_MES_Config')
        
# class default_mf_hmc_single_config(general_config):
#     constraint = None
#     fixed_fidelity = None
    
#     def __init__(self,):
#         super().__init__('MF_MES_Config')
        
class default_mf_hmc_single_config(general_config):
    constraint = None
#     surrogate_placement = None
    fixed_fidelity = None
    
    def __init__(self,):
        super().__init__('MF_HMC_Single_Config')
        
class default_mf_hmc_parallel_config(general_config):
    constraint = None
    n_threads = None
    
    def __init__(self,):
        super().__init__('MF_HMC_Parallel_Config')
        
class default_mf_hmc_batch_config(general_config):
    constraint = None
    batch_mode = None
    batch_size = None
    beta = None
    
    def __init__(self,):
        super().__init__('MF_HMC_Batch_Config')
        
        
class default_mf_hmc_random_config(general_config):
    constraint = None
#     surrogate_placement = None
    random_mode = None
    
    def __init__(self,):
        super().__init__('MF_HMC_Random_Config')
        
class default_mtbo_config(general_config):
    
    base_dim = None
    base_hidden_depth = None
    base_hidden_width = None
    surrogate_placement = None
    
    def __init__(self,):
        super().__init__('Multitask_BO_Config')
        
# class default_mtbo_config(general_config):
    
#     num_workers = 
#     base_hidden_depth = None
    
#     def __init__(self,):
#         super().__init__('Multitask_BO_Config')

        


        
