import numpy as np
import sobol_seq

def generate_random_inputs(N, dim, lb, ub, seed):
    rand_state = np.random.get_state()
    try:
        np.random.seed(seed)
        noise = np.random.uniform(0,1,size=[N,dim])
        scale = (ub - lb).reshape([1,-1])
    except:
        print('Errors occured when generating random noise...')
    finally:
        np.random.set_state(rand_state)
    #
    X = noise*scale + lb
    return X

def generate_sobol_inputs(N, dim, lb, ub):
    noise = sobol_seq.i4_sobol_generate(dim, N)
    scale = (ub - lb).reshape([1,-1])
    X = noise*scale + lb
    return X 

# def generate_uniform_inputs(N, lb, ub, seed=None):
    
#     rand_state = np.random.get_state()
    
#     if seed is None:
#         seed = int(time.time()*1000000%(0xFFFFFFFF))
    
#     if lb.size != ub.size:
#         raise Exception('Error: check the lower bound and upper bound')
#     else:
#         dim = lb.size
    
#     try:
#         np.random.seed(seed)
#         noise = np.random.uniform(0,1,size=[N,dim])
#         scale = (ub - lb).reshape([1,-1])
#     except:
#         raise Exception('Error occured when generating uniform noise...')
#     finally:
#         np.random.set_state(rand_state)
#     #
    
#     X = noise*scale + lb
#     X = X.reshape([N, dim])
    
#     return X
# #

# def generate_sobol_inputs(N, lb, ub):

#     if lb.size != ub.size:
#         raise Exception('Error: check the lower bound and upper bound')
#     else:
#         dim = lb.size
    
#     try:
#         noise = sobol_seq.i4_sobol_generate(dim, N)
#     except:
#         raise Exception('Error occured when generating sobol noise...')
#     #

#     scale = (ub - lb).reshape([1,-1])
#     X = noise*scale + lb
#     X = X.reshape([N, dim])
    
#     return X
# #


def perm_by_seed(Nall, seed):
    rand_state = np.random.get_state()
    try:
        np.random.seed(seed)
        perm = np.random.permutation(Nall)
    except:
        print('Errors occured when generating random noise...')
    finally:
        np.random.set_state(rand_state)
        
    return perm

def full_auto_regressive_layers(in_dim, out_dim, hidden_depths, hidden_widths):
    layers = []
    for l in range(len(hidden_depths)):
        depth = hidden_depths[l]
        width = hidden_widths[l]
        layer = [in_dim+l] + [width]*depth + [out_dim]
        layers.append(layer)
    #
    return layers

def seq_auto_regressive_layers(in_dim, out_dim, hidden_depths, hidden_widths):
    layers = []
    for l in range(len(hidden_depths)):
        depth = hidden_depths[l]
        width = hidden_widths[l]
        if l == 0:
            layer = [in_dim] + [width]*depth + [out_dim]
        else:
            layer = [in_dim+1] + [width]*depth + [out_dim]
        #
        layers.append(layer)
    #
    return layers

