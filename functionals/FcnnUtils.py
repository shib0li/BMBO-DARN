import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict 
import numpy as np
import warnings

from tqdm.auto import tqdm, trange

def fcnn_binary_decoder(X):
    
    log_dropout = X[-1]
    dropout_rate = np.power(10, log_dropout)
    
    if dropout_rate > 1.0:
        warnings.warn("Dropout rate larger than 1.0, P_drop = "+str(dropout_rate))
        dropout_rate=1.0
    
    binary_code = X[:-1]
    binary_code = (binary_code>=0.5).astype(int)
    #print(binary_code)
    
    bits_hidden_depth = 3
    bits_hidden_width = 6
    bits_activation = 2
    bits_log_dropout = 1

    curr = 0
    
    binary_hidden_depth = ''.join([str(x) for x in binary_code[curr: curr+bits_hidden_depth].tolist()])
    hidden_depth_offset = 0
    hidden_depth = int(binary_hidden_depth, 2)+hidden_depth_offset
    curr += bits_hidden_depth
    
    binary_hidden_width = ''.join([str(x) for x in binary_code[curr: curr+bits_hidden_width].tolist()])
    hidden_depth_offset = 2
    hidden_width = int(binary_hidden_width,2)+hidden_depth_offset
    
    curr += bits_hidden_width
    
    binary_activation = ''.join([str(x) for x in binary_code[curr: curr+bits_activation].tolist()])
    i_activation = int(binary_activation,2)
    activation = {0:'relu',1:'sigmoid',2:'tanh', 3:'leak_relu'}[i_activation]
    #print(activation)
    curr += bits_activation

    fcnn_config = {}
    fcnn_config['hidden_depth'] = hidden_depth
    fcnn_config['hidden_width'] = hidden_width
    fcnn_config['activation'] = activation
    fcnn_config['dropout_rate'] = dropout_rate
    
    return fcnn_config


class Net(nn.Module):
    def __init__(self, layers_config, activation, dropout_rate):
        super(Net, self).__init__()

        self.layers_config = layers_config
        self.num_layers = len(self.layers_config)
        
        self.actfn = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid, 'leak_relu':nn.LeakyReLU}[activation]
        
        sequential_dict = OrderedDict()
        
        for l in range(self.num_layers-2):
            in_d = self.layers_config[l]
            out_d = self.layers_config[l+1]
            fc_key = 'linear'+str(l+1)
            dropout_key = 'dropout'+str(l+1)
            activation_key = 'activation'+str(l+1)
            sequential_dict[fc_key] = nn.Linear(in_d, out_d)
            sequential_dict[dropout_key] = nn.Dropout(p=dropout_rate)
            sequential_dict[activation_key] = self.actfn()
        #
        
        fc_key = 'linear'+str(self.num_layers-1)
        sequential_dict[fc_key] = nn.Linear(layers_config[-2], layers_config[-1])
        
        self.seqentail = nn.Sequential(sequential_dict)
            

    def forward(self, x):
        y = self.seqentail(x)
        
        return y
    
    
def eval_fcnn_performace(domain, binary_config, max_epochs, mode, device):
    in_dim = domain.in_dim
    out_dim = domain.out_dim
    
    if binary_config.ndim == 2:
        binary_config = binary_config.squeeze()
    
    fcnn_config = fcnn_binary_decoder(binary_config)
    #print(fcnn_config)
    layers = [in_dim] + [fcnn_config['hidden_width']]*fcnn_config['hidden_depth'] + [out_dim]
    activation = fcnn_config['activation']
    dropout_rate = fcnn_config['dropout_rate']
    
    net = Net(layers, activation, dropout_rate).to(device)
    #print(net)
    
    np_nXtr, np_nytr = domain.get_data(train=True, normalize=True)
    np_nXte, np_nyte = domain.get_data(train=False, normalize=True)
    
    nXtr = torch.from_numpy(np_nXtr).float().to(device)
    nXte = torch.from_numpy(np_nXte).float().to(device)
    
    if domain.problem_category == 'regression':
        nytr = torch.from_numpy(np_nytr).float().to(device)
        nyte = torch.from_numpy(np_nyte).float().to(device)
        criterion = nn.MSELoss()
    elif domain.problem_category == 'classification':
        nytr = torch.from_numpy(np_nytr).long().to(device).squeeze()
        nyte = torch.from_numpy(np_nyte).long().to(device).squeeze()
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("ERROR: no task category found!")
    #
    
    optimizer = optim.SGD(net.parameters(),lr=1e-3,momentum = 0.9,weight_decay=1e-2)

    hist_scores = []

    for ie in range(max_epochs):
        optimizer.zero_grad()
        pred = net(nXtr)
        loss = criterion(pred, nytr)
        loss.backward()
        optimizer.step()
        
        if mode == 'generate':
            with torch.no_grad():
                nPred = net(nXte)
                score = domain.metric(nPred, normalize=True, torch_tensor=True)
                hist_scores.append(score)
                #print('%d-th epcoh, score=%.3f' %(ie, score))
            #
        #
    #
    
    if mode == 'query':
        with torch.no_grad():
            nPred = net(nXte)
            score = domain.metric(nPred, normalize=True, torch_tensor=True)
            return score
        #
    elif mode == 'generate':
        return np.array(hist_scores)
    
    