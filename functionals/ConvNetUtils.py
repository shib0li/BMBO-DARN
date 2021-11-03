import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
import numpy as np
from tqdm.auto import tqdm, trange
from collections import OrderedDict

import warnings


NUM_CONV_LAYERS = [1,2,3,4] #2
# NUM_INIT_CHANNELS = [16,32,64,128] #2
NUM_INIT_CHANNELS = np.arange(8,136).astype(int).tolist() #7
FCNN_LAYERS_DEPTH = [1,2,3,4,5,6,7,8] #3
# FCNN_LAYERS_WIDTH = [32,64,128,256,512,1024,2048,4096] #3
FCNN_LAYERS_WIDTH = np.arange(32,2080).astype(int).tolist() #11
POOLING_TYPE = ['max', 'average']
# FCNN LOG DROPOUT RATE -3 ~ 0


def conv_binary_decoder(X):

    log_dropout = X[-1]
    dropout_rate = np.power(10, log_dropout)
    
    if dropout_rate > 1.0:
        warnings.warn("Dropout rate larger than 1.0, P_drop = "+str(dropout_rate))
        dropout_rate=1.0
    
    binary_code = X[:-1]
    binary_code = (binary_code>=0.5).astype(int)
    
    #print(binary_code)
    
    bits_num_conv_layers = 2
    bits_num_init_channels = 7
    bits_fcnn_layers_depth = 3
    bits_fcnn_layers_width = 11
    bits_pooling_type = 1
    bits_log_dropout = 1

    curr = 0
    
    binary_num_conv_layers = ''.join([str(x) for x in binary_code[curr: curr+bits_num_conv_layers].tolist()])
    num_conv_layers = NUM_CONV_LAYERS[int(binary_num_conv_layers, 2)]
    curr += bits_num_conv_layers
    #print(num_conv_layers)
    
    binary_num_init_channels = ''.join([str(x) for x in binary_code[curr: curr+bits_num_init_channels].tolist()])
    num_init_channels = NUM_INIT_CHANNELS[int(binary_num_init_channels, 2)]
    curr += bits_num_init_channels
    #print(num_init_channels)
    
    binary_fcnn_layers_depth = ''.join([str(x) for x in binary_code[curr: curr+bits_fcnn_layers_depth].tolist()])
    fcnn_layers_depth = FCNN_LAYERS_DEPTH[int(binary_fcnn_layers_depth, 2)]
    curr += bits_fcnn_layers_depth
    #print(fcnn_layers_depth)
    
    binary_fcnn_layers_width = ''.join([str(x) for x in binary_code[curr: curr+bits_fcnn_layers_width].tolist()])
    fcnn_layers_width = FCNN_LAYERS_WIDTH[int(binary_fcnn_layers_width, 2)]
    curr += bits_fcnn_layers_width
    #print(fcnn_layers_width)
    
    binary_pooling_type = ''.join([str(x) for x in binary_code[curr: curr+bits_pooling_type].tolist()])
    pooling_type = POOLING_TYPE[int(binary_pooling_type, 2)]
    curr += bits_pooling_type
    #print(pooling_type)

    conv_config = {}
    conv_config['num_conv_layers'] = num_conv_layers
    conv_config['num_init_channels'] = num_init_channels
    conv_config['fcnn_layers_depth'] = fcnn_layers_depth
    conv_config['fcnn_layers_width'] = fcnn_layers_width
    conv_config['pooling_type'] = pooling_type
    conv_config['dropout_rate'] = dropout_rate
    
    return conv_config

def conv_binary_decoder_v2(X):
    
    if X.ndim == 2:
        X = np.squeeze(X)
    #

    log_dropout = X[-1]
    dropout_rate = np.power(10, log_dropout)
    
    if dropout_rate > 1.0:
        warnings.warn("Dropout rate larger than 1.0, P_drop = "+str(dropout_rate))
        dropout_rate=1.0
    
    binary_code = X[:-1]
    binary_code = (binary_code>=0.5).astype(int)
    
    #print(binary_code)
    
    bits_num_conv_layers = 2
    bits_num_init_channels = 7
    bits_fcnn_layers_depth = 3
    bits_fcnn_layers_width = 11
    bits_pooling_type = 1
    bits_log_dropout = 1

    curr = 0
    
    binary_num_conv_layers = ''.join([str(x) for x in binary_code[curr: curr+bits_num_conv_layers].tolist()])
    num_conv_layers = NUM_CONV_LAYERS[int(binary_num_conv_layers, 2)]
    curr += bits_num_conv_layers
    #print(num_conv_layers)
    
    binary_num_init_channels = ''.join([str(x) for x in binary_code[curr: curr+bits_num_init_channels].tolist()])
    num_init_channels = NUM_INIT_CHANNELS[int(binary_num_init_channels, 2)]
    curr += bits_num_init_channels
    #print(num_init_channels)
    
    binary_fcnn_layers_depth = ''.join([str(x) for x in binary_code[curr: curr+bits_fcnn_layers_depth].tolist()])
    fcnn_layers_depth = FCNN_LAYERS_DEPTH[int(binary_fcnn_layers_depth, 2)]
    curr += bits_fcnn_layers_depth
    #print(fcnn_layers_depth)
    
    binary_fcnn_layers_width = ''.join([str(x) for x in binary_code[curr: curr+bits_fcnn_layers_width].tolist()])
    fcnn_layers_width = FCNN_LAYERS_WIDTH[int(binary_fcnn_layers_width, 2)]
    curr += bits_fcnn_layers_width
    #print(fcnn_layers_width)
    
    binary_pooling_type = ''.join([str(x) for x in binary_code[curr: curr+bits_pooling_type].tolist()])
    pooling_type = POOLING_TYPE[int(binary_pooling_type, 2)]
    curr += bits_pooling_type
    #print(pooling_type)

    conv_config = {}
    conv_config['num_conv_layers'] = num_conv_layers
    conv_config['num_init_channels'] = num_init_channels
    conv_config['fcnn_layers_depth'] = fcnn_layers_depth
    conv_config['fcnn_layers_width'] = fcnn_layers_width
    conv_config['pooling_type'] = pooling_type
    conv_config['dropout_rate'] = dropout_rate
    
    return conv_config


class ConvNet(nn.Module):
    def __init__(self, conv_config):
        super(ConvNet, self).__init__()

        num_conv_layers = conv_config['num_conv_layers']
        init_filter_channels = conv_config['num_init_channels']
        pooling_type = conv_config['pooling_type']
        fcnn_layer_width = conv_config['fcnn_layers_width']
        fcnn_layer_depth = conv_config['fcnn_layers_depth']
        dropout_rate = conv_config['dropout_rate']
        
        channels = init_filter_channels*np.power(2, np.arange(num_conv_layers)).astype(int)

        conv_layers_output_dims = int(32/np.power(2,num_conv_layers))
        fcnn_input_dims = channels[-1]*conv_layers_output_dims*conv_layers_output_dims
        
        self.conv_channels = [3] + channels.tolist()
        self.fcnn_layers = [fcnn_input_dims] + [fcnn_layer_width]*fcnn_layer_depth + [10]
        #print(self.fcnn_layers)

        sequential_dict_conv = OrderedDict()
        for i_conv in range(num_conv_layers):
            in_channels = self.conv_channels[i_conv]
            out_channels = self.conv_channels[i_conv+1]
            #print(in_channels, out_channels)
            conv_key = 'conv'+str(i_conv+1)
            act_key = 'act_conv'+str(i_conv+1)
            pool_key = 'pool'+str(i_conv+1)
            sequential_dict_conv[conv_key] = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            sequential_dict_conv[act_key] = nn.ReLU()
            if pooling_type == 'max':
                sequential_dict_conv[pool_key] = nn.MaxPool2d(2, 2)
            elif pooling_type == 'average':
                sequential_dict_conv[pool_key] = nn.AvgPool2d(2, 2)
            else:
                raise Exception('Error: Unrecognized pooling type!')
            #
        #
        self.sequential_conv_layers = nn.Sequential(sequential_dict_conv)
        
        sequential_dict_fcnn = OrderedDict()
        num_fcnn_layers = len(self.fcnn_layers)
        for i_layer in range(num_fcnn_layers-2):
            in_d = self.fcnn_layers[i_layer]
            out_d = self.fcnn_layers[i_layer+1]
            linear_key = 'linear'+str(i_layer+1)
            dropout_key = 'dropout'+str(i_layer+1)
            act_key = 'act_linear'+str(i_layer+1)
            sequential_dict_fcnn[linear_key] = nn.Linear(in_d, out_d)
            sequential_dict_fcnn[dropout_key] = nn.Dropout(p=dropout_rate)
            sequential_dict_fcnn[act_key] = nn.ReLU()
        #
        
        linear_key = 'linear'+str(num_fcnn_layers-1)
        sequential_dict_fcnn[linear_key] = nn.Linear(self.fcnn_layers[-2], self.fcnn_layers[-1])
        
        self.sequential_fcnn_layers = nn.Sequential(sequential_dict_fcnn)
        
    def forward(self, X):
        H = self.sequential_conv_layers(X)
        H = H.view([-1, self.fcnn_layers[0]])
        y = self.sequential_fcnn_layers(H)
        return y
    
# def eval_conv_net_performance(domain, binary_config, max_epochs, mode, device):
    
#     if binary_config.ndim == 2:
#         binary_config = binary_config.squeeze()
    
#     conv_config = conv_binary_decoder(binary_config)
#     #print(conv_config)
#     conv_net = ConvNet(conv_config).to(device)
#     #print(conv_net)
    
#     optimizer = optim.SGD(conv_net.parameters(),lr=1e-3,momentum = 0.9,weight_decay=1e-5)
#     criterion = nn.CrossEntropyLoss()
    
#     hist_scores = []
    
#     for epoch in trange(max_epochs):  # loop over the dataset multiple times

#         for i, data in enumerate(domain.train_loader, 0):
#             inputs, labels = data[0].to(device), data[1].to(device)
#             optimizer.zero_grad()

#             preds = conv_net(inputs)
#             loss = criterion(preds, labels)
#             loss.backward()
#             optimizer.step()
#         #
        
#         if mode == 'generate':    
# #             train_acc = domain.metric(conv_net, device, score_type='train_pred_acc')
# #             test_acc = domain.metric(conv_net, device, score_type='test_pred_acc')
#             nll = domain.metric(conv_net, device, score_type='log_loss')
#             hist_scores.append(nll)
#             #print('%d-th epoch, train_acc=%.4f, test_acc=%.4f, test_log_loss=%.4f' % (epoch, train_acc, test_acc, nll))
#         #
#     #
#     if mode == 'query':
#         score = domain.metric(conv_net, device, score_type='log_loss')
#         return score 
#     elif mode == 'generate':
#         return np.array(hist_scores)
#     #
    
    
def eval_conv_net_performance(domain, binary_config, max_epochs, mode, device):
    
    if binary_config.ndim == 2:
        binary_config = binary_config.squeeze()
    
    conv_config = conv_binary_decoder(binary_config)
    #print(conv_config)
    conv_net = ConvNet(conv_config).to(device)
    #print(conv_net)
    
    optimizer = optim.SGD(conv_net.parameters(),lr=1e-3,momentum = 0.9,weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    hist_scores = []
    early_stop_cnt = 0

    for epoch in range(max_epochs):  # loop over the dataset multiple times

        for i, data in enumerate(domain.train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            preds = conv_net(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
        #
        
        score = domain.metric(conv_net, device, score_type='log_loss')
        
        if epoch == 0:
            hist_best = -np.inf
        else:
            hist_best = np.max(np.array(hist_scores))
        #

        if score <= hist_best:
            early_stop_cnt += 1
        else:
            early_stop_cnt = 0
        #
        
        hist_scores.append(score)

        if early_stop_cnt >= 5:
            break
        #
            
        
#         print(hist_best)
#         print(score)
#         print(early_stop_cnt)
#         print('')
    #
    
    if mode == 'query':
        return hist_scores[-1]
    elif mode == 'generate':
        if len(hist_scores) < max_epochs:
            append_hist_scores = [hist_scores[-1]]*(max_epochs-len(hist_scores))
            hist_scores = hist_scores + append_hist_scores
        #
        return np.array(hist_scores)
    #   
 





