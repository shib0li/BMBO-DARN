import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np

from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import roc_auc_score
import pandas as pd
from scipy.special import softmax

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


from utils import Misc

ROOT= os.getcwd()

def eval_rmse_score(pred, ground):
    rmse = np.sqrt(np.mean(np.square(pred-ground)))
    return rmse

def eval_accuracy_score(pred, ground):
    pred_labels = np.argmax(pred, axis=1)
    correct = np.sum(pred_labels==ground.squeeze())
    total = pred_labels.size
    return correct/total

def eval_auc_score(pred, ground):
    probs = softmax(pred, axis=1)
    label = ground.squeeze()
    auc = roc_auc_score(label, probs[:,1])
    return auc


class BostonDomain:
    def __init__(self, partition_ratio, partition_seed):
        Xall, yall = load_boston(return_X_y=True)
        if yall.ndim == 1:
            yall = yall.reshape([-1,1])
        #
        
        self.in_dim = Xall.shape[1]
        self.out_dim = yall.shape[1]
        self.problem_category = 'regression'
        
        Nall = Xall.shape[0]
        perm = Misc.perm_by_seed(Nall, partition_seed)

        Xall_perm = Xall[perm, :]
        yall_perm = yall[perm,:]
        
        Ntr = int(Nall*partition_ratio)
        self.Xtr = Xall_perm[0:Ntr,:]
        self.ytr = yall_perm[0:Ntr,:]
        self.Xte = Xall_perm[Ntr:,:]
        self.yte = yall_perm[Ntr:,:]
        
    def get_data(self,train=True, normalize=False):
        if train:
            X = self.Xtr
            y = self.ytr
        else:
            X = self.Xte
            y = self.yte
        #
        
        if normalize:
            scaler_X = preprocessing.StandardScaler().fit(self.Xtr)
            scaler_y = preprocessing.StandardScaler().fit(self.ytr)
            
            X = scaler_X.transform(X)
            y = scaler_y.transform(y)
            
        #
        
        return X, y
    
    def metric(self, pred, normalize=True, torch_tensor=True):
        if torch_tensor:
            pred = pred.data.cpu().numpy()
        #
        scaler_y = preprocessing.StandardScaler().fit(self.ytr)
        if normalize:
            pred = scaler_y.inverse_transform(pred)
        #
        score = eval_rmse_score(pred, self.yte)
        score = score/scaler_y.scale_
        
        # minimize the rmse
        return -score.squeeze()

    
class CaliforniaDomain:
    def __init__(self, partition_ratio, partition_seed):
        raw_cal_housing = fetch_california_housing(data_home=os.path.join('functionals/cache'))
        Xall = raw_cal_housing.data
        yall = raw_cal_housing.target
        if yall.ndim == 1:
            yall = yall.reshape([-1,1])
        #
        
        self.in_dim = Xall.shape[1]
        self.out_dim = yall.shape[1]
        self.problem_category = 'regression'

        Nall = Xall.shape[0]
        perm = Misc.perm_by_seed(Nall, partition_seed)

        Xall_perm = Xall[perm, :]
        yall_perm = yall[perm,:]
        
        rsrv_ratio = 0.9
        Nrsrv = int(Nall*rsrv_ratio)
        
        Xall_perm_rsrv = Xall_perm[0:Nrsrv]
        yall_perm_rsrv = yall_perm[0:Nrsrv]
        
        #print(Xall_perm_rsrv.shape)
        #print(yall_perm_rsrv.shape)
        
        Ntr = int(Nrsrv*partition_ratio)
        #print(Ntr)
        self.Xtr = Xall_perm_rsrv[0:Ntr,:]
        self.ytr = yall_perm_rsrv[0:Ntr,:]
        self.Xte = Xall_perm_rsrv[Ntr:,:]
        self.yte = yall_perm_rsrv[Ntr:,:]
        
    def get_data(self,train=True, normalize=False):
        if train:
            X = self.Xtr
            y = self.ytr
        else:
            X = self.Xte
            y = self.yte
        #
        
        if normalize:
            scaler_X = preprocessing.StandardScaler().fit(self.Xtr)
            scaler_y = preprocessing.StandardScaler().fit(self.ytr)
            
            X = scaler_X.transform(X)
            y = scaler_y.transform(y)
            
        #
        
        return X, y
    
    def metric(self, pred, normalize=True, torch_tensor=True):
        if torch_tensor:
            pred = pred.data.cpu().numpy()
        #
        scaler_y = preprocessing.StandardScaler().fit(self.ytr)
        if normalize:
            pred = scaler_y.inverse_transform(pred)
        #
        score = eval_rmse_score(pred, self.yte)
        score = score/scaler_y.scale_

        # minimize the rmse
        return -score.squeeze()

class SonarDomain:
    def __init__(self, partition_ratio, partition_seed):
        sonar = pd.read_csv(os.path.join(ROOT,'functionals/cache/sonar.txt'),header=None)
        label_map_dic = {'M':1,'R':0}
        
        yall = sonar.iloc[:,-1].map(label_map_dic).to_numpy().reshape([-1,1])
        Xall = sonar.iloc[:,:-1].to_numpy()
        
        self.in_dim = 60
        self.out_dim = 2

        self.problem_category = 'classification'
        
        Nall = Xall.shape[0]
        perm = Misc.perm_by_seed(Nall, partition_seed)

        Xall_perm = Xall[perm, :]
        yall_perm = yall[perm, :]
        
        Ntr = int(Nall*partition_ratio)
        self.Xtr = Xall_perm[0:Ntr,:]
        self.ytr = yall_perm[0:Ntr,:]
        self.Xte = Xall_perm[Ntr:,:]
        self.yte = yall_perm[Ntr:,:]
        
    def get_data(self,train=True, normalize=False):
        if train:
            X = self.Xtr
            y = self.ytr
        else:
            X = self.Xte
            y = self.yte
        #
        
        if normalize:
            scaler_X = preprocessing.StandardScaler().fit(self.Xtr)
            #scaler_y = preprocessing.StandardScaler().fit(self.ytr)
            X = scaler_X.transform(X)
            #y = scaler_y.transform(y)
        #
        
        return X, y

    def metric(self, pred, normalize=True, torch_tensor=True, score_type='auc'):
        if torch_tensor:
            pred = pred.data.cpu().numpy()
        #
        if score_type == 'accuracy':
            #print('use ACC')
            return eval_accuracy_score(pred, self.yte)
        elif score_type == 'auc':
            #print('use AUC')
            return eval_auc_score(pred, self.yte)
        else:
            raise Exception("ERROR: not valid score type for binary classification")
        #

        
class Cifar10Domain:
    def __init__(self,):

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(ROOT,'functionals/cache'),
                                                   train=True, 
                                                   transform=transform,
                                                   download=True)

        test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(ROOT,'functionals/cache'),
                                                   train=False, 
                                                   transform=transform,
                                                   download=True)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=100, 
                                                   shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1000, 
                                                  shuffle=False)


    def metric(self, net, device, score_type='log_loss'):
        if score_type == 'log_loss':
            nll_list = []
            log_softmax_op = nn.LogSoftmax(dim=1)
            eval_loss = nn.NLLLoss()
            
            with torch.no_grad():
                for data in self.test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    nll = eval_loss(log_softmax_op(outputs), labels)
                    nll_list.append(nll.data.cpu().numpy())

            return -np.mean(np.array(nll_list))

        elif score_type == 'test_pred_acc':
            correct = 0
            total = 0

            with torch.no_grad():
                for data in self.test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            return correct/total
        elif score_type == 'train_pred_acc':
            correct = 0
            total = 0

            with torch.no_grad():
                for data in self.train_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            return correct/total
        else:
            raise Exception('Error: Unrecognized socre type!')
        
class NewsGroupDomain:
    def __init__(self):
 
        data_train, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                                     subset='train',
                                     remove=('headers', 'footers', 'quotes'),
                                     return_X_y=True)

        data_val, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                                     subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     return_X_y=True)
        
        n_train_samples = 512
        n_test_samples = 256
        data_train = data_train[:n_train_samples]
        data_val = data_val[:n_test_samples]
        
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=1000,
                                        stop_words='english')
        
        self.tf_train = tf_vectorizer.fit_transform(data_train)
        self.tf_val = tf_vectorizer.fit_transform(data_val)
        
    def metric(self, lda_model, device, score_type='perplexity'):
        if score_type == 'perplexity':
            return -lda_model.perplexity(self.tf_train)
        else:
            raise Exception('Error: Unrecognized score type.')
        #
    #
    
class BurgersShock:
    def __init__(self,):
        pass
    def metric(self,):
        pass
    
class BurgersShockExt:
    def __init__(self,):
        pass
    def metric(self,):
        pass
    
class DiabetesDomain:
    def __init__(self, partition_ratio, partition_seed):
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(
            X, y, test_size=0.333, random_state=27)

        self.ytr = self.ytr.reshape([-1,1])
        self.yte = self.yte.reshape([-1,1])

        
    def get_data(self,train=True, normalize=False):
        if train:
            X = self.Xtr
            y = self.ytr
        else:
            X = self.Xte
            y = self.yte
        #
        
        if normalize:
            scaler_X = preprocessing.StandardScaler().fit(self.Xtr)
            scaler_y = preprocessing.StandardScaler().fit(self.ytr)
            
            X = scaler_X.transform(X)
            y = scaler_y.transform(y)
            
        #
        
        return X, y
    
    def metric(self, pred, normalize=True, torch_tensor=False):
        if pred.ndim == 1:
            pred = pred.reshape([-1,1])
        
        if torch_tensor:
            pred = pred.data.cpu().numpy()
        #
        
        scaler_y = preprocessing.StandardScaler().fit(self.ytr)
        if normalize:
            pred = scaler_y.inverse_transform(pred)
        #
        
        score = eval_rmse_score(pred, self.yte)        
        #score = score/scaler_y.scale_
        score = np.log(score/scaler_y.scale_)
        
        # minimize the rmse
        return -score.squeeze()

    
    