#!/usr/bin/env python
# coding: utf-8



#### utils.py #####
import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import pandas as pd
from nltk.corpus import stopwords


from pprint import pprint
from scipy import sparse
from scipy import io as sio

def set_random_seed(seed=0):
    """Set random/np.random/torch.random/cuda.random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
        
def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise 

def get_data_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix



# The configuration below is from the paper and set as default value.
default_configure = {
    'lr': 0.01,          # Learning rate
    'num_layers': 1,     # Number of layers
    'num_mlp_layers': 4,    # Number of mlp layers in each Layer of Model. 
    'hidden_size': 64,     # Number of hidden units.
    'dropout' : 0.6,      # dropout rate
    'weight_decay' : 0.001,
    'num_epochs' : 200,
    'patience' : 30
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = 'ACM'
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_acm_raw(ratio, remove_self_loop):
    """ Use ACM RAW Dataset which should be preprocessed."""
    assert not remove_self_loop
    data_file_path = 'ACM.mat'
    data = sio.loadmat(data_file_path)
    
    p_vs_l = data['PvsL']       # paper-field?  l:subject
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words, features?
    p_vs_c = data['PvsC']       # paper-conference, labels come from that 
    
    # assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]  # KDD, SIGMOD, SIGCOMM, MOBICOMM, VLDB
    label_ids = [0, 1, 2, 2, 1]
    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]
    
    hg = dgl.heterograph({
        ('paper', 'written-by', 'author') : p_vs_a.nonzero(),
        ('author', 'writing', 'paper') : p_vs_a.transpose().nonzero(),
        ('paper', 'is-about', 'subject') : p_vs_l.nonzero(),
        ('subject', 'has', 'paper') : p_vs_l.transpose().nonzero()
    })
    features = torch.FloatTensor(p_vs_t.toarray()) # features.
    
    # setup labels
    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)
    
    num_classes = 3
    
    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    
    # make sure the samples of train/val/test will have exact num_classes
    '''
    # You can randomly split the dataset
    train_idx = np.where(float_mask <= 0.8)[0]  
    val_idx = np.where((float_mask > 0.8) & (float_mask <= 0.9))[0]
    test_idx = np.where(float_mask > 0.9)[0]
    np.savez('split_80%',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx) 
    '''
   # or load pre-defined split.
    loadpath = 'split_' + str(ratio) + '%_best.npz'
    npzfile=np.load(loadpath) 
    
    train_idx= npzfile['train_idx']
    val_idx= npzfile['val_idx']
    test_idx= npzfile['test_idx']
    
    
    num_nodes = hg.number_of_nodes('paper') # target nodes.
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    
    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask

def load_data(dataset, ratio, remove_self_loop = False):
    if dataset == 'ACM':
        return load_acm_raw(ratio, remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))
    
class EarlyStopping(object):
    def __init__(self, dataset, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'results/early_stop_{}_{}_{:02d}-{:02d}-{:02d}.pth'.format(dataset,dt.date(),dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
    
    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
            
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
            
        return self.early_stop
    
    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)
        
    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
      
