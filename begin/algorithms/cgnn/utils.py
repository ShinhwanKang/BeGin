import os 
import sys
import logging
import time
import copy
from itertools import chain
from collections import defaultdict, deque
import pickle
import json
import numpy as np 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import dgl
import dgl.function as fn
from sklearn.cluster import MiniBatchKMeans

def detect(data, cur_g, X, t, strategy, new_node_size, device):
    return detect_bfs(data, cur_g, X, t, new_node_size, device)

def detect_bfs(data, cur_g, X, t, new_node_size, device):
    model = copy.deepcopy(data['sage'])
    h_pre = get_h(model, data['before_g'].to(device), X.to(device), data['train_cha_nodes_list'])
    h_cur = get_h(model, cur_g.to(device), X.to(device), data['train_cha_nodes_list'])

    # Calculate delta of embeddings
    delta_h = (h_cur - h_pre).abs().sum(1)    
    # Calculate influenced nodes
    f_matrix = bfs_sharp(cur_g, data['train_cha_nodes_list'], delta_h, device, 2)
    
    candidate_mask = torch.zeros_like(f_matrix).bool()
    candidate_mask[data['train_cha_nodes_list']] = True
    candidate_mask[data['old_nodes_list']] = True
    node_indices = torch.arange(cur_g.num_nodes())[candidate_mask.cpu()]
    f_matrix = f_matrix[candidate_mask]
    new_nodes = node_indices[torch.argsort(f_matrix, dim=0)[-new_node_size:].cpu()].detach().cpu().numpy().tolist()
    return new_nodes

def get_h(model, g, X, train_cha_nodes_list):
    return model.forward(g, X)[train_cha_nodes_list]
    
def bfs_sharp(g, nodes, weights, device, hop = 2):
    g = copy.deepcopy(g).to(device)
    with g.local_scope():
        g.srcdata['_bfs_input'] = torch.zeros(g.num_nodes()).to(device)
        g.srcdata['_bfs_input'][torch.LongTensor(nodes).to(device)] = weights
        for i in range(hop):
            g.update_all(fn.copy_u('_bfs_input', '_bfs_m'), fn.mean('_bfs_m', '_bfs_output'))
            g.srcdata['_bfs_input'] = g.dstdata.pop('_bfs_output')
        return g.srcdata.pop('_bfs_input')
        
def bfs_plus(g, nodes, device, hop = 2):
    adj_lists = g.adj()
    
    node2f = defaultdict(dict)
    node2idx = dict()
    center2idx = dict()
    
    for center in nodes:
        # Search from each changed nodes
        vis_nodes = dict()
        vis_nodes[center] = 1
        f = list()
        f.append(dict())
        f[0][center] = 1
        for i in range(hop):
            f.append(dict())
            new_nodes = dict()
            for node in vis_nodes:
                for neigh in adj_lists[node].coalesce().indices().squeeze().tolist():
                    if neigh not in vis_nodes and neigh not in new_nodes:
                        new_nodes[neigh] = 1
            vis_nodes.update(new_nodes)
            for node in vis_nodes:
                fun = 0.0
                if node in f[i]:
                    fun += f[i][node]
                num_neighs = (len(adj_lists[node].coalesce().indices().squeeze().tolist()) + 1)
                for neigh in adj_lists[node].coalesce().indices().squeeze().tolist():
                    if (neigh in f[i]) and (neigh != node):
                        fun += f[i][neigh]
                    if neigh == node:
                        num_neighs -= 1
                f[i + 1][node] = fun / num_neighs
                
        for node in vis_nodes:
            if node not in node2idx:
                node2idx[node] = len(node2idx)
            node2f[node][center] = f[hop][node]
        if center not in center2idx:
            center2idx[center] = len(center2idx)
    
    f_matrix = np.zeros((len(node2idx), len(nodes)), dtype=np.float32)
    for node in node2f:
        for center in node2f[node]:
            f_matrix[node2idx[node]][center2idx[center]] = node2f[node][center]
    
    return list(node2idx.keys()), f_matrix

class MemoryHandler(object):

    def __init__(self, memory_size, p, strategy, alpha, device):
        # strategy: random / class
        super(MemoryHandler, self).__init__()
        
        self.memory_size = memory_size
        self.p = p
        self.strategy = strategy
        self.clock = 0
        self.device = device
        
        self.memory = list()
    
        if self.strategy == 'class':
            self.data_size = 0
            self.data_size_per_class = defaultdict(int)
            self.memory_size_per_class = defaultdict(int)
            self.memory_per_class = defaultdict(list)
            self.memory_per_class_log = defaultdict(list)
            self.alpha = alpha

    def update(self, nodes, g, x = None, y = None):
        if self.strategy == 'class':
            importance = self._compute_node_importance(nodes, y, g)
            self._update_class(nodes, y[nodes], importance)
        else:
            self._update_random(nodes)
        self.clock += 1
        

    def _update_random(self, nodes):
        for i, node in enumerate(nodes):
            if node in self.memory:
                continue
            elif len(self.memory) < self.memory_size:
                self.memory.append(node)
            else:
                if random.random() > self.p:
                    continue
                replace_idx = random.randint(0, self.memory_size - 1)
                self.memory[replace_idx] = node


    def _update_class(self, nodes, y, importance):
        # update memory
        self.data_size += len(nodes)
        for i in y:
            self.data_size_per_class[i] += 1
        for i in self.data_size_per_class:
            self.memory_size_per_class[i] = int(self.data_size_per_class[i] / self.data_size * self.memory_size)
            while self.memory_size_per_class[i] < len(self.memory_per_class[i]):
                replace_idx = random.randint(0, len(self.memory_per_class[i]) - 1)
                del(self.memory_per_class[i][replace_idx])
                del(self.memory_per_class_log[i][replace_idx])
        for i, node in enumerate(nodes):
            if node in self.memory_per_class[y[i]]:
                continue
            elif self.memory_size_per_class[y[i]] > len(self.memory_per_class[y[i]]):
                self.memory_per_class[y[i]] += [node]
                self.memory_per_class_log[y[i]] += [(node, int(y[i]), self.clock, importance[i])]
            else:
                prob = self.p * self.memory_size_per_class[y[i]] / self.data_size_per_class[y[i]] * (1 + self.alpha * importance[i])
                if random.random() > prob:
                    continue
                replace_idx = random.randint(0, len(self.memory_per_class[y[i]]) - 1)
                self.memory_per_class[y[i]][replace_idx] = node 
                self.memory_per_class_log[y[i]][replace_idx] = (node, int(y[i]), self.clock, float(importance[i]))
        
        self.memory.clear()
        for i, m in self.memory_per_class.items():
            self.memory += m
              
            
    def _compute_node_importance(self, nodes, labels, g):
        graph = copy.deepcopy(g).to(self.device)
        with graph.local_scope():
            _labels = torch.LongTensor(labels).to(self.device)
            if _labels.dim() == 1:
                srcs, dsts = graph.edges()
                graph.srcdata['_valid_label'] = (_labels != -1).float()
                graph.edata['_differents'] = (_labels[srcs] != _labels[dsts]).float()
                graph.update_all(fn.copy_u('_valid_label', '_m1'), fn.sum('_m1', '_n_valid_nbrs'))
                graph.update_all(fn.u_mul_e('_valid_label', '_differents', '_m2'), fn.sum('_m2', '_n_diff_nbrs'))
                _importance = (graph.dstdata['_n_diff_nbrs'] / graph.dstdata['_n_valid_nbrs'])[nodes]
            else:
                srcs, dsts = graph.edges()
                graph.srcdata['_valid_label'] = (_labels != -1).float()
                graph.edata['_differents'] = (_labels[srcs] != _labels[dsts]).float()
                graph.update_all(fn.copy_u('_valid_label', '_m1'), fn.sum('_m1', '_n_valid_nbrs'))
                graph.update_all(fn.u_mul_e('_valid_label', '_differents', '_m2'), fn.sum('_m2', '_n_diff_nbrs'))
                _importance = (graph.dstdata['_n_diff_nbrs'] / graph.dstdata['_n_valid_nbrs'])[nodes]
        _importance = (_importance * 10 - 5)
        _importance = 1 / (1 + torch.exp(-_importance))
        return _importance.detach().cpu().numpy().tolist()

class EWC(nn.Module):
    def __init__(self, model, ewc_lambda = 0, ewc_type = 'ewc'):
        super(EWC, self).__init__()
        self.backbone_model = False
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.ewc_type = ewc_type

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params(self, g, X, mask, labels, _loss_fn):
        preds = self.model(g, X)
        log_likelihood = _loss_fn(preds[mask], labels[mask])
        grad_log_likelihood = autograd.grad(log_likelihood, self.model.parameters(), allow_unused=True)
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_likelihood):
            if param == None:
                continue
            self.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, g, X, mask, labels, _loss_fn=nn.CrossEntropyLoss()):
        self._update_fisher_params(g, X, mask, labels, _loss_fn)
        self._update_mean_params()

    def _compute_consolidation_loss(self):
        losses = []
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
            if self.ewc_type == 'l2':
                losses.append((10e-6 * (param - estimated_mean) ** 2).sum())
            else:
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
        return 1 * (self.ewc_lambda / 2) * sum(losses)

    def forward(self, g, X):
        return self.model(g, X)

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)