import torch
from torch import nn
from dgl.nn import GraphConv
import torch.nn.functional as F
from .models import GCNNode

class GCN(GCNNode):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=3, incr_type='class'):
        super().__init__(in_feats, n_classes, n_hidden, activation, dropout, n_layers, incr_type)

    def forward(self, graph, feat, task_masks=None):
        h = feat
        h = self.dropout(h)
        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        self.last_h = h
        h = self.classifier(h, task_masks)
        return h
    
    def bforward(self, graph, feat, task_masks=None):
        h = feat
        h = self.dropout(h)
        for i in range(self.n_layers):
            conv = self.convs[i](graph[i], h)
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        self.last_h = h
        h = self.classifier(h, task_masks)
        return h