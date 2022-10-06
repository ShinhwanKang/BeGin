import sys
import logging
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import GCNNode
from dgl.nn import GraphConv

class GCN(GCNNode):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=3, incr_type='class'):
        super().__init__(in_feats, n_classes, n_hidden, activation, dropout, n_layers, incr_type)
        self.backbone_model = True