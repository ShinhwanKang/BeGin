import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
import math

class PretrainingMethod(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loss_fn = None
        
    def inference(self, inputs):
        raise NotImplementedError

    def update(self):
        self.best_checkpoint = copy.deepcopy(self.encoder.state_dict())

    def processAfterTraining(self, original_model):
        original_model.load_state_dict(self.best_checkpoint)

class DGI(PretrainingMethod):
    class Discriminator(nn.Module):
        def __init__(self, n_hidden):
            super().__init__()
            self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
            self.reset_parameters()
    
        def uniform(self, size, tensor):
            bound = 1.0 / math.sqrt(size)
            if tensor is not None:
                tensor.data.uniform_(-bound, bound)
    
        def reset_parameters(self):
            size = self.weight.size(0)
            self.uniform(size, self.weight)
    
        def forward(self, features, summary):
            features = torch.matmul(features, torch.matmul(self.weight, summary))
            return features
            
    def __init__(self, encoder, link_level=True):
        super().__init__(encoder)
        print("PRETRAINING_ALGO: DGI")
        self.discriminator = self.Discriminator(encoder.n_hidden)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.link_level = link_level
        
    def inference(self, inputs):
        graph, features = inputs, inputs.ndata['feat']
        if self.link_level:
            srcs, dsts = graph.edges()
            positive = self.encoder.forward_without_classifier(graph, features, srcs, dsts)
            perm = torch.randperm(graph.number_of_nodes()).to(features.device)
            negative = self.encoder.forward_without_classifier(graph, features[perm], srcs, dsts)
        else:
            positive = self.encoder.forward_without_classifier(graph, features)
            perm = torch.randperm(graph.number_of_nodes()).to(features.device)
            negative = self.encoder.forward_without_classifier(graph, features[perm])
            
        summary = torch.sigmoid(positive.mean(dim=0))
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)
        l1 = self.loss_fn(positive, torch.ones_like(positive))
        l2 = self.loss_fn(negative, torch.zeros_like(negative))
        return l1 + l2

class LightGCL(PretrainingMethod):
    def __init__(self, encoder, link_level=True):
        super().__init__(encoder)

    def inference(self, inputs):
        graph, features = inputs, inputs.ndata['feat']
        return None
        
class InfoGraph(PretrainingMethod):
    class FeedforwardNetwork(nn.Module):
        def __init__(self, in_dim, hid_dim, out_dim):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(in_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, out_dim)
            )
            self.jump_con = nn.Linear(in_dim, out_dim)
    
        def forward(self, feat):
            block_out = self.block(feat)
            jump_out = self.jump_con(feat)
            out = block_out + jump_out
            return out

    def get_positive_expectation(self, p_samples, average=True):
        """Computes the positive part of a JS Divergence.
        Args:
            p_samples: Positive samples.
            average: Average the result over samples.
        Returns:
            th.Tensor
        """
        log_2 = math.log(2.0)
        Ep = log_2 - F.softplus(-p_samples)
    
        if average:
            return Ep.mean()
        else:
            return Ep
    
    
    def get_negative_expectation(self, q_samples, average=True):
        """Computes the negative part of a JS Divergence.
        Args:
            q_samples: Negative samples.
            average: Average the result over samples.
        Returns:
            th.Tensor
        """
        log_2 = math.log(2.0)
        Eq = F.softplus(-q_samples) + q_samples - log_2
    
        if average:
            return Eq.mean()
        else:
            return Eq
    
    
    def local_global_loss_(self, l_enc, g_enc, graph_id):
        num_graphs = g_enc.shape[0]
        num_nodes = l_enc.shape[0]
    
        device = g_enc.device
    
        pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
        neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
    
        for nodeidx, graphidx in enumerate(graph_id.tolist()):
            pos_mask[nodeidx][graphidx] = 1.0
            neg_mask[nodeidx][graphidx] = 0.0
    
        res = torch.mm(l_enc, g_enc.t())
    
        E_pos = self.get_positive_expectation(res * pos_mask, average=False).sum()
        E_pos = E_pos / num_nodes
        E_neg = self.get_negative_expectation(res * neg_mask, average=False).sum()
        E_neg = E_neg / (num_nodes * (num_graphs - 1))
    
        return E_neg - E_pos

    
    def __init__(self, encoder):
        super().__init__(encoder)
        print("PRETRAINING_ALGO: InfoGraph")
        self.local_d = self.FeedforwardNetwork(encoder.n_hidden * encoder.n_layers, encoder.n_hidden, encoder.n_hidden // (1 << encoder.n_mlp_layers))
        
    def inference(self, inputs):
        graph, features = inputs, inputs.ndata['feat']
        _, intermediate_outputs = self.encoder(graph, features, get_intermediate_outputs=True)
        global_h = intermediate_outputs[-1]
        local_h = self.local_d(torch.cat(intermediate_outputs[:-1], dim=-1))
        graph_id = torch.cat([(torch.ones(_num, dtype=torch.long) * i) for i, _num in enumerate(graph.batch_num_nodes().tolist())], dim=-1)
        loss = self.local_global_loss_(local_h, global_h, graph_id)
        return loss