import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
import math
import dgl
from dgl.utils import expand_as_pair
import dgl.function as fn

class PretrainingMethod(nn.Module):
    r""" Base framework for implementing pretraining methods.
    
    Arguments:
        encoder (torch.nn.Module): Pytorch model for pretraining.
    """
    class PretrainIterator:
        r""" Base itearator for pretraining iterations. This class assumes full-batch training.
        """
        def __init__(self, inputs, device):
            self.inputs = inputs
            self.count = 0
            self.device = device
            
        def __iter__(self):
            return self
    
        def __next__(self):
            if self.count == 0:
                self.count = 1
                return self.inputs.to(self.device)
            else:
                raise StopIteration
            
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loss_fn = None
        
    def iterator(self, inputs, device):
        """
            Return iterator for the given input dataset.
            
            Args:
                inputs (object): the input graph dataset.
                device (str): target GPU device.
                
            Returns:
                An iterator for pretraining epoch.
        """
        return self.PretrainIterator(inputs, device)
        
    def inference(self, inputs):
        """
            Return iterator for the given input dataset.
            Implementing this function is mandatory to operate the pretraining procedure.
            
            Args:
                inputs (object): the input sample drawn from iterator.
                
            Returns:
                a scalar which represents loss for pretraining.
        """
        raise NotImplementedError

    def update(self):
        """
            This function is called when the best checkpoint needs to be updated.
            The default implementation stores the current `state_dict` of the model in `self.best_checkpoint`.
        """
        self.best_checkpoint = copy.deepcopy(self.encoder.state_dict())

    def processAfterTraining(self, original_model):
        """
            This function is called once when the trainer concludes pretraining.
            The default implementation initializes the model using the saved best checkpoint (spec., `self.best_checkpoint`) before the main training begins.
        """
        original_model.load_state_dict(self.best_checkpoint)

class DGI(PretrainingMethod):
    r""" An implementation of DGL for node-level and link-level problems. This code was implemented based on the official implementation by authors.
    For the details, see the `original paper <https://arxiv.org/pdf/1809.10341>`_.
    
    Arguments:
        encoder (torch.nn.Module): Pytorch model for pretraining.
    """
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
        print("PRETRAINING_METHOD: DGI")
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

class DGISubgraphCL(PretrainingMethod):
    r""" An implementation of GraphCL (utilized with DGI, subgraph augmentation) for node-level and link-level problems. This code was implemented based on the official implementation by authors.
    For the details, see the `original paper <https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf>`_.

    Arguments:
        encoder (torch.nn.Module): Pytorch model for pretraining.
    """
    
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
        print("PRETRAINING_METHOD: GraphCL (Node/Link)")
        self.discriminator = self.Discriminator(encoder.n_hidden)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.link_level = link_level

    def do_augmentation(self, graph):
        device = graph.ndata['feat'].device
        srcs, dsts = graph.edges()
        n_nodes = graph.num_nodes()
        with torch.no_grad():
            with graph.local_scope():
                root = torch.randint(n_nodes, (1,))
                init_feat = n_nodes * torch.ones(n_nodes).to(device)
                init_feat[root.item()] = 0.
                graph.ndata['hop'] = init_feat
                graph.edata['w'] = torch.ones(graph.num_edges()).to(device)
                
                graph.edata['w'][srcs == dsts] = 0.
                cnt = (graph.ndata['hop'] < (n_nodes - 0.5)).sum().item()
                hop_cnt = 0
                while cnt <= n_nodes * 0.8:
                    graph.update_all(fn.u_add_e('hop', 'w', 'm'), fn.min('m', 'hop'))
                    new_cnt = (graph.ndata['hop'] < (n_nodes - 0.5)).sum().item()
                    hop_cnt += 1
                    if cnt == new_cnt:
                        graph.ndata['hop'][(torch.arange(n_nodes).to(device)[graph.ndata['hop'] > (n_nodes - 0.5)])[torch.randperm(n_nodes - cnt)[0]]] = hop_cnt
                        new_cnt += 1
                    cnt = new_cnt
                target_nodes = torch.argsort(graph.ndata['hop'].long() * n_nodes + torch.randperm(n_nodes).to(device), dim=0)[int(n_nodes * 0.8):]
        new_graph = copy.deepcopy(graph)
        return dgl.remove_nodes(new_graph, target_nodes)
        
    def inference(self, inputs):
        graph, features = inputs, inputs.ndata['feat']
        if self.link_level:
            srcs, dsts = graph.edges()
            positive = self.encoder.forward_without_classifier(graph, features, srcs, dsts)
            perm = torch.randperm(graph.number_of_nodes()).to(features.device)
            negative = self.encoder.forward_without_classifier(graph, features[perm], srcs, dsts)
            
            aug1_graph = self.do_augmentation(graph)
            aug1_srcs, aug1_dsts = aug1_graph.edges()
            aug1 = self.encoder.forward_without_classifier(aug1_graph, aug1_graph.ndata['feat'], aug1_srcs, aug1_dsts)
            aug2_graph = self.do_augmentation(graph)
            aug2_srcs, aug2_dsts = aug2_graph.edges()
            aug2 = self.encoder.forward_without_classifier(aug2_graph, aug2_graph.ndata['feat'], aug2_srcs, aug2_dsts)
            
        else:
            positive = self.encoder.forward_without_classifier(graph, features)
            perm = torch.randperm(graph.number_of_nodes()).to(features.device)
            negative = self.encoder.forward_without_classifier(graph, features[perm])

            aug1_graph = self.do_augmentation(graph)
            aug1 = self.encoder.forward_without_classifier(aug1_graph, aug1_graph.ndata['feat'])
            aug2_graph = self.do_augmentation(graph)
            aug2 = self.encoder.forward_without_classifier(aug2_graph, aug2_graph.ndata['feat'])
            
        summary_aug1 = torch.sigmoid(aug1.mean(dim=0))
        summary_aug2 = torch.sigmoid(aug2.mean(dim=0))
        pos_logit1 = self.discriminator(positive, summary_aug1)
        neg_logit1 = self.discriminator(negative, summary_aug1)
        pos_logit2 = self.discriminator(positive, summary_aug2)
        neg_logit2 = self.discriminator(negative, summary_aug2)
        aug1_loss = self.loss_fn(pos_logit1, torch.ones_like(pos_logit1)) + self.loss_fn(neg_logit1, torch.zeros_like(neg_logit1))
        aug2_loss = self.loss_fn(pos_logit2, torch.ones_like(pos_logit2)) + self.loss_fn(neg_logit2, torch.zeros_like(neg_logit2))
        
        return aug1_loss + aug2_loss

class SubgraphCL(PretrainingMethod):        
    r""" An implementation of GraphCL (subgraph augmentation) for graph-level problems. This code was implemented based on the official implementation by authors.
    For the details, see the `original paper <https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf>`_.

    Arguments:
        encoder (torch.nn.Module): Pytorch model for pretraining.
    """
    
    def __init__(self, encoder):
        super().__init__(encoder)
        print("PRETRAINING_ALGO: GraphCL (Graph)")
        
    def do_augmentation(self, graph):
        device = graph.ndata['feat'].device
        srcs, dsts = graph.edges()
        n_nodes = graph.num_nodes()
        with torch.no_grad():
            with graph.local_scope():
                root = torch.randint(n_nodes, (1,))
                init_feat = n_nodes * torch.ones(n_nodes).to(device)
                init_feat[root.item()] = 0.
                graph.ndata['hop'] = init_feat
                if (srcs == dsts).sum() == 0:
                    graph = graph.add_self_loop()
                    srcs, dsts = graph.edges()
                graph.edata['w'] = torch.ones(graph.num_edges()).to(device)
                graph.edata['w'][srcs == dsts] = 0.
                cnt = (graph.ndata['hop'] < (n_nodes - 0.5)).sum().item()
                hop_cnt = 0
                while cnt <= n_nodes * 0.8:
                    graph.update_all(fn.u_add_e('hop', 'w', 'm'), fn.min('m', 'hop'))
                    new_cnt = (graph.ndata['hop'] < (n_nodes - 0.5)).sum().item()
                    hop_cnt += 1
                    if cnt == new_cnt:
                        graph.ndata['hop'][(torch.arange(n_nodes).to(device)[graph.ndata['hop'] > (n_nodes - 0.5)])[torch.randperm(n_nodes - cnt)[0]]] = hop_cnt
                        new_cnt += 1
                    cnt = new_cnt
                target_nodes = torch.argsort(graph.ndata['hop'].long() * n_nodes + torch.randperm(n_nodes).to(device), dim=0)[int(n_nodes * 0.8):]
        new_graph = copy.deepcopy(graph)
        return dgl.remove_nodes(new_graph, target_nodes.to(torch.int32))
        
    def inference(self, inputs):
        graphs = inputs
        aug_graphs = dgl.batch(list(map(self.do_augmentation, dgl.unbatch(graphs))))

        _, original_outputs = self.encoder(graphs,
                                           graphs.ndata['feat'] if 'feat' in graphs.ndata else None,
                                           edge_attr = graphs.edata['feat'] if 'feat' in graphs.edata else None,
                                           edge_weight = graphs.edata['weight'] if 'weight' in graphs.edata else None,
                                           get_intermediate_outputs=True)

        _, aug_outputs = self.encoder(aug_graphs,
                                      aug_graphs.ndata['feat'] if 'feat' in aug_graphs.ndata else None,
                                      edge_attr = aug_graphs.edata['feat'] if 'feat' in aug_graphs.edata else None,
                                      edge_weight = aug_graphs.edata['weight'] if 'weight' in aug_graphs.edata else None,
                                      get_intermediate_outputs=True)
        
        # compute cl loss
        neg_score = torch.logsumexp(aug_outputs[-1] @ original_outputs[-1].t(), dim=-1).mean()
        pos_score = torch.sum(aug_outputs[-1] * original_outputs[-1], dim=-1).mean()
        loss = -pos_score + neg_score
        return loss


class LightGCL(PretrainingMethod):
    r""" An implementation of LightGCL for node-level and link-level problems. This code was implemented based on the official implementation by authors.
    Note that this method only supports bipartite graphs.
    For the details, see the `original paper <https://arxiv.org/pdf/2302.08191>`_.

    Arguments:
        encoder (torch.nn.Module): Pytorch model for pretraining.
    """
    
    class PretrainIterator(PretrainingMethod.PretrainIterator):
        def __init__(self, inputs, batch_size, samples, device):
            super().__init__(inputs, device)
            self.samples = samples[torch.randperm(samples.shape[0])]
            self.batch_size = batch_size
            
        def __iter__(self):
            return self
    
        def __next__(self):
            if self.count * self.batch_size >= self.samples.shape[0]:
                raise StopIteration
            else:
                self.count += 1
                return (self.inputs.to(self.device), self.samples[(self.count-1) * self.batch_size:self.count * self.batch_size].to(self.device))
                
    def __init__(self, encoder, link_level=True, bipartite=False):
        super().__init__(encoder)
        print("PRETRAINING_ALGO: LightGCL (Bipartite Graph Only)")
        self.link_level = link_level
        self.bipartite = bipartite
        self.batch_size = 4096
        if self.link_level:
            self.target_module = self.encoder.gcn
        else:
            self.target_module = self.encoder
        self.svd_u, self.svd_s, self.svd_v = None, None, None
        
    def exact_forward(self, graph, feats):
        final_h = 0.
        h = feats
        h = self.target_module.dropout(h)
        for i in range(self.target_module.n_layers):
            conv = self.target_module.convs[i](graph, h)
            h = conv
            h = self.target_module.norms[i](h)
            h = self.target_module.activation(h)
            final_h = final_h + h
            h = self.target_module.dropout(h)
        return final_h
        
    def approx_conv(self, conv, graph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            degs = graph.out_degrees().to(feat_src).clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm
            weight = conv.weight
            if conv._in_feats > conv._out_feats:
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                if self.bipartite:
                    updated_feat = torch.cat((self.svd_u.to(feat_src.device) @ ((self.svd_v @ torch.diag(self.svd_s)).t().to(feat_src.device) @ feat_src[self.num_srcs:]),
                                              self.svd_v.to(feat_src.device) @ ((self.svd_u @ torch.diag(self.svd_s)).t().to(feat_src.device) @ feat_src[:self.num_srcs])), dim=0)
                else:
                    updated_feat = self.svd_u.to(feat_src.device) @ ((self.svd_v @ torch.diag(self.svd_s)).t().to(feat_src.device) @ feat_src)
                rst = updated_feat + feat_src
            else:
                # aggregate first then mult W
                if self.bipartite:
                    updated_feat = torch.cat((self.svd_u.to(feat_src.device) @ ((self.svd_v @ torch.diag(self.svd_s)).t().to(feat_src.device) @ feat_src[self.num_srcs:]),
                                          self.svd_v.to(feat_src.device) @ ((self.svd_u @ torch.diag(self.svd_s)).t().to(feat_src.device) @ feat_src[:self.num_srcs])), dim=0)
                else:
                    updated_feat = self.svd_u.to(feat_src.device) @ ((self.svd_v @ torch.diag(self.svd_s)).t().to(feat_src.device) @ feat_src)
                rst = updated_feat + feat_src
                if weight is not None:
                    rst = torch.matmul(rst, weight)
            
            degs = graph.in_degrees().to(feat_dst).clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm
            
        return rst
            
    def approx_forward(self, graph, feats):
        final_h = 0.
        h = feats
        h = self.target_module.dropout(h)
        for i in range(self.target_module.n_layers):
            conv = self.approx_conv(self.target_module.convs[i], graph, h)
            h = conv
            h = self.target_module.norms[i](h)
            h = self.target_module.activation(h)
            final_h = final_h + h
            h = self.target_module.dropout(h)
            
        return final_h

    def iterator(self, inputs, device):
        graph, features = inputs, inputs.ndata['feat']
        if self.svd_s is None:
            if self.link_level and self.bipartite:
                srcs, dsts = graph.edges()
                valid = srcs < dsts
                srcs, dsts = srcs[valid], dsts[valid]
                self.srcs = srcs
                self.dsts = dsts
                # self.srcs = srcs
                # self.dsts = dsts
                self.num_srcs = srcs.max() + 1
                self.num_dsts = dsts.max() + 1 - self.num_srcs
                self.svd_u, self.svd_s, self.svd_v = torch.svd_lowrank(torch.sparse_coo_tensor([srcs.tolist(), (dsts - self.num_srcs).tolist()], torch.ones_like(srcs).float().tolist()), q=5)
            else:
                srcs, dsts = graph.edges()
                self.srcs = srcs
                self.dsts = dsts
                valid = srcs != dsts
                srcs, dsts = srcs[valid], dsts[valid]
                self.svd_u, self.svd_s, self.svd_v = torch.svd_lowrank(torch.sparse_coo_tensor([srcs.tolist(), dsts.tolist()], torch.ones_like(srcs).float().tolist(), (graph.num_nodes(), graph.num_nodes())), q=5)

        if self.bipartite and self.link_level:
            return self.PretrainIterator(inputs, self.batch_size, torch.stack((self.srcs, self.dsts), dim=-1), device)
        else:
            return self.PretrainIterator(inputs, self.batch_size, torch.arange(graph.num_nodes()).to(self.srcs.device), device)
        
    def inference(self, inputs):
        graph, features, targets = inputs[0], inputs[0].ndata['feat'], inputs[1]
        exact_outs = self.exact_forward(graph, features)
        approx_outs = self.approx_forward(graph, features)

        if self.link_level and self.bipartite:
            target_srcs, target_dsts = targets[:, 0], targets[:, 1]
            neg_score = torch.logsumexp(approx_outs[target_srcs] @ exact_outs[:self.num_srcs].t(), dim=-1).mean()
            neg_score = neg_score + torch.logsumexp(approx_outs[target_dsts] @ exact_outs[self.num_srcs:].t(), dim=-1).mean()
            pos_score = torch.sum(approx_outs[target_srcs] * exact_outs[target_srcs], dim=-1).mean()
            pos_score = pos_score + torch.sum(approx_outs[target_dsts] * exact_outs[target_dsts], dim=-1).mean()
            loss = -pos_score + neg_score
        else:
            neg_score = torch.logsumexp(approx_outs[targets] @ exact_outs.t(), dim=-1).mean()
            pos_score = torch.sum(approx_outs[targets] * exact_outs[targets], dim=-1).mean()
            loss = -pos_score + neg_score
        
        return loss
        
class InfoGraph(PretrainingMethod):
    r""" An implementation of InfoGraph for graph-level problems. This code was implemented based on the official implementation by authors.
    For the details, see the `original paper <https://arxiv.org/pdf/1908.01000>`_.

    Arguments:
        encoder (torch.nn.Module): Pytorch model for pretraining.
    """
    
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
        log_2 = math.log(2.0)
        Ep = log_2 - F.softplus(-p_samples)
    
        if average:
            return Ep.mean()
        else:
            return Ep
    
    
    def get_negative_expectation(self, q_samples, average=True):
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
        graphs = inputs
        _, intermediate_outputs = self.encoder(graphs,
                                               graphs.ndata['feat'] if 'feat' in graphs.ndata else None,
                                               edge_attr = graphs.edata['feat'] if 'feat' in graphs.edata else None,
                                               edge_weight = graphs.edata['weight'] if 'weight' in graphs.edata else None,
                                               get_intermediate_outputs=True)
        global_h = intermediate_outputs[-1]
        local_h = self.local_d(torch.cat(intermediate_outputs[:-1], dim=-1))
        graph_id = torch.cat([(torch.ones(_num, dtype=torch.long) * i) for i, _num in enumerate(graphs.batch_num_nodes().tolist())], dim=-1)
        loss = self.local_global_loss_(local_h, global_h, graph_id)
        return loss