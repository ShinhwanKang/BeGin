import torch
from torch import nn
from dgl.nn import GraphConv, SumPooling, AvgPooling, MaxPooling
import torch.nn.functional as F
from dgl.base import DGLError
from dgl.utils import expand_as_pair
import dgl.function as fn
from torch_scatter import segment_csr

class AdaptiveLinear(nn.Module):
    r"""
        The linear layer for helping the continual learning procedure, by masking the outputs.
        
        Arguments:
            in_channels (int): the number of input channels (in_features of `torch.nn.Linear`).
            out_channels (int): the number of output channels (out_features of `torch.nn.Linear`).
            bias (bool): If set to False, the layer will not learn an additive bias.
            accum (bool): If set to True, the layer can also provide the logits for the previously seen outputs.
    """
    def __init__(self, in_channels, out_channels, bias=True, accum=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias)
        self.bias = bias
        self.accum = accum
        self.num_outputs = out_channels
        self.output_masks = None
        self.observed = torch.zeros(out_channels, dtype=torch.bool)
        
    def observe_outputs(self, new_outputs, verbose=True):
        r"""
            Observes the ideal outputs in the training dataset.
            By observing the outputs, the layer determines which outputs (logits) will be masked or not.
            
            Arguments:
                new_outputs (torch.Tensor): the ideal outputs in the training dataset.
        """
        device = self.lin.weight.data.device
        new_outputs = torch.unique(new_outputs)
        new_num_outputs = max(self.num_outputs, new_outputs.max() + 1)
        new_output_mask = torch.zeros(new_num_outputs, dtype=torch.bool).to(device)
        new_output_mask[new_outputs] = True
        
        prv_observed = self.observed
        if self.output_masks is None: self.output_masks = [new_output_mask]
        else:
            if new_num_outputs > self.num_outputs:
                self.output_masks = [torch.cat((output_mask, torch.zeros(new_num_outputs - self.num_outputs, dtype=torch.bool).to(device)), dim=-1) for output_mask in self.output_masks]
            if self.accum: self.output_masks.append(self.output_masks[-1] | new_output_mask)
            else: self.output_masks.append(new_output_mask)    
        
        # if a new class whose index exceeds `self.num_outputs` is observed, expand the output
        if new_num_outputs > self.num_outputs:
            prev_weight, prev_bias = self.lin.weight.data[prv_observed], (self.lin.bias.data[prv_observed] if self.bias else None)
            self.observed = torch.cat((self.observed.to(device), torch.zeros(new_num_outputs - self.num_outputs, dtype=torch.bool).to(device)), dim=-1)
            self.lin = nn.Linear(in_features, new_num_outputs, bias=self.bias)
            self.lin.weight.data[self.observed] = prev_weight
            if self.bias: self.lin.bias.data[self.observed] = prev_bias    
            self.num_outputs = new_num_outputs
        self.observed = self.observed.to(device) | new_output_mask
    
    
    def get_output_mask(self, task_ids=None):
        r"""
            Returns the mask managed by the layer.
            
            Arguments:
                task_ids (torch.Tensor or None): If task_ids is provided, the layer returns the mask for each index of the task. Otherwise, it returns the mask for the last task it observed.
        """
        if task_ids is None:
            # for class-IL, time-IL, and domain-IL
            return self.output_masks[-1]
        else:
            # for task-IL
            mask = torch.zeros(task_ids.shape[0], self.num_outputs).bool().to(task_ids.device)
            observed_mask = task_ids < len(self.output_masks)
            mask[observed_mask] = torch.stack(self.output_masks, dim=0)[task_ids[observed_mask]]
            return mask
    
    def forward(self, x, task_masks=None):
        r"""
            Returns the masked results of the inner linear layer.
            
            Arguments:
                x (torch.Tensor): the input features.
                task_masks (torch.Tensor or None): If task_masks is provided, the layer uses the tensor for masking the outputs. Otherwise, the layer uses the mask managed by the layer.
        """
        out = self.lin(x)
        if task_masks is None:
            out[..., ~self.observed] = -1e12
        else:
            out[~task_masks] = -1e12
        
        return out
    
class GCNNode(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=3, incr_type='class', use_classifier=True):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden
            self.convs.append(GraphConv(in_hidden, out_hidden, "both", bias=False, allow_zero_in_degree=True))
            self.norms.append(nn.BatchNorm1d(out_hidden))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        if use_classifier:
            self.classifier = AdaptiveLinear(n_hidden, n_classes, bias=True, accum = False if incr_type == 'task' else True)
        else:
            self.classifier = None
            
    def forward(self, graph, feat, task_masks=None):
        h = feat
        h = self.dropout(h)
        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        if self.classifier is not None:
            h = self.classifier(h, task_masks)
        return h
    
    def forward_hat(self, graph, feat, hat_masks, task_masks=None):
        h = feat
        h = self.dropout(h)
        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            
            device = h.get_device()
            h=h*hat_masks[i].to(device).expand_as(h)
            
        if self.classifier is not None:
            h = self.classifier(h, task_masks)
        return h
    
    def forward_without_classifier(self, graph, feat, task_masks=None):
        h = feat
        h = self.dropout(h)
        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        return h
    
    def bforward(self, blocks, feat, task_masks=None):
        h = feat
        h = self.dropout(h)
        for i in range(self.n_layers):
            conv = self.convs[i](blocks[i], h)
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        if self.classifier is not None:
            h = self.classifier(h, task_masks)
        return h
    
    def observe_labels(self, new_labels, verbose=True):
        self.classifier.observe_outputs(new_labels, verbose=verbose)
    
    def get_observed_labels(self, tid=None):
        if tid is None or tid < 0:
            return self.classifier.observed
        else:
            return self.classifier.output_masks[tid]

class GCNLink(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=3, incr_type='class'):
        super().__init__()
        self.gcn = GCNNode(in_feats, n_hidden, n_hidden, activation = F.relu, dropout=0.0, n_layers=3, incr_type='class', use_classifier=False)
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.linears = nn.ModuleList()
        for i in range(n_layers - 1):
            in_hidden = n_hidden
            out_hidden = n_hidden
            self.linears.append(nn.Linear(in_hidden, out_hidden))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.classifier = AdaptiveLinear(n_hidden, n_classes, bias=True, accum = False if incr_type == 'task' else True)
            
    def forward(self, graph, feat, srcs, dsts, task_masks=None):
        _h = self.gcn(graph, feat, task_masks)
        x = _h[srcs] * _h[dsts]
        x = self.dropout(x)
        for i in range(self.n_layers - 1):
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.classifier(x, task_masks)
        return x
    
    def forward_hat(self, graph, feat, srcs, dsts, hat_masks=None, task_masks=None):
        _h = self.gcn.forward_hat(graph, feat, hat_masks[:-(self.n_layers - 1)], task_masks)
        x = _h[srcs] * _h[dsts]
        x = self.dropout(x)
        for i in range(self.n_layers - 1):
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.classifier(x, task_masks)
        return x
    
    def forward_without_classifier(self, graph, feat, srcs, dsts, task_masks=None):
        _h = self.gcn(graph, feat, task_masks)
        x = _h[srcs] * _h[dsts]
        x = self.dropout(x)
        for i in range(self.n_layers - 1):
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        return x
    
    def observe_labels(self, new_labels, verbose=True):
        self.classifier.observe_outputs(new_labels, verbose=verbose)
    
    def get_observed_labels(self, tid=None):
        if tid is None or tid < 0:
            return self.classifier.observed
        else:
            return self.classifier.output_masks[tid]

        
class GCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm='both', weight=True, bias=True, activation=None, allow_zero_in_degree=False, edge_encoder_fn=None):
        super(GCNConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self._activation = activation
        
        if edge_encoder_fn is not None:
            self.edge_encoder = edge_encoder_fn()
        
    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None, edge_attr=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                # print("EWI")
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            elif edge_attr is not None:
                assert edge_attr.shape[0] == graph.number_of_edges()
                graph.edata['_edge_attr'] = self.edge_encoder(edge_attr)
                aggregate_fn = fn.u_add_e('h', '_edge_attr', 'm')
                
            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

    
class GCNGraph(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=4, n_mlp_layers=2, incr_type='class', readout='mean', node_encoder_fn=None, edge_encoder_fn=None):
        super().__init__()
        self.n_layers = n_layers
        self.n_mlp_layers = n_mlp_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        if node_encoder_fn is None:
            self.enc = nn.Linear(in_feats, n_hidden)
        else:
            self.enc = node_encoder_fn()
        for i in range(n_layers):
            in_hidden = n_hidden # if i > 0 else in_feats
            out_hidden = n_hidden
            self.convs.append(GCNConv(in_hidden, out_hidden, "both", bias=False, allow_zero_in_degree=True, edge_encoder_fn=edge_encoder_fn))
            self.norms.append(nn.BatchNorm1d(out_hidden))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.mlp_layers = nn.ModuleList([nn.Linear(n_hidden // (1 << i), n_hidden // (1 << (i+1))) for i in range(n_mlp_layers)])
        self.classifier = AdaptiveLinear(n_hidden // (1 << n_mlp_layers), n_classes, bias=True, accum = False if incr_type == 'task' else True)
        self.readout_mode = readout
        
    def forward(self, graph, feat, task_masks=None, edge_weight=None, edge_attr=None, get_intermediate_outputs=False):
        h = self.enc(feat)
        h = self.dropout(h)
        inter_hs = []
        for i in range(self.n_layers):
            conv = self.convs[i](graph, h, edge_weight=edge_weight, edge_attr=edge_attr)
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            inter_hs.append(h)
            
        if self.readout_mode != 'none':
            # h0 = self.readout_fn(graph, h)
            # use deterministic algorithm instead
            ptrs = torch.cat((torch.LongTensor([0]).to(h.device), torch.cumsum(graph.batch_num_nodes(), dim=-1)), dim=-1)
            h1 = segment_csr(h, ptrs, reduce=self.readout_mode)
            # print((h1 - h0).abs().sum()) => 0
            h = h1
        for layer in self.mlp_layers:
            h = layer(h)
            h = self.activation(h)
            h = self.dropout(h)
        inter_hs.append(h)
        
        h = self.classifier(h, task_masks)
        if get_intermediate_outputs:
            return h, inter_hs
        else:
            return h
        
    def forward_hat(self, graph, feat, hat_masks, task_masks=None, edge_weight=None, edge_attr=None):
        h = self.enc(feat)
        h = self.dropout(h)
        device = h.get_device()
        
        h = h * hat_masks[0].to(device).expand_as(h)
        for i in range(self.n_layers):
            conv = self.convs[i](graph, h, edge_weight=edge_weight, edge_attr=edge_attr)
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            h = h * hat_masks[1 + i].to(device).expand_as(h)
            
        if self.readout_mode != 'none':
            ptrs = torch.cat((torch.LongTensor([0]).to(h.device), torch.cumsum(graph.batch_num_nodes(), dim=-1)), dim=-1)
            h1 = segment_csr(h, ptrs, reduce=self.readout_mode)
            h = h1
            
        for layer in self.mlp_layers:
            h = layer(h)
            h = self.activation(h)
            h = self.dropout(h)
            h=h*hat_masks[1 + self.n_layers + i].to(device).expand_as(h)
            
        h = self.classifier(h, task_masks)
        return h
    
    def forward_without_classifier(self, graph, feat, task_masks=None, edge_weight=None, edge_attr=None):
        h = self.enc(feat)
        h = self.dropout(h)
        for i in range(self.n_layers):
            conv = self.convs[i](graph, h, edge_weight=edge_weight, edge_attr=edge_attr)
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        return h
    
    def observe_labels(self, new_labels, verbose=True):
        self.classifier.observe_outputs(new_labels, verbose=verbose)
    
    def get_observed_labels(self, tid=None):
        if tid is None or tid < 0:
            return self.classifier.observed
        else:
            return self.classifier.output_masks[tid]