import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from .models import AdaptiveLinear
from dgl.nn.pytorch import edge_softmax
from torch.nn import init
from dgl.utils import expand_as_pair
from torch_scatter import segment_csr

class TWPGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=False,
                 activation=None,
                 allow_zero_in_degree=False, 
                negative_slope = 0.2):
        super(TWPGraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.leaky_relu = nn.LeakyReLU(negative_slope)

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

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, return_elist, weight=None, edge_weight=None):
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
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = dgl.utils.expand_as_pair(feat, graph)
            
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
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
                    
                if return_elist:
                    graph.srcdata['rst_h_src'] = feat_src
                    graph.dstdata['rst_h_dst'] = torch.tanh(feat_dst)
                    graph.apply_edges(fn.u_mul_v('rst_h_src', 'rst_h_dst', 'new_e'))
                    new_e = self.leaky_relu(torch.sum(graph.edata.pop('new_e'), 1))
                    # print(e.shape, new_e.shape, (new_e - e).abs().sum())
                    e_soft = dgl.nn.functional.edge_softmax(graph, new_e)
            
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                
                if return_elist:
                    _weighted = torch.matmul(feat_src, weight)
                    _weighted_dst = torch.matmul(feat_dst, weight)
                    graph.srcdata['rst_h_src'] = _weighted
                    graph.dstdata['rst_h_dst'] = torch.tanh(_weighted_dst)
                    graph.apply_edges(fn.u_mul_v('rst_h_src', 'rst_h_dst', 'new_e'))
                    new_e = self.leaky_relu(torch.sum(graph.edata.pop('new_e'), 1))
                    # print(e.shape, new_e.shape, (new_e - e).abs().sum())
                    e_soft = dgl.nn.functional.edge_softmax(graph, new_e)
            
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm in ['right', 'both']:
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
            
            if return_elist:
                return rst, [e_soft]
            else:
                return rst
            
    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
    
class TWPGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm='both', weight=True, bias=True, activation=None, allow_zero_in_degree=False, edge_encoder_fn=None, negative_slope = 0.2):
        super(TWPGCNConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
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

    def forward(self, graph, feat, return_elist=False, weight=None, edge_weight=None, edge_attr=None):
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
                
                if return_elist:
                    graph.srcdata['rst_h_src'] = feat_src
                    graph.dstdata['rst_h_dst'] = torch.tanh(feat_src)
                    graph.apply_edges(fn.u_mul_v('rst_h_src', 'rst_h_dst', 'new_e'))
                    new_e = self.leaky_relu(torch.sum(graph.edata.pop('new_e'), 1))
                    # print(e.shape, new_e.shape, (new_e - e).abs().sum())
                    e_soft = dgl.nn.functional.edge_softmax(graph, new_e)
                    
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                
                if return_elist:
                    _weighted = torch.matmul(feat_src, weight)
                    graph.srcdata['rst_h_src'] = _weighted
                    graph.dstdata['rst_h_dst'] = torch.tanh(_weighted)
                    graph.apply_edges(fn.u_mul_v('rst_h_src', 'rst_h_dst', 'new_e'))
                    new_e = self.leaky_relu(torch.sum(graph.edata.pop('new_e'), 1))
                    # print(e.shape, new_e.shape, (new_e - e).abs().sum())
                    e_soft = dgl.nn.functional.edge_softmax(graph, new_e)
                    
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

            if return_elist:
                return rst, [e_soft]
            else:
                return rst

    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
    
class GCN(nn.Module):
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
            self.convs.append(TWPGraphConv(in_hidden, out_hidden, "both", bias=False, allow_zero_in_degree=True))
            self.norms.append(nn.BatchNorm1d(out_hidden))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        if use_classifier:
            self.classifier = AdaptiveLinear(n_hidden, n_classes, bias=True, accum = False if incr_type == 'task' else True)
        else:
            self.classifier = None
            
    def forward(self, graph, feat, return_elist=False, task_masks=None):
        h = feat
        h = self.dropout(h)
        e_list = []
        for i in range(self.n_layers):
            if return_elist:
                h, e = self.convs[i](graph, h, return_elist=return_elist)
                e_list = e_list + e
            else:
                h = self.convs[i](graph, h, return_elist=return_elist)
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        
        if self.classifier is not None:
            h = self.classifier(h, task_masks)
            
        if return_elist:
            return h, e_list
        else:
            return h
    
    def bforward(self, graph, feat, return_elist=False, task_masks=None):
        h = feat
        h = self.dropout(h)
        e_list = []
        for i in range(self.n_layers):
            if return_elist:
                h, e = self.convs[i](graph[i], h, return_elist=return_elist)
                e_list = e_list + e
            else:
                h = self.convs[i](graph[i], h, return_elist=return_elist)
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        
        if self.classifier is not None:
            h = self.classifier(h, task_masks)
            
        if return_elist:
            return h, e_list
        else:
            return h
    
    def observe_labels(self, new_labels, verbose=True):
        self.classifier.observe_outputs(new_labels, verbose=verbose)
    
    def get_observed_labels(self, tid=None):
        if tid is None or tid < 0:
            return self.classifier.observed
        else:
            return self.classifier.output_masks[tid]
        
class GCNEdge(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=3, incr_type='class'):
        super().__init__()
        self.gcn = GCN(in_feats, n_hidden, n_hidden, activation = F.relu, dropout=0.0, n_layers=3, incr_type='class', use_classifier=False)
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
            
    def forward(self, graph, feat, srcs, dsts, return_elist=False, task_masks=None):
        if return_elist:
            _h, e_list = self.gcn(graph, feat, return_elist=return_elist, task_masks=task_masks)
        else:
            _h = self.gcn(graph, feat, return_elist=return_elist, task_masks=task_masks)
        x = _h[srcs] * _h[dsts]
        x = self.dropout(x)
        for i in range(self.n_layers - 1):
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.classifier(x, task_masks)
        if return_elist:
            return x, e_list
        else:
            return x
        
    def observe_labels(self, new_labels, verbose=True):
        self.classifier.observe_outputs(new_labels, verbose=verbose)
    
    def get_observed_labels(self, tid=None):
        if tid is None or tid < 0:
            return self.classifier.observed
        else:
            return self.classifier.output_masks[tid]

class FullGCN(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=4, n_mlp_layers=2, incr_type='class', readout='mean', node_encoder_fn=None, edge_encoder_fn=None):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        if node_encoder_fn is None:
            self.enc = nn.Linear(in_feats, n_hidden)
        else:
            self.enc = node_encoder_fn()
        for i in range(n_layers):
            in_hidden = n_hidden
            out_hidden = n_hidden
            self.convs.append(TWPGCNConv(in_hidden, out_hidden, "both", bias=False, allow_zero_in_degree=True, edge_encoder_fn=edge_encoder_fn))
            self.norms.append(nn.BatchNorm1d(out_hidden))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.mlp_layers = nn.ModuleList([nn.Linear(n_hidden // (1 << i), n_hidden // (1 << (i+1))) for i in range(n_mlp_layers)])
        self.classifier = AdaptiveLinear(n_hidden // (1 << n_mlp_layers), n_classes, bias=True, accum = False if incr_type == 'task' else True)
        self.readout_mode = readout
        
    def forward(self, graph, feat, return_elist=False, task_masks=None, edge_weight=None, edge_attr=None):
        h = self.enc(feat)
        h = self.dropout(h)
        e_list = []
        for i in range(self.n_layers):
            raw_h = self.convs[i](graph, h, return_elist=return_elist, edge_weight=edge_weight, edge_attr=edge_attr)
            if return_elist:
                h, e = raw_h 
                e_list = e_list + e
            else:
                h = raw_h
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
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
            
        h = self.classifier(h, task_masks)
        if return_elist:
            return h, e_list
        else:
            return h
    
    def observe_labels(self, new_labels, verbose=True):
        self.classifier.observe_outputs(new_labels, verbose=verbose)
    
    def get_observed_labels(self, tid=None):
        if tid is None or tid < 0:
            return self.classifier.observed
        else:
            return self.classifier.output_masks[tid]