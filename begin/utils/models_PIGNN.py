import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
from torch.nn import init
from dgl.utils import expand_as_pair
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
        self.lins = nn.ModuleList([])
        self.input_lengths = []
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
        device = self.lins[-1].weight.data.device
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
                
        self.observed = self.observed.to(device) | new_output_mask

    def expand_parameters(self, in_feats_delta, device):
        self.lins.append(nn.Linear(in_feats_delta, self.num_outputs, self.bias).to(device))
        self.input_lengths.append(in_feats_delta)
        return list(self.lins[-1].parameters())
        
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
        xs = torch.split(x, self.input_lengths, dim=-1)
        out = 0.
        for _x, lin in zip(xs, self.lins):
            out = out + lin(_x)
        
        if task_masks is None:
            out[..., ~self.observed] = -1e12
        else:
            out[~task_masks] = -1e12
        
        return out 
        
class ProgressiveGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=False,
                 activation=None,
                 allow_zero_in_degree=False, 
                negative_slope = 0.2):
        super(ProgressiveGraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.weights = nn.ParameterList()
        self.norms = nn.ModuleList()
        self.in_feats_len = []
        self.out_feats_len = []
        self._activation = activation

    def expand_parameters(self, in_feats_delta, out_feats_delta, device):
        if len(self.in_feats_len) == 0:
            self.in_feats_len.append(in_feats_delta)
        else:
            self.in_feats_len.append(self.in_feats_len[-1] + in_feats_delta)
        self.out_feats_len.append(out_feats_delta)
        self.weights.append(nn.Parameter(torch.Tensor(self.in_feats_len[-1], self.out_feats_len[-1])).to(device))
        self.norms.append(nn.BatchNorm1d(self.out_feats_len[-1]).to(device))
        self.reset_parameters(self.weights[-1])
        return [self.weights[-1]] + list(self.norms[-1].parameters())
        
    def reset_parameters(self, target_weight=None):
        if target_weight is not None:
            init.xavier_uniform_(target_weight)
        else:
            pass
            
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
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

            """
            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight
            """
            
            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                transformed_feats = []
                for w, feat_len in zip(self.weights, self.in_feats_len):
                    transformed_feats.append(torch.matmul(feat_src[..., :feat_len], w))
                feat_src = torch.cat(transformed_feats, dim=-1)
                    
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                
                transformed_rsts = []
                for w, feat_len in zip(self.weights, self.in_feats_len):
                    transformed_rsts.append(torch.matmul(rst[..., :feat_len], w))
                rst = torch.cat(transformed_rsts, dim=-1)
                # rst = torch.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # if self.bias is not None:
            #     rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            rsts = torch.split(rst, self.out_feats_len, dim=-1)
            final_rst = torch.cat([bn(_rst) for _rst, bn in zip(rsts, self.norms)], dim=-1)
            return final_rst
            
    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
        
            
class GCN(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=3, incr_type='class', use_classifier=True, num_tasks = 0):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.not_empty = nn.Parameter(torch.FloatTensor([0]))
        self.convs = nn.ModuleList()
        # self.norms = nn.ModuleList()

        original_num_params = in_feats * n_hidden + n_hidden * n_hidden * (n_layers - 1) + 2 * n_layers * n_hidden + n_hidden * n_classes + n_classes
        while True:
            curr_num_params = in_feats * n_hidden + (num_tasks + 1) / (2 * num_tasks) * n_hidden * n_hidden * (n_layers - 1) + 2 * n_layers * n_hidden + num_tasks * n_classes + n_hidden * n_classes
            if curr_num_params < original_num_params:
                n_hidden += 1
            else:
                break
        self.n_hidden = n_hidden
        print(self.n_hidden)
        
        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden
            self.convs.append(ProgressiveGraphConv(in_hidden, out_hidden, "both", bias=False, allow_zero_in_degree=True))
            # self.norms.append(nn.BatchNorm1d(out_hidden))
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
            # h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        if self.classifier is not None:
            h = self.classifier(h, task_masks)
        return h

    def expand_parameters(self, delta, device):
        new_parameters = []
        for i, conv in enumerate(self.convs):
            if i == 0:
                new_parameters = new_parameters + conv.expand_parameters(0 if (len(conv.weights) > 0) else conv._in_feats, delta, device)
                print(conv.in_feats_len)
            else:
                new_parameters = new_parameters + conv.expand_parameters(delta, delta, device)
        new_parameters = new_parameters + self.classifier.expand_parameters(delta, device)
            
        return new_parameters
        
    def observe_labels(self, new_labels, verbose=True):
        self.classifier.observe_outputs(new_labels, verbose=verbose)
    
    def get_observed_labels(self, tid=None):
        if tid is None or tid < 0:
            return self.classifier.observed
        else:
            return self.classifier.output_masks[tid]