from torch import nn
import math
import torch
from torch_scatter import segment_csr

class DiscriminatorGC(nn.Module):
    def __init__(self, n_h):
        super(DiscriminatorGC, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h, c):
        return torch.squeeze(self.f_k(h, c), dim=-1)
        
class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
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

class DGI(nn.Module):
    def __init__(self, encoder):
        super(DGI, self).__init__()
        self.encoder = encoder
        self.discriminator = Discriminator(encoder.n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, graph, features):
        positive = self.encoder.forward_without_classifier(graph, features)
        perm = torch.randperm(graph.number_of_nodes()).to(features.device)
        negative = self.encoder.forward_without_classifier(graph, features[perm])
        summary = torch.sigmoid(positive.mean(dim=0))
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)
        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))
        return l1 + l2
    
class DGIGC(nn.Module):
    def __init__(self, encoder):
        super(DGIGC, self).__init__()
        self.encoder = encoder
        self.discriminator = DiscriminatorGC(encoder.n_hidden)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, graph, features):
        positive = self.encoder.forward_without_classifier(graph, features)
        bnns = graph.batch_num_nodes().tolist()
        cnt = 0
        perms = []
        for i in range(len(bnns)):
            perms.append(torch.randperm(bnns[i]).to(features.device) + cnt)
            cnt += bnns[i]
        perm = torch.cat(perms, dim=0)
        negative = self.encoder.forward_without_classifier(graph, features[perm])
        
        ptrs = torch.cat((torch.LongTensor([0]).to(features.device), torch.cumsum(graph.batch_num_nodes(), dim=-1)), dim=-1)
        reverse_ptrs = torch.cat([i * torch.ones(n, dtype=torch.long) for i, n in enumerate(bnns)], dim=0)
        
        summary = torch.sigmoid(segment_csr(positive, ptrs, reduce='mean'))
        
        perm = torch.randperm(len(bnns)).to(features.device)
        permuted_summary = summary[perm]
        d_positive = self.discriminator(positive, summary[reverse_ptrs])
        d_negative = self.discriminator(positive, permuted_summary[reverse_ptrs])
        l1 = segment_csr(self.loss(d_positive, torch.ones_like(d_positive)), ptrs, reduce='mean').sum()
        l2 = segment_csr(self.loss(d_negative, torch.zeros_like(d_negative)), ptrs, reduce='mean').sum()
        return l1 + l2