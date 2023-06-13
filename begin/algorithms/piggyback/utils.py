from torch import nn
import torch.nn.functional as F
import math
import torch
from torch_scatter import segment_csr

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

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

class DGILC(nn.Module):
    def __init__(self, encoder):
        super(DGILC, self).__init__()
        self.encoder = encoder
        self.discriminator = Discriminator(encoder.n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, graph, features, srcs, dsts):
        positive = self.encoder.forward_without_classifier(graph, features, srcs, dsts)
        perm = torch.randperm(graph.number_of_nodes()).to(features.device)
        negative = self.encoder.forward_without_classifier(graph, features[perm], srcs, dsts)
        summary = torch.sigmoid(positive.mean(dim=0))
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)
        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))
        return l1 + l2
    
def get_positive_expectation(p_samples, average=True):
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


def get_negative_expectation(q_samples, average=True):
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


def local_global_loss_(l_enc, g_enc, graph_id):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    device = g_enc.device

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)

    for nodeidx, graphidx in enumerate(graph_id.tolist()):
        pos_mask[nodeidx][graphidx] = 1.0
        neg_mask[nodeidx][graphidx] = 0.0

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos

class FeedforwardNetwork(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(FeedforwardNetwork, self).__init__()
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
    
class InfoGraph(nn.Module):
    def __init__(self, encoder, n_hidden, n_layers, n_mlp_layers):
        super(InfoGraph, self).__init__()
        self.encoder = encoder
        self.local_d = FeedforwardNetwork(n_hidden * n_layers, n_hidden, n_hidden // (1 << n_mlp_layers))
        
    def forward(self, graph, features):
        _, intermediate_outputs = self.encoder(graph, features, get_intermediate_outputs=True)
        
        global_h = intermediate_outputs[-1]
        local_h = self.local_d(torch.cat(intermediate_outputs[:-1], dim=-1))
        graph_id = torch.cat([(torch.ones(_num, dtype=torch.long) * i) for i, _num in enumerate(graph.batch_num_nodes().tolist())], dim=-1)
        loss = local_global_loss_(local_h, global_h, graph_id)

        return loss
    
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def logistic_classify(x, y, device="cpu"):
    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).to(
            device
        ), torch.from_numpy(train_lbls).to(device)
        test_embs, test_lbls = torch.from_numpy(test_embs).to(
            device
        ), torch.from_numpy(test_lbls).to(device)

        log = LogReg(hid_units, nb_classes)
        log = log.to(device)

        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())
    return np.mean(accs)


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if search:
            params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(
                SVC(), params, cv=5, scoring="accuracy", verbose=0
            )
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)


def evaluate_embedding(embeddings, labels, search=False, device="cpu"):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    logreg_accuracy = logistic_classify(x, y, device)
    return logreg_accuracy