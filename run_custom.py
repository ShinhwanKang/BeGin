import torch
from begin.algorithms.ewc.nodes import *
from begin.scenarios.nodes import NCScenarioLoader
from begin.utils import GCNNode

import dgl

def dataset_load_func(save_path):
    dataset = dgl.data.CoraGraphDataset(raw_dir=save_path, verbose=False)
    graph = dataset._g
    num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
    return {'graph': graph, 'num_classes': num_classes, 'num_feats': num_feats}

scenario = NCScenarioLoader(dataset_name='customcora', dataset_load_func=dataset_load_func, num_tasks=3, metric='accuracy', save_path='data', incr_type='class', task_shuffle=1)


model = GCNNode(scenario.num_feats, scenario.num_classes, 256, dropout=0.0)
benchmark = NCClassILEWCTrainer(model = model,
                                scenario = scenario,
                                optimizer_fn = lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=0),
                                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1),
                                device = torch.device('cuda:0'),
                                scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=20, min_lr=1e-3 * 0.001 * 2., verbose=False),
                                benchmark = True, seed = 42)
results = benchmark.run(epoch_per_task = 1000)