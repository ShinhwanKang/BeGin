import torch
from begin.algorithms.lwf.graphs import *
from begin.scenarios.graphs import GCScenarioLoader
from begin.utils import GCNGraph
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

scenario = GCScenarioLoader(dataset_name='aromaticity', num_tasks=10, metric='accuracy', save_path='data', incr_type='task', task_shuffle=1)
model = GCNGraph(scenario.num_feats, scenario.num_classes, 146, dropout=0.0, n_layers = 4)

benchmark = GCTaskILLwFTrainer(model = model,
                               scenario = scenario,
                               optimizer_fn = lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=0),
                               loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1),
                               device = torch.device('cuda:0'),
                               scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=10, min_lr=1e-3 * 0.01 * 2., verbose=False),
                               benchmark = True, seed = 42)
results = benchmark.run(epoch_per_task = 100)