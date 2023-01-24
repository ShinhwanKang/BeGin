"""
import torch
from begin.algorithms.packnet.nodes import *
from begin.scenarios.nodes import NCScenarioLoader
from begin.utils import GCNNode

scenario = NCScenarioLoader(dataset_name='cora', num_tasks=3, metric='accuracy', save_path='data', incr_type='task', task_shuffle=1)
model = GCNNode(scenario.num_feats, scenario.num_classes, 256, dropout=0.0)
benchmark = NCTaskILPackNetTrainer(model = model,
                                scenario = scenario,
                                optimizer_fn = lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=0),
                                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1),
                                device = torch.device('cuda:0'),
                                scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=20, min_lr=1e-3 * 0.001 * 2., verbose=False),
                                benchmark = True, seed = 42)
results = benchmark.run(epoch_per_task = 1000)
"""

"""
import torch
from begin.algorithms.packnet.links import *
from begin.scenarios.links import LCScenarioLoader
from begin.utils import GCNLink

scenario = LCScenarioLoader(dataset_name='bitcoin', num_tasks=3, metric='accuracy', save_path='data', incr_type='task', task_shuffle=1)
model = GCNLink(scenario.num_feats, scenario.num_classes, 256, dropout=0.0)
benchmark = LCTaskILPackNetTrainer(model = model,
                                scenario = scenario,
                                optimizer_fn = lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=0),
                                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1),
                                device = torch.device('cuda:0'),
                                scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=20, min_lr=1e-3 * 0.001 * 2., verbose=False),
                                benchmark = True, seed = 42)
results = benchmark.run(epoch_per_task = 1000)
"""

import torch
from begin.algorithms.packnet.graphs import *
from begin.scenarios.graphs import GCScenarioLoader
from begin.utils import GCNGraph

scenario = GCScenarioLoader(dataset_name='mnist', num_tasks=3, metric='accuracy', save_path='data', incr_type='task', task_shuffle=1)
model = GCNGraph(scenario.num_feats, scenario.num_classes, 256, dropout=0.0)
benchmark = GCTaskILPackNetTrainer(model = model,
                                scenario = scenario,
                                optimizer_fn = lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=0),
                                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1),
                                device = torch.device('cuda:0'),
                                scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=20, min_lr=1e-3 * 0.001 * 2., verbose=False),
                                benchmark = True, seed = 42)
results = benchmark.run(epoch_per_task = 10)
