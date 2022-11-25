import torch
from begin.algorithms.mas.links import *
from begin.scenarios.links import LCScenarioLoader
from begin.utils import GCNLink

scenario = LCScenarioLoader(dataset_name='bitcoin', num_tasks=7, metric='rocauc', save_path='data', incr_type='time', task_shuffle=1)
model = GCNLink(scenario.num_feats, scenario.num_classes, 256)
benchmark = LCTimeILMASTrainer(model = model,
                               scenario = scenario,
                               optimizer_fn = lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=0),
                               loss_fn = lambda preds, gt: torch.nn.BCEWithLogitsLoss()(preds, gt.float()),
                               device = torch.device('cuda:0'),
                               scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=20, min_lr=1e-3 * 0.001 * 2., verbose=False),
                               benchmark = True, seed = 42)
results = benchmark.run(epoch_per_task = 200)