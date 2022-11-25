import torch
from begin.algorithms.twp.nodes import *
from begin.scenarios.nodes import NCScenarioLoader
from begin.utils.models_TWP import GCN
scenario = NCScenarioLoader(dataset_name='ogbn-arxiv', num_tasks=8, metric='accuracy', save_path='data', incr_type='class', task_shuffle=1)
model = GCN(scenario.num_feats, scenario.num_classes, 256, dropout=0.0)
benchmark = NCClassILTWPTrainer(model = model,
                                scenario = scenario,
                                optimizer_fn = lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=0),
                                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1),
                                device = torch.device('cuda:0'),
                                scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=20, min_lr=1e-3 * 0.001 * 2., verbose=False),
                                benchmark = True, seed = 42, lambda_l=10000, lambda_t=1000, beta=0.01)
results = benchmark.run(epoch_per_task = 1000)