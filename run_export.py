import torch
import random
import numpy as np
import dgl
import argparse
import importlib
import copy
import pickle
import os
import shutil
import tqdm
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

task_level = {'NC': 'nodes', 'LC': 'links', 'LP': 'links', 'GC': 'graphs'}
model_suffix = {'NC': 'Node', 'LC': 'Link', 'LP': 'Link', 'GC': 'Graph'}
model_settings = {'NC': (3, 256), 'LC': (3, 256), 'LP': (3, 256), 'GC': (4, 146)}
exp_settings = {('cora', 'task'): (3, 'accuracy', 1000, 20, 0.001),
                ('cora', 'class'): (3, 'accuracy', 1000, 20, 0.001),
                ('citeseer', 'task'): (3, 'accuracy', 1000, 20, 0.001),
                ('citeseer', 'class'): (3, 'accuracy', 1000, 20, 0.001),
                ('ogbn-arxiv', 'task'): (8, 'accuracy', 1000, 20, 0.001),
                ('ogbn-arxiv', 'class'): (8, 'accuracy', 1000, 20, 0.001),
                ('ogbn-arxiv', 'time'): (24, 'accuracy', 1000, 20, 0.001),
                ('corafull', 'task'): (35, 'accuracy', 1000, 20, 0.001),
                ('ogbn-products', 'class'): (9, 'accuracy', 100, 20, 0.001),
                ('ogbn-mag', 'task'): (128, 'accuracy', 100, 20, 0.001),
                ('ogbn-mag', 'class'): (128, 'accuracy', 100, 20, 0.001),
                ('ogbn-mag', 'time'): (10, 'accuracy', 1000, 20, 0.001),
                ('ogbn-proteins', 'domain'): (8, 'rocauc', 200, 20, 0.001),
                ('twitch', 'domain'): (21, 'accuracy', 200, 20, 0.001),
                ('bitcoin', 'task'): (3, 'accuracy', 1000, 20, 0.001),
                ('bitcoin', 'class'): (3, 'accuracy', 1000, 20, 0.001),
                ('bitcoin', 'time'): (7, 'rocauc', 1000, 20, 0.001),
                ('wikics', 'domain'): (54, 'hits@50', 200, 10, 0.001),
                ('ogbl-collab', 'time'): (50, 'hits@50', 200, 10, 0.01),
                ('askubuntu', 'time'): (69, 'hits@50', 200, 10, 0.01),
                ('facebook', 'domain'): (8, 'hits@50', 200, 10, 0.01),
                ('mnist', 'task'): (5, 'accuracy', 100, 10, 0.01),
                ('mnist', 'class'): (5, 'accuracy', 100, 10, 0.01),
                ('cifar10', 'task'): (5, 'accuracy', 100, 10, 0.01),
                ('cifar10', 'class'): (5, 'accuracy', 100, 10, 0.01),
                ('aromaticity', 'task'): (10, 'accuracy', 100, 10, 0.01),
                ('aromaticity', 'class'): (10, 'accuracy', 100, 10, 0.01),
                ('ogbg-molhiv', 'domain'): (20, 'rocauc', 100, 10, 0.01),
                ('ogbg-ppa', 'domain'): (11, 'accuracy', 100, 10, 0.01),
                ('nyctaxi', 'time'): (16, 'accuracy', 100, 10, 0.01),
                ('sentiment', 'time'): (11, 'accuracy', 100, 10, 0.01)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph CL Benchmark Example')
    parser.add_argument("--dataset-name", type=str, default="cora",
                        help="dataset name for export")
    parser.add_argument("--incr", type=str, default="class",
                        help="incremental setting (task, class, domain, or time)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu_id")
    parser.add_argument("--task-type", type=str, default="NC",
                        help="target task (NC, LC, LP, or GC)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    args = parser.parse_args()
    
    _scenario_loader_path = f'begin.scenarios.{task_level[args.task_type]}'
    _scenario_loader_module = f'{args.task_type}ScenarioLoader'
    print("scenario_loader_path:", '.'.join([_scenario_loader_path, _scenario_loader_module]))
    _scenario_loader = getattr(importlib.import_module(_scenario_loader_path), _scenario_loader_module)
    
    num_task, metric, max_num_epochs, patience, min_scale = exp_settings[(args.dataset_name, args.incr)]
    seeds = [args.seed]
        
    for seed in tqdm.tqdm(seeds):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        scenario = _scenario_loader(dataset_name=args.dataset_name,
                                    num_tasks=num_task,
                                    metric=metric,
                                    save_path='data',
                                    incr_type=args.incr,
                                    task_shuffle=1)
        
        torch.save(scenario.export_dataset(full=True), f'/data/begin_exported_data/export_{args.task_type}_{args.dataset_name}_{args.incr}_{args.seed}')