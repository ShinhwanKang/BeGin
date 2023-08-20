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
                ('nyctaxi', 'time'): (12, 'accuracy', 100, 10, 0.01),
                ('sentiment', 'time'): (11, 'accuracy', 100, 10, 0.01)}

num_memories = {'cora': 12,
                'citeseer': 12,
                'ogbn-arxiv': 2000,
                'mnist': 500,
                'cifar10': 500,
                'aromaticity': 50,
                'ogbn-proteins': 2000,
                'ogbn-products': 25000,
                'ogbl-collab': 20000,
                'ogbg-molhiv': 500,
                'nyctaxi': 180,
                'wikics': 4000,
                'bitcoin': 500,
                'corafull': 210,
                'ogbn-mag': 8000,
                'twitch': 2000,
                'ogbg-ppa': 500,
                'askubuntu': 2000,
                'facebook': 20000,
                'sentiment': 60}

special_kwargs = {'Bare': {},
                  'LwF': {'lamb': None, 'T': 2.},
                  'EWC': {'lamb': None},
                  'MAS': {'lamb': None},
                  'GEM': {'lamb': 0.5, 'num_memories': None},
                  'TWP': {'lambda_l': 10000., 'lambda_t': None, 'beta': 0.01},
                  'ERGNN': {'num_experience_nodes': None, 'sampler_name': 'CM', 'distance_threshold': 0.5},
                  'CGNN': {'detect_strategy': 'bfs', 'memory_strategy': 'class', 'p': 1, 'alpha': 0.0, 'ewc_lambda': 80.0, 'ewc_type': 'ewc', 'memory_size': None, 'new_nodes_size': None},
                  'PackNet': {},
                  'Piggyback': {'threshold': None},
                  'HAT': {'lamb': 0.75, 'smax': 400.}}

special_params = {'Bare': ('none', [None]),
                  'LwF': ('lamb', [1.]),
                  'EWC': ('lamb', [10000.]),
                  'MAS': ('lamb', [1.]),
                  'GEM': ('none', [None]),
                  'TWP': ('lambda_t', [100., 1000.]),
                  'ERGNN': ('none', [None]),
                  'CGNN': ('none', [None]),
                  'PackNet': ('none', [None]),
                  'Piggyback': ('threshold', [1e-1, 1e-2]),
                  'HAT': ('none', [None])}
                       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph CL Benchmark Example')
    parser.add_argument("--dataset-name", type=str, default="cora",
                        help="dataset name (cora, citeseer, ogbn-arxiv, corafull, ogbn-mag, ogbn-products, ogbn-proteins, bitcoin, ogbl-collab, wikics, mnist, cifar10, aromaticity, nyctaxi, ogbg-molhiv)")
    parser.add_argument("--algo", type=str, default="Bare",
                        help="algorithm name (Bare, LwF, EWC, MAS, GEM, TWP, ERGNN, CGNN, PackNet, Piggyback, HAT)") 
    parser.add_argument("--incr", type=str, default="class",
                        help="incremental setting (task, class, domain, or time)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu_id")
    parser.add_argument("--task-type", type=str, default="NC",
                        help="target task (NC, LC, LP, or GC)")
    parser.add_argument("--save-path", type=str, default="./",
                        help="result save path (default: '.')")
    args = parser.parse_args()
    
    _scenario_loader_path = f'begin.scenarios.{task_level[args.task_type]}'
    _scenario_loader_module = f'{args.task_type}ScenarioLoader'
    if args.algo.lower() in ['bare', 'lwf', 'ewc', 'mas', 'gem', 'packnet', 'piggyback', 'hat']:
        _model_path = f'begin.utils.models'
        _model_module = f'GCN{model_suffix[args.task_type]}'
    elif args.algo.lower() in ['twp', 'ergnn', 'cgnn']:
        special_module_name = {'NC': 'GCN', 'LC': 'GCNEdge', 'LP': 'GCNEdge', 'GC': 'FullGCN'}
        _model_path = f'begin.utils.models_{args.algo}'
        _model_module = f'{special_module_name[args.task_type]}'
    _trainer_path = f'begin.algorithms.{args.algo.lower()}.{task_level[args.task_type]}'
    _trainer_module = f'{args.task_type}{args.incr[0].upper()}{args.incr[1:].lower()}IL{args.algo}{"Trainer" if args.dataset_name != "ogbn-products" else "MinibatchTrainer"}'
    
    print("scenario_loader_path:", '.'.join([_scenario_loader_path, _scenario_loader_module]))
    _scenario_loader = getattr(importlib.import_module(_scenario_loader_path), _scenario_loader_module)
    print("model_path:", '.'.join([_model_path, _model_module]))
    _model = getattr(importlib.import_module(_model_path), _model_module)
    print("trainer_path:", '.'.join([_trainer_path, _trainer_module]))
    _trainer = getattr(importlib.import_module(_trainer_path), _trainer_module)
    
    num_task, metric, max_num_epochs, patience, min_scale = exp_settings[(args.dataset_name, args.incr)]
    n_layers, n_hidden = model_settings[args.task_type]
    special_param_name, special_param_range = special_params[args.algo]
    
    lrs = [1e-3, 5e-3, 1e-2]
    drs = [0.0, 0.25, 0.5]
    wds = [0.0, 5e-4]
    seeds = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    
    try:
        log_path = os.path.join(args.save_path, f'benchmark_{args.task_type}_{args.dataset_name}_{args.algo}_{args.incr}')
        os.mkdir(log_path)
    except:
        pass
    
    if args.algo == 'TWP':
        lrs = [1e-3, 5e-3]
        drs = [0.0, 0.25, 0.5]
        wds = [0.0]
        
    print(f"The result will be saved at {log_path} directory (See _result.log for the final results)")
    for lr in lrs: # learning rate
        for dr in drs: # dropout
            for wd in wds: # weight decay
                for special_param in special_param_range:
                    total_val_ap, total_val_af, total_test_ap, total_test_af = [], [], [], []
                    print(f'Current Hyperparameter: lr={lr} dropout={dr} weight_decay={wd} {(str(special_param_name) + "=" + str(special_param)) if special_param_name != "none" else ""}')
                    try:
                        for seed in tqdm.tqdm(seeds):
                            pickle_path = f'{log_path}/result_{lr}_{dr}_{wd}_{str(special_param)}_{seed}.pkl'
                            if not os.path.exists(pickle_path):
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

                                if args.task_type == 'GC':
                                    edge_encoder_fn = None
                                    if args.dataset_name == 'nyctaxi':
                                        edge_encoder_fn = lambda: torch.nn.Linear(1, n_hidden)
                                    elif args.dataset_name == 'ogbg-molhiv':
                                        edge_encoder_fn = lambda: BondEncoder(emb_dim = n_hidden)
                                    elif args.dataset_name == 'ogbg-ppa':
                                        edge_encoder_fn = lambda: torch.nn.Linear(7, n_hidden)

                                    model = _model(scenario.num_feats,
                                                   scenario.num_classes,
                                                   n_hidden,
                                                   dropout=dr,
                                                   n_layers=n_layers,
                                                   incr_type=args.incr,
                                                   node_encoder_fn = None if args.dataset_name != 'ogbg-molhiv' else (lambda: AtomEncoder(emb_dim = n_hidden)),
                                                   edge_encoder_fn = edge_encoder_fn)
                                else:
                                    model = _model(scenario.num_feats,
                                                   scenario.num_classes,
                                                   n_hidden,
                                                   dropout=dr,
                                                   n_layers=n_layers,
                                                   incr_type=args.incr)

                                algo_kwargs = copy.deepcopy(special_kwargs[args.algo])
                                if special_param_name in algo_kwargs:
                                    algo_kwargs[special_param_name] = special_param
                                if args.algo == 'GEM':
                                    algo_kwargs['num_memories'] = num_memories[args.dataset_name]
                                if args.algo == 'CGNN':
                                    algo_kwargs['memory_size'] = num_memories[args.dataset_name]
                                    algo_kwargs['new_nodes_size'] = num_memories[args.dataset_name]
                                if args.algo == 'ERGNN':
                                    algo_kwargs['num_experience_nodes'] = num_memories[args.dataset_name] // (num_task if args.incr in ['time', 'domain'] else scenario.num_classes)

                                benchmark = _trainer(model = model,
                                                     scenario = scenario,
                                                     optimizer_fn = lambda x: torch.optim.Adam(x, lr=lr, weight_decay=wd),
                                                     loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1) if metric == 'accuracy' else (lambda preds, gt: torch.nn.BCEWithLogitsLoss()(preds, gt.float())),
                                                     device = torch.device(f'cuda:{args.gpu}'),
                                                     scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='max' if args.dataset_name in ['wikics', 'ogbl-collab'] else 'min', patience=patience, min_lr= lr * min_scale * 2., verbose=False),
                                                     benchmark = True, seed = seed, verbose=True, binary = (metric != 'accuracy'), **algo_kwargs)

                                benchmark.run(epoch_per_task = max_num_epochs)

                                shutil.copy(f'{benchmark.result_path}/{benchmark.save_file_name}.pkl', f'{log_path}/result_{lr}_{dr}_{wd}_{str(special_param)}_{seed}.pkl')
                            with open(pickle_path, 'rb') as f:    
                                result = pickle.load(f)
                                total_val_ap.append(result['exp_val'][-1][:-1].sum() / result['exp_val'].shape[0])
                                total_test_ap.append(result['exp_test'][-1][:-1].sum() / result['exp_test'].shape[0])
                                total_val_af.append((result['exp_val'][np.arange(result['exp_val'].shape[0]), np.arange(result['exp_val'].shape[0])] - result['exp_val'][-1, :-1]).sum() / (result['exp_val'].shape[0] - 1))
                                total_test_af.append((result['exp_test'][np.arange(result['exp_test'].shape[0]), np.arange(result['exp_test'].shape[0])] - result['exp_test'][-1, :-1]).sum() / (result['exp_test'].shape[0] - 1))
                        with open(f'{log_path}/_result.log', 'a') as f_log:
                            f_log.write(f'{args.dataset_name}_{args.algo}_{args.incr}_lr={lr}_dropout={dr}_weightdecay={wd}_{special_param_name}={str(special_param)} val_AP: {np.round(np.mean(total_val_ap), 4)}±{np.round(np.std(total_val_ap, ddof=1), 4)} test_AP: {np.round(np.mean(total_test_ap), 4)}±{np.round(np.std(total_test_ap, ddof=1), 4)} val_AF: {np.round(np.mean(total_val_af), 4)}±{np.round(np.std(total_val_af, ddof=1), 4)} test_AF: {np.round(np.mean(total_test_af), 4)}±{np.round(np.std(total_test_af, ddof=1), 4)}\n')
                            f_log.flush()
                    except:
                        print(ee)
                        pass