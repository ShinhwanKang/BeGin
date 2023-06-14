import torch
import numpy as np
import pickle
import copy
import dgl
import random
from .common import BaseTrainer

class GCTrainer(BaseTrainer):
    r"""
        The trainer for handling graph classification (GC).
        
        Base:
            `BaseTrainer`
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.scheduler_fn = kwargs['scheduler_fn']
        
        # For reproducibility
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        self._dataloader_seed_worker = seed_worker
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {}
    
    def prepareLoader(self, curr_dataset, curr_training_states):
        # dataloader for training dataset
        g_train = torch.Generator()
        g_train.manual_seed(0)
        train_loader = dgl.dataloading.GraphDataLoader(curr_dataset['train'], batch_size=128, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=self._dataloader_seed_worker, generator=g_train)
        
        # dataloader for validation dataset
        g_val = torch.Generator()
        g_val.manual_seed(0)
        val_loader = dgl.dataloading.GraphDataLoader(curr_dataset['val'], batch_size=128, shuffle=False, drop_last=False, num_workers=4, worker_init_fn=self._dataloader_seed_worker, generator=g_val)
        
        # dataloader for test dataset
        g_test = torch.Generator()
        g_test.manual_seed(0)
        test_loader = dgl.dataloading.GraphDataLoader(curr_dataset['test'], batch_size=128, shuffle=False, drop_last=False, num_workers=4, worker_init_fn=self._dataloader_seed_worker, generator=g_test)
        return train_loader, val_loader, test_loader
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        
        curr_model.observe_labels(torch.LongTensor([curr_dataset['train'][i][1] for i in range(len(curr_dataset['train']))]))
        self._reset_optimizer(curr_optimizer)
    
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        pass
    
    def inference(self, model, _curr_batch, training_states):
        graphs, labels = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        results['loss'].backward()
        optimizer.step()
        return {'_num_items': results['preds'].shape[0], 'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[1].to(self.device))}
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        optimizer.zero_grad()
        before_inference_results = self.beforeInference(model, optimizer, _curr_batch, training_states)
        inference_results = self.inference(model, _curr_batch, training_states)
        inference_results['_before_inference'] = before_inference_results
        return self.afterInference(inference_results, model, optimizer, _curr_batch, training_states)
        
    def processEvalIteration(self, model, _curr_batch):
        results = self.inference(model, _curr_batch, None)
        return torch.argmax(results['preds'], dim=-1), {'_num_items': results['preds'].shape[0], 'loss': results['loss'].item(),
                                                        'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[1].to(self.device))}
    
    def processTrainingLogs(self, task_id, epoch_cnt, val_metric_result, train_stats, val_stats):
        print('task_id:', task_id, f'Epoch #{epoch_cnt}:', 'train_acc:', round(train_stats['acc'], 4), 'val_acc:', round(val_metric_result, 4), 'train_loss:', round(train_stats['loss'], 4), 'val_loss:', round(val_stats['loss'], 4))
    
    def processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        val_loss = curr_iter_results['val_stats']['loss']
        # maintain the best parameter
        if val_loss < curr_training_states['best_val_loss']:
            curr_training_states['best_val_loss'] = val_loss
            curr_training_states['best_weights'] = copy.deepcopy(curr_model.state_dict())
        
        # integration with scheduler
        scheduler = curr_training_states['scheduler']
        scheduler.step(val_loss)
        # stopping criteria for training
        if -1e-9 < (curr_optimizer.param_groups[0]['lr'] - scheduler.min_lrs[0]) < 1e-9:
            # earlystopping!
            return False
        return True
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        # before measuring the test performance, we need to set the best parameters
        curr_model.load_state_dict(curr_training_states['best_weights'])
    
    def run(self, epoch_per_task=1):
        results = super().run(epoch_per_task)
        
        # dump the results as pickle
        with open(f'{self.result_path}/{self.save_file_name}.pkl', 'wb') as f:
            pickle.dump({k: v.detach().cpu().numpy() for k, v in results.items() if 'val' in k or 'test' in k}, f)
        if self.full_mode:
            init_acc, accum_acc_mat, base_acc_mat, algo_acc_mat = map(lambda x: results[x].detach().cpu().numpy(), ('init_test', 'accum_test', 'base_test', 'exp_test'))
        else:
            init_acc, algo_acc_mat = map(lambda x: results[x].detach().cpu().numpy(), ('init_test', 'exp_test'))
            
        if self.verbose:
            print('init_acc:', init_acc[:-1])
            print('algo_acc_mat:', algo_acc_mat[:, :-1])
            print('AP:', round(results['exp_AP'], 4))
            print('AF:', round(results['exp_AF'], 4))
            if results['exp_FWT'] is not None: print('FWT:', round(results['exp_FWT'], 4))
            if self.full_mode:
                print('joint_acc_mat:', accum_acc_mat[:, :-1])
                print('intransigence:', round((accum_acc_mat - algo_acc_mat)[np.arange(self.num_tasks), np.arange(self.num_tasks)].mean(), 4))
        return results