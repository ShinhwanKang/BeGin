import torch
import numpy as np
import pickle
import copy
import dgl
import random
from .common import BaseTrainer

class NCTrainer(BaseTrainer):
    r"""
        The trainer for handling node classification (NC).
        
        Base:
            `BaseTrainer`
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        # integration with scheduler
        self.scheduler_fn = kwargs['scheduler_fn']
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {}
    
    def prepareLoader(self, curr_dataset, curr_training_states):
        # the default setting for NC is full-batch training
        return [(curr_dataset, curr_dataset.ndata['train_mask'])], [(curr_dataset, curr_dataset.ndata['val_mask'])], [(curr_dataset, curr_dataset.ndata['test_mask'])]
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        # initialize scheduler, optimizer, and best_val_loss
        self._reset_optimizer(curr_optimizer, curr_model)
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_loss'] = 1e10
        
        if self.binary:
            # it uses all outputs for every task
            curr_model.observe_labels(torch.arange(curr_dataset.ndata['label'].shape[-1]))
        else:
            # it enables to predict the new classes from the current task
            curr_model.observe_labels(curr_dataset.ndata['label'][curr_dataset.ndata['train_mask'] | curr_dataset.ndata['val_mask']])    
    
    def predictionFormat(self, results):
        if self.binary:
            return results['preds']
        else:
            return results['preds'].argmax(-1)
    
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        pass
    
    def inference(self, model, _curr_batch, training_states):
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        return {'preds': preds, 'loss': loss}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        results['loss'].backward()
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(self.predictionFormat(results), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        optimizer.zero_grad()
        before_inference_results = self.beforeInference(model, optimizer, _curr_batch, training_states)
        inference_results = self.inference(model, _curr_batch, training_states)
        inference_results['_before_inference'] = before_inference_results
        return self.afterInference(inference_results, model, optimizer, _curr_batch, training_states)
    
    def processEvalIteration(self, model, _curr_batch):
        results = self.inference(model, _curr_batch, None)
        return self.predictionFormat(results), {'loss': results['loss'].item()}
    
    def processTrainingLogs(self, task_id, epoch_cnt, val_metric_result, train_stats, val_stats):
        if epoch_cnt % 10 == 0: print('task_id:', task_id, f'Epoch #{epoch_cnt}:', 'train_acc:', round(train_stats['acc'], 4), 'val_acc:', round(val_metric_result, 4), 'train_loss:', round(train_stats['loss'], 4), 'val_loss:', round(val_stats['loss'], 4))
        pass
        
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
    
class NCMinibatchTrainer(NCTrainer):
    r"""
        The mini-batch trainer (with neighborhood sampler) for handling node classification (NC).
        
        Base:
            ``NCTrainer``
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.scheduler_fn = kwargs['scheduler_fn']
        
        # for randomness of dataloader
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        self._dataloader_seed_worker = seed_worker
        
    def prepareLoader(self, curr_dataset, curr_training_states):
        dataset = curr_dataset.clone()
        # pop train, val, test masks contained in the dataset
        train_mask = dataset.ndata.pop('train_mask')
        val_mask = dataset.ndata.pop('val_mask')
        test_mask = dataset.ndata.pop('test_mask')
        
        # use multi-layer neighborhood sampler for efficient training
        g_train = torch.Generator()
        g_train.manual_seed(0)
        train_sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 10])
        train_loader = dgl.dataloading.NodeDataLoader(
            dataset, torch.nonzero(train_mask, as_tuple=True)[0], train_sampler,
            batch_size=131072,
            shuffle=True,
            drop_last=False,
            num_workers=1, worker_init_fn=self._dataloader_seed_worker, generator=g_train)
        
        g_val = torch.Generator()
        g_val.manual_seed(0)
        val_sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 10])
        # use fixed order (shuffle=False)
        val_loader = dgl.dataloading.NodeDataLoader(
            dataset, torch.nonzero(val_mask, as_tuple=True)[0], val_sampler,
            batch_size=131072,
            shuffle=False,
            drop_last=False,
            num_workers=1, worker_init_fn=self._dataloader_seed_worker, generator=g_val)
        
        g_test = torch.Generator()
        g_test.manual_seed(0)
        test_sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 10])
        # use fixed order (shuffle=False)
        test_loader = dgl.dataloading.NodeDataLoader(
            dataset, torch.nonzero(test_mask, as_tuple=True)[0], test_sampler,
            batch_size=131072,
            shuffle=False,
            drop_last=False,
            num_workers=3, worker_init_fn=self._dataloader_seed_worker, generator=g_test)
        
        return train_loader, val_loader, test_loader
    
    def inference(self, model, _curr_batch, training_states):
        # inference function for mini-batch 
        input_nodes, output_nodes, blocks = _curr_batch
        blocks = [b.to(self.device) for b in blocks]
        labels = blocks[-1].dstdata['label']
        preds = model.bforward(blocks, blocks[0].srcdata['feat'])
        loss = self.loss_fn(preds, labels)
        return {'preds': preds, 'loss': loss}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        results['loss'].backward()
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[-1][-1].dstdata['label'].to(self.device))}
        
    def processTrainingLogs(self, task_id, epoch_cnt, val_metric_result, train_stats, val_stats):
        if epoch_cnt % 1 == 0: print('task_id:', task_id, f'Epoch #{epoch_cnt}:', 'train_acc:', round(train_stats['acc'], 4), 'val_acc:', round(val_metric_result, 4), 'train_loss:', round(train_stats['loss'], 4), 'val_loss:', round(val_stats['loss'], 4))
        