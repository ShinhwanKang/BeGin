import torch
import numpy as np
import pickle
import copy
import dgl
import random
from .common import BaseTrainer

class LPTrainer(BaseTrainer):
    r"""
        The trainer for handling link prediction (LP).
        
        Base:
            `BaseTrainer`
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.scheduler_fn = kwargs['scheduler_fn']
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        self._dataloader_seed_worker = seed_worker
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {}
    
    def prepareLoader(self, curr_dataset, curr_training_states):
        graph = curr_dataset['graph'].clone()
        node_feats = graph.ndata.pop('feat')
        curr_training_states['graph'] = (graph, node_feats)
        
        datasets = {_split: (curr_dataset[_split]['edge'], curr_dataset[_split]['label']) for _split in ['train', 'val', 'test']}
        train_dataset = torch.utils.data.TensorDataset(*datasets['train'])
        g_train = torch.Generator()
        g_train.manual_seed(0)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= (64 * 1024) // 2, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=self._dataloader_seed_worker, generator=g_train) 
        return train_loader, [(graph, node_feats, *datasets['val'])], [(graph, node_feats, *datasets['test'])]
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_score'] = -1.
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)
    
    def predictionFormat(self, results):
        return results['preds']
        
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        pass
    
    def inference(self, model, _curr_batch, training_states):
        pass
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        pass
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        graph, feats = map(lambda x: x.to(self.device), training_states['graph'])
        edges, labels = map(lambda x: x.to(self.device), _curr_batch)
        optimizer.zero_grad()
        srcs, dsts = edges[:, 0], edges[:, 1]
        neg_dsts = torch.randint(low=0, high=graph.num_nodes(), size=(srcs.shape[0],)).to(self.device)
        preds = model(graph, feats,
                      srcs.repeat(2), torch.cat((edges[:, 1], neg_dsts), dim=0)).squeeze(-1)
        labels = labels.to(self.device)
        loss = self.loss_fn(preds, torch.cat((labels, torch.zeros_like(labels)), dim=0))
        loss.backward()
        optimizer.step()
        return {'_num_items': preds.shape[0], 'loss': loss.item()}
        
    def processEvalIteration(self, model, _curr_batch):
        graph, feats, edges, _ = map(lambda x: x.to(self.device), _curr_batch)
        srcs, dsts = edges[:, 0], edges[:, 1]
        preds = model(graph, feats, srcs, dsts).squeeze(-1)
        return preds, {}
    
    def processTrainingLogs(self, task_id, epoch_cnt, val_metric_result, train_stats, val_stats):
        if epoch_cnt % 1 == 0: print('task_id:', task_id, f'Epoch #{epoch_cnt}:', 'train_loss:', round(train_stats['loss'], 4), 'val_hits:', round(val_metric_result, 4))
    
    def processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        scheduler = curr_training_states['scheduler']
        val_score = curr_iter_results['val_metric']
        if val_score > curr_training_states['best_val_score']:
            curr_training_states['best_val_score'] = val_score
            curr_training_states['best_weights'] = copy.deepcopy(curr_model.state_dict())
        scheduler.step(val_score)
        if -1e-9 < (curr_optimizer.param_groups[0]['lr'] - scheduler.min_lrs[0]) < 1e-9:
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

class LCTrainer(BaseTrainer):
    r"""
        The trainer for handling link classification (LC).
        
        Base:
            `BaseTrainer`
    """
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.scheduler_fn = kwargs['scheduler_fn']
        
    def initTrainingStates(self, scenario, model, optimizer):
        return {}
    
    def prepareLoader(self, _curr_dataset, curr_training_states):
        curr_dataset = copy.deepcopy(_curr_dataset)
        srcs, dsts = curr_dataset.edges()
        labels = curr_dataset.edata.pop('label')
        train_mask = curr_dataset.edata.pop('train_mask')
        val_mask = curr_dataset.edata.pop('val_mask')
        test_mask = curr_dataset.edata.pop('test_mask')
        # print(labels[train_mask].min(), labels[train_mask].max())
        curr_dataset = dgl.add_self_loop(curr_dataset)
        return [(curr_dataset, srcs[train_mask], dsts[train_mask], labels[train_mask])], [(curr_dataset, srcs[val_mask], dsts[val_mask], labels[val_mask])], [(curr_dataset, srcs[test_mask], dsts[test_mask], labels[test_mask])]
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        if self.binary:
            curr_model.observe_labels(torch.LongTensor([0]))
        else:
            curr_model.observe_labels(curr_dataset.edata['label'][curr_dataset.edata['train_mask'] | curr_dataset.edata['val_mask']])
        self._reset_optimizer(curr_optimizer)
    
    def predictionFormat(self, results):
        if self.binary:
            return results['preds']
        else:
            return results['preds'].argmax(-1)
    
    def beforeInference(self, model, optimizer, _curr_batch, training_states):
        pass
    
    def inference(self, model, _curr_batch, training_states):
        curr_batch, srcs, dsts, labels = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), srcs, dsts)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        results['loss'].backward()
        optimizer.step()
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(self.predictionFormat(results), _curr_batch[-1].to(self.device))}
        
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
            
        print('init_acc:', init_acc[:-1])
        print('algo_acc_mat:', algo_acc_mat[:, :-1])
        print('AP:', round(results['exp_AP'], 4))
        print('AF:', round(results['exp_AF'], 4))
        if results['exp_FWT'] is not None: print('FWT:', round(results['exp_FWT'], 4))
        if self.full_mode:
            print('joint_acc_mat:', accum_acc_mat[:, :-1])
            print('intransigence:', round((accum_acc_mat - algo_acc_mat)[np.arange(self.num_tasks), np.arange(self.num_tasks)].mean(), 4))