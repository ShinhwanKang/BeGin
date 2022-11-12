import torch
import numpy as np
import pickle
import copy
import dgl
import random
from .common import BaseTrainer

class NCTrainer(BaseTrainer):
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
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_loss'] = 1e10
        self._reset_optimizer(curr_optimizer)
        # it enables to predict the new classes from the current task
        curr_model.observe_labels(curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']])    
    
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
        return {'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        optimizer.zero_grad()
        self.beforeInference(model, optimizer, _curr_batch, training_states)
        inference_results = self.inference(model, _curr_batch, training_states)
        return self.afterInference(inference_results, model, optimizer, _curr_batch, training_states)
    
    def processEvalIteration(self, model, _curr_batch):
        results = self.inference(model, _curr_batch, None)
        return torch.argmax(results['preds'], dim=-1), {'loss': results['loss'].item()}
    
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
            
        print('init_acc:', init_acc)
        print('algo_acc_mat:', algo_acc_mat)
        print('AP:', round(results['exp_AP'], 4))
        print('AF:', round(results['exp_AF'], 4))
        print('FWT:', round(results['exp_FWT'], 4))
        if self.full_mode:
            print('base_acc_mat:', base_acc_mat)
            print('joint_acc_mat:', accum_acc_mat)
            print('intransigence:', round((accum_acc_mat - algo_acc_mat)[np.arange(self.num_tasks), np.arange(self.num_tasks)].mean(), 4))
        