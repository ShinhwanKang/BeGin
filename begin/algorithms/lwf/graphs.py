import sys
import numpy as np
import torch
import copy
import torch.nn.functional as F
from begin.trainers.graphs import GCTrainer

class GCTaskILLwFTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
    
    def inference(self, model, _curr_batch, training_states):
        graphs, labels, masks = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                      task_masks = masks)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}

    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        kd_loss = 0.
        graphs, labels, _ = _curr_batch
        if 'prev_model' in training_states:
            for tid in range(training_states['task_id']):
                observed_labels = model.get_observed_labels(tid)
                task_specific_mask = observed_labels.unsqueeze(0).repeat(graphs.batch_size, 1)
                task_specific_batch = (graphs, labels, task_specific_mask)
                prv_results = self.inference(training_states['prev_model'], task_specific_batch, training_states)
                curr_results = self.inference(model, task_specific_batch, training_states)
                curr_kd_loss = F.softmax(prv_results['preds'] / self.T, dim=-1)
                curr_kd_loss = curr_kd_loss * F.log_softmax(curr_results['preds'] / self.T, dim=-1)
                curr_kd_loss[..., ~observed_labels] = 0.
                kd_loss = kd_loss - curr_kd_loss.sum(-1).mean()
            
        total_loss = results['loss'] + (self.lamb * kd_loss)
        total_loss.backward()
        optimizer.step()
        return {'_num_items': results['preds'].shape[0],
                'loss': total_loss.item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), labels.to(self.device))}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_training_states['task_id'] = task_id
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)

class GCClassILLwFTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        kd_loss = 0.
        if 'prev_model' in training_states:
            prv_results = self.inference(training_states['prev_model'], _curr_batch, training_states)
            observed_labels = training_states['prev_observed_labels']
            kd_loss = F.softmax(prv_results['preds'][..., observed_labels].detach() / self.T, dim=-1)
            kd_loss = kd_loss * F.log_softmax(results['preds'][..., observed_labels] / self.T, dim=-1)
            kd_loss = (-kd_loss).sum(-1).mean()    
        total_loss = results['loss'] + (self.lamb * kd_loss)
        total_loss.backward()
        optimizer.step()
        return {'_num_items': results['preds'].shape[0], 'loss': total_loss.item(), 'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[1].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_model.load_state_dict(curr_training_states['best_weights'])
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)
        curr_training_states['prev_observed_labels'] = curr_model.get_observed_labels().clone().detach()

class GCDomainILLwFTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        graphs, labels = _curr_batch
        optimizer.zero_grad()
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        
        kd_loss = 0.
        if 'prev_model' in training_states:
            prev_model = training_states['prev_model']
            prv_preds = prev_model(graphs.to(self.device),
                                   graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                                   edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                                   edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None).detach()
            
            kd_loss = F.sigmoid((prv_preds / 2) / self.T) * F.logsigmoid((preds / 2) / self.T)
            kd_loss = kd_loss + F.sigmoid((-prv_preds / 2) / self.T) * F.logsigmoid((-preds / 2) / self.T)
            kd_loss = (-kd_loss).mean()
        loss = loss + (self.lamb * kd_loss)
        
        loss.backward()
        optimizer.step()
        return {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
        
    def processEvalIteration(self, model, _curr_batch):
        graphs, labels = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        return preds, {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_model.load_state_dict(curr_training_states['best_weights'])
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)
        
class GCTimeILLwFTrainer(GCClassILLwFTrainer):
    pass