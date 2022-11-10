import sys
import numpy as np
import torch
import copy
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer

class NCTaskILLwFTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
    
    def inference(self, model, _curr_batch, training_states):
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        return {'preds': preds, 'loss': loss}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states):
        kd_loss = 0.
        if 'prev_model' in training_states:
            for tid in range(training_states['task_id']):
                task_specific_batch = copy.deepcopy(_curr_batch)
                observed_labels = model.get_observed_labels(tid)
                task_specific_mask = observed_labels.unsqueeze(0).repeat(_curr_batch[1].shape[0], 1)
                task_specific_batch[0].ndata['task_specific_mask'] = task_specific_mask.cpu()
                prv_results = self.inference(training_states['prev_model'], task_specific_batch, training_states)
                curr_results = self.inference(model, task_specific_batch, training_states)
                curr_kd_loss = F.softmax(prv_results['preds'] / self.T, dim=-1)
                curr_kd_loss = curr_kd_loss * F.log_softmax(curr_results['preds'] / self.T, dim=-1)
                curr_kd_loss[..., ~observed_labels] = 0.
                kd_loss = kd_loss - curr_kd_loss.sum(-1).mean()
        total_loss = results['loss'] + (self.lamb * kd_loss)
        total_loss.backward()
        optimizer.step()
        return {'loss': total_loss.item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        curr_training_states['task_id'] = task_id
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)

class NCClassILLwFTrainer(NCTrainer):
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
        return {'loss': total_loss.item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[0].ndata['label'][_curr_batch[1]].to(self.device))}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_model.load_state_dict(curr_training_states['best_weights'])
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)
        curr_training_states['prev_observed_labels'] = curr_model.get_observed_labels().clone().detach()\
        
class NCDomainILLwFTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.
        self.T = kwargs['T'] if 'T' in kwargs else 2.
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        curr_batch, mask = _curr_batch
        optimizer.zero_grad()
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].float().to(self.device))
        kd_loss = 0.
        if 'prev_model' in training_states:
            prev_model = training_states['prev_model']
            prv_preds = prev_model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask].detach()
            kd_loss = F.sigmoid((prv_preds / 2) / self.T) * F.logsigmoid((preds / 2) / self.T)
            kd_loss = kd_loss + F.sigmoid((-prv_preds / 2) / self.T) * F.logsigmoid((-preds / 2) / self.T)
            kd_loss = (-kd_loss).sum(-1).mean()
            
        loss = loss + (self.lamb * kd_loss)
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'acc': self.eval_fn(preds, curr_batch.ndata['label'][mask].to(self.device))}
    
    def processEvalIteration(self, model, _curr_batch):
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].float().to(self.device))
        return preds, {'loss': loss.item()}
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        curr_model.observe_labels(torch.arange(curr_dataset.ndata['label'].shape[-1]))
        self._reset_optimizer(curr_optimizer)
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_model.load_state_dict(curr_training_states['best_weights'])
        curr_training_states['prev_model'] = copy.deepcopy(curr_model)
        
class NCTimeILLwFTrainer(NCClassILLwFTrainer):
    pass
