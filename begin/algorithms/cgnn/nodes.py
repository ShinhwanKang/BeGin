import sys
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from begin.trainers.nodes import NCTrainer
from .utils import *

class NCTaskILCGNNTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        
        self.device = device
        self.detect_strategy = kwargs['detect_strategy']
        self.memory_size = kwargs['memory_size']
        self.memory_strategy = kwargs['memory_strategy']
        self.ewc_lambda = kwargs['ewc_lambda']
        self.ewc_type = kwargs['ewc_type']
        self.new_nodes_size = kwargs['new_nodes_size']
        self.p = kwargs['p']
        self.alpha = kwargs['alpha']
        
        self.save_file_name = f'{self.save_file_name}_{self.memory_size}_{self.ewc_lambda}_{self.new_nodes_size}_{self.p}_{self.alpha}'
        
    def prepareLoader(self, curr_dataset, curr_training_states):
        _temp = deepcopy(curr_dataset)
        _temp.ndata['train_mask'][curr_training_states['train_nodes']] = True
        return [(_temp, _temp.ndata['train_mask'])], [(_temp, _temp.ndata['val_mask'])], [(_temp, _temp.ndata['test_mask'])]
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        curr_batch, mask = _curr_batch

        optimizer.zero_grad()
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))
        loss = self.loss_fn(preds[mask], curr_batch.ndata['label'][mask].to(self.device))
        if not model.backbone_model:
            loss = loss + model._compute_consolidation_loss()
        loss.backward()
        optimizer.step() 
        
        return {'loss': loss.item(), 'acc': self.eval_fn(preds[mask].argmax(-1), curr_batch.ndata['label'][mask].to(self.device))}

    def processEvalIteration(self, model, _curr_batch):
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        return torch.argmax(preds, dim=-1), {'loss': loss.item()}    
    
    def initTrainingStates(self, scenario, model, optimizer):
        self.num_tasks = scenario.num_tasks
        return {'old_nodes_list':list(), 'train_cha_nodes_list':list(), 'train_nodes':list(), 'sage':None, 'before_g':None}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        if task_id == 0:
            curr_training_states['train_cha_nodes_list'] = torch.nonzero(curr_dataset.ndata['train_mask']).squeeze().tolist()
            curr_training_states['train_nodes'] = curr_training_states['train_cha_nodes_list']
        elif task_id>0:
            
            curr_training_states['old_nodes_list'] += curr_training_states['train_cha_nodes_list']
            curr_training_states['train_cha_nodes_list'] = torch.nonzero(curr_dataset.ndata['train_mask']).squeeze().tolist()
            curr_training_states['train_nodes'] = curr_training_states['train_cha_nodes_list']

        if self.new_nodes_size > 0 and task_id > 0 and len(curr_training_states['old_nodes_list'])>0:
            train_new_nodes_list = detect(curr_training_states, curr_dataset, curr_dataset.ndata['feat'], task_id, self.detect_strategy, self.new_nodes_size, self.device)
            curr_training_states['train_nodes'] = list(set(curr_training_states['train_nodes'] + train_new_nodes_list)) if len(train_new_nodes_list) > 0 else curr_training_states['train_nodes']
        else:
            detect_time = 0  

        if task_id == 0 or len(curr_training_states['old_nodes_list'])==0:
            if self.memory_size > 0:
                curr_training_states['memory_h'] = MemoryHandler(self.memory_size, self.p, self.memory_strategy, self.alpha, self.device)
                
        elif task_id > 0:
            if curr_model.backbone_model:
                curr_model = EWC(curr_model, self.ewc_lambda, self.ewc_type).to(self.device)
            else:
                curr_model = EWC(curr_model.model, self.ewc_lambda, self.ewc_type).to(self.device)
            
            
            # whether use memory to store important nodes
            if self.memory_size > 0:
                important_nodes_list = curr_training_states['memory_h'].memory
                curr_training_states['train_nodes'] = list(set(curr_training_states['train_nodes'] + important_nodes_list))
            else:
                important_nodes_list = curr_training_states['old_nodes_list']
            # calculate weight importance
            _mask = torch.zeros(curr_dataset.ndata['feat'].shape[0]).to(torch.bool).to(self.device)
            _mask[important_nodes_list] = True
            curr_model.register_ewc_params(curr_dataset.to(self.device), curr_dataset.ndata['feat'].to(self.device), _mask, curr_dataset.ndata['label'].to(self.device), _loss_fn = self.loss_fn)

        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        if curr_model.backbone_model:
            curr_model.observe_labels(curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']])
        else:
            curr_model.model.observe_labels(curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']])
        self._reset_optimizer(curr_optimizer)
        
    def processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        scheduler = curr_training_states['scheduler']
        val_acc = curr_iter_results['val_metric']
        val_loss = curr_iter_results['val_stats']['loss']
        if val_acc > curr_training_states['best_val_acc']:
            curr_training_states['best_val_acc'] = val_acc
        if val_loss < curr_training_states['best_val_loss']:
            curr_training_states['best_val_loss'] = val_loss
            if curr_model.backbone_model:
                curr_training_states['best_weights'] = deepcopy(curr_model.state_dict())
            else:
                curr_training_states['best_weights'] = deepcopy(curr_model.model.state_dict())
        scheduler.step(val_loss)
        
        if -1e-9 < (curr_optimizer.param_groups[0]['lr'] - scheduler.min_lrs[0]) < 1e-9:
            # earlystopping!
            return False
        return True
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        if curr_model.backbone_model:
            curr_model.load_state_dict(curr_training_states['best_weights'])
            curr_training_states['sage'] = deepcopy(curr_model)
        else:
            curr_model.model.load_state_dict(curr_training_states['best_weights'])
            curr_training_states['sage'] = deepcopy(curr_model.model)
        
        curr_training_states['before_g'] = curr_dataset.clone()
        
        if self.memory_size > 0:
            train_output = curr_model(curr_dataset.to(self.device), curr_dataset.ndata['feat'].to(self.device), task_masks=curr_dataset.ndata['task_specific_mask'].to(self.device)).data.cpu().numpy()
            curr_training_states['memory_h'].update(curr_training_states['train_nodes'], curr_dataset, x=train_output[curr_dataset.ndata['train_mask']], y=curr_dataset.ndata['label'].numpy())

class NCClassILCGNNTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        
        self.device = device
        self.detect_strategy = kwargs['detect_strategy']
        self.memory_size = kwargs['memory_size']
        self.memory_strategy = kwargs['memory_strategy']
        self.ewc_lambda = kwargs['ewc_lambda']
        self.ewc_type = kwargs['ewc_type']
        self.new_nodes_size = kwargs['new_nodes_size']
        self.p = kwargs['p']
        self.alpha = kwargs['alpha']
        
        self.save_file_name = f'{self.save_file_name}_{self.memory_size}_{self.ewc_lambda}_{self.new_nodes_size}_{self.p}_{self.alpha}'
        
    def prepareLoader(self, curr_dataset, curr_training_states):
        _temp = deepcopy(curr_dataset)
        _temp.ndata['train_mask'][curr_training_states['train_nodes']] = True
        return [(_temp, _temp.ndata['train_mask'])], [(_temp, _temp.ndata['val_mask'])], [(_temp, _temp.ndata['test_mask'])]
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        curr_batch, mask = _curr_batch

        optimizer.zero_grad()
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))
        loss = self.loss_fn(preds[mask], curr_batch.ndata['label'][mask].to(self.device))
        if not model.backbone_model :
            loss =  loss + model._compute_consolidation_loss()
        loss.backward()
        optimizer.step() 
        
        return {'loss': loss.item(), 'acc': self.eval_fn(preds[mask].argmax(-1), curr_batch.ndata['label'][mask].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        self.num_tasks = scenario.num_tasks
        return {'old_nodes_list':list(), 'train_cha_nodes_list':list(), 'train_nodes':list(), 'sage':None, 'before_g':None}

    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        if task_id == 0:
            curr_training_states['train_cha_nodes_list'] = torch.nonzero(curr_dataset.ndata['train_mask']).squeeze().tolist()
            curr_training_states['train_nodes'] = curr_training_states['train_cha_nodes_list']
        elif task_id>0:
            curr_training_states['old_nodes_list'] += curr_training_states['train_cha_nodes_list']
            curr_training_states['train_cha_nodes_list'] = torch.nonzero(curr_dataset.ndata['train_mask']).squeeze().tolist()
            curr_training_states['train_nodes'] = curr_training_states['train_cha_nodes_list']
            
        if self.new_nodes_size > 0 and task_id > 0 and len(curr_training_states['old_nodes_list'])>0:
            train_new_nodes_list = detect(curr_training_states, curr_dataset, curr_dataset.ndata['feat'], task_id, self.detect_strategy, self.new_nodes_size, self.device)
            curr_training_states['train_nodes'] = list(set(curr_training_states['train_nodes'] + train_new_nodes_list)) if len(train_new_nodes_list) > 0 else curr_training_states['train_nodes']
        else:
            detect_time = 0
            
        if task_id == 0 or len(curr_training_states['old_nodes_list'])==0:
            if self.memory_size > 0:
                curr_training_states['memory_h'] = MemoryHandler(self.memory_size, self.p, self.memory_strategy, self.alpha, self.device)
                
        elif task_id > 0:
            if curr_model.backbone_model:
                curr_model = EWC(curr_model, self.ewc_lambda, self.ewc_type).to(self.device)
            else:
                curr_model = EWC(curr_model.model, self.ewc_lambda, self.ewc_type).to(self.device)
                
            # whether use memory to store important nodes
            if self.memory_size > 0:
                important_nodes_list = curr_training_states['memory_h'].memory
                curr_training_states['train_nodes'] = list(set(curr_training_states['train_nodes'] + important_nodes_list))
            else:
                important_nodes_list = curr_training_states['old_nodes_list']

            # calculate weight importance
            _mask = torch.zeros(curr_dataset.ndata['feat'].shape[0]).to(torch.bool).to(self.device)
            _mask[important_nodes_list] = True
            curr_model.register_ewc_params(curr_dataset.to(self.device), curr_dataset.ndata['feat'].to(self.device), _mask, curr_dataset.ndata['label'].to(self.device), _loss_fn = self.loss_fn)

        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        if curr_model.backbone_model:
            curr_model.observe_labels(curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']])
        else:
            curr_model.model.observe_labels(curr_dataset.ndata['label'][curr_dataset.ndata['train_mask']])
        self._reset_optimizer(curr_optimizer)
        
    def processAfterEachIteration(self, curr_model, curr_optimizer, curr_training_states, curr_iter_results):
        scheduler = curr_training_states['scheduler']
        val_acc = curr_iter_results['val_metric']
        val_loss = curr_iter_results['val_stats']['loss']
        if val_acc > curr_training_states['best_val_acc']:
            curr_training_states['best_val_acc'] = val_acc
        if val_loss < curr_training_states['best_val_loss']:
            curr_training_states['best_val_loss'] = val_loss
            if curr_model.backbone_model:
                curr_training_states['best_weights'] = deepcopy(curr_model.state_dict())
            else:
                curr_training_states['best_weights'] = deepcopy(curr_model.model.state_dict())
        scheduler.step(val_loss)
        
        if -1e-9 < (curr_optimizer.param_groups[0]['lr'] - scheduler.min_lrs[0]) < 1e-9:
            # earlystopping!
            return False
        return True
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        if curr_model.backbone_model:
            curr_model.load_state_dict(curr_training_states['best_weights'])
            curr_training_states['sage'] = deepcopy(curr_model)
        else:
            curr_model.model.load_state_dict(curr_training_states['best_weights'])
            curr_training_states['sage'] = deepcopy(curr_model.model)
        
        curr_training_states['before_g'] = curr_dataset.clone()
        
        if self.memory_size > 0:
            train_output = curr_model(curr_dataset.to(self.device), curr_dataset.ndata['feat'].to(self.device)).data.cpu().numpy()
            
            curr_training_states['memory_h'].update(curr_training_states['train_nodes'], curr_dataset, x=train_output[curr_dataset.ndata['train_mask']], y=curr_dataset.ndata['label'].numpy())
        
class NCDomainILCGNNTrainer(NCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        raise NotImplementedError
        
class NCTimeILCGNNTrainer(NCClassILCGNNTrainer):
    pass
