import sys
import random
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from begin.trainers.graphs import GCTrainer

class GCTaskILTWPTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lambda_l = kwargs['lambda_l'] if 'lambda_l' in kwargs else 10000
        self.lambda_t = kwargs['lambda_t']  if 'lambda_t' in kwargs else 10000
        self.beta = kwargs['beta'] if 'beta' in kwargs else 0.1
        
    def inference(self, model, _curr_batch, training_states, return_elist=False):
        graphs, labels, masks = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                      return_elist = return_elist,
                      task_masks = masks)
        if return_elist:
            preds, elist = preds
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss, 'elist': elist if return_elist else None}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states): 
        cls_loss = results['loss']
        cls_loss.backward(retain_graph=True)
        cur_importance_score = 0.
        for gs in model.parameters():
            cur_importance_score += torch.norm(gs.grad.data.clone(), p=1) 
        old_importance_score = 0.
        for tt in range(0, training_states['current_task']):
            for i, p in enumerate(model.parameters()):
                I_n = self.lambda_l * training_states['cls_important_score'][tt][i] + self.lambda_t * training_states['topology_important_score'][tt][i]
                I_n = I_n * (p - training_states['optpar'][tt][i]).pow(2)
                old_importance_score += I_n.sum()            
        cls_loss += old_importance_score + self.beta * cur_importance_score
        cls_loss.backward()
        optimizer.step()
        return {'_num_items': results['preds'].shape[0],
                'loss': results['loss'].item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'current_task':0, 'fisher_loss':{}, 'fisher_att':{}, 'optpar':{}, 'mem_mask':None, 'cls_important_score':{}, 'topology_important_score':{}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):        
        curr_model.load_state_dict(curr_training_states['best_weights'])
        
        optpars = [None for (name, p) in curr_model.named_parameters()]
        cls_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        topology_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            results = self.inference(curr_model, _curr_batch, curr_training_states, return_elist=True)
            results['loss'].backward(retain_graph=True)    
            curr_sz = results['preds'].shape[0]
            total_num_items += curr_sz
            
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                optpars[idx] = p.data.clone().detach()
                cls_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
            eloss = torch.norm(results['elist'][0]) 
            eloss.backward()
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                topology_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
        
        for idx, (name, p) in enumerate(curr_model.named_parameters()):
            cls_scores[idx] /= total_num_items
            topology_scores[idx] /= total_num_items
        _idx = curr_training_states['current_task']
        curr_training_states['cls_important_score'][_idx] = cls_scores  
        curr_training_states['topology_important_score'][_idx] = topology_scores
        curr_training_states['optpar'][_idx] = optpars
        curr_training_states['current_task'] += 1
        
class GCClassILTWPTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lambda_l = kwargs['lambda_l'] if 'lambda_l' in kwargs else 10000
        self.lambda_t = kwargs['lambda_t']  if 'lambda_t' in kwargs else 10000
        self.beta = kwargs['beta'] if 'beta' in kwargs else 0.1
        
    def inference(self, model, _curr_batch, training_states, return_elist=False):
        graphs, labels = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                      return_elist = return_elist)
        if return_elist:
            preds, elist = preds
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss, 'elist': elist if return_elist else None}
    
    def afterInference(self, results, model, optimizer, _curr_batch, training_states): 
        cls_loss = results['loss']
        cls_loss.backward(retain_graph=True)
        cur_importance_score = 0.
        for gs in model.parameters():
            cur_importance_score += torch.norm(gs.grad.data.clone(), p=1) 
        old_importance_score = 0.
        for tt in range(0, training_states['current_task']):
            for i, p in enumerate(model.parameters()):
                I_n = self.lambda_l * training_states['cls_important_score'][tt][i] + self.lambda_t * training_states['topology_important_score'][tt][i]
                I_n = I_n * (p - training_states['optpar'][tt][i]).pow(2)
                old_importance_score += I_n.sum()            
        cls_loss += old_importance_score + self.beta * cur_importance_score
        cls_loss.backward()
        optimizer.step()
        return {'_num_items': results['preds'].shape[0],
                'loss': results['loss'].item(),
                'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[1].to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'current_task':0, 'fisher_loss':{}, 'fisher_att':{}, 'optpar':{}, 'mem_mask':None, 'cls_important_score':{}, 'topology_important_score':{}}
    
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):        
        curr_model.load_state_dict(curr_training_states['best_weights'])
        
        optpars = [None for (name, p) in curr_model.named_parameters()]
        cls_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        topology_scores = [torch.zeros_like(p.data) for (name, p) in curr_model.named_parameters()]
        
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            results = self.inference(curr_model, _curr_batch, curr_training_states, return_elist=True)
            results['loss'].backward(retain_graph=True)    
            curr_sz = results['preds'].shape[0]
            total_num_items += curr_sz
            
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                optpars[idx] = p.data.clone().detach()
                cls_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
            eloss = torch.norm(results['elist'][0]) 
            eloss.backward()
            for idx, (name, p) in enumerate(curr_model.named_parameters()):
                topology_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
        
        for idx, (name, p) in enumerate(curr_model.named_parameters()):
            cls_scores[idx] /= total_num_items
            topology_scores[idx] /= total_num_items
        _idx = curr_training_states['current_task']
        curr_training_states['cls_important_score'][_idx] = cls_scores  
        curr_training_states['topology_important_score'][_idx] = topology_scores
        curr_training_states['optpar'][_idx] = optpars
        curr_training_states['current_task'] += 1

class GCDomainILTWPTrainer(GCTrainer):
    def __init__(self, model, scenario, optimizer_fn, loss_fn, device, **kwargs):
        super().__init__(model.to(device), scenario, optimizer_fn, loss_fn, device, **kwargs)
        self.lambda_l = kwargs['lambda_l'] if 'lambda_l' in kwargs else 10000
        self.lambda_t = kwargs['lambda_t']  if 'lambda_t' in kwargs else 10000
        self.beta = kwargs['beta'] if 'beta' in kwargs else 0.1
        
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        graphs, labels = _curr_batch
        optimizer.zero_grad()
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                      return_elist = False)
        cls_loss = self.loss_fn(preds, labels.to(self.device))    
        
        cur_importance_score = 0.
        cls_loss.backward(retain_graph=True)
        for gs in model.parameters():
            cur_importance_score += torch.norm(gs.grad.data.clone(), p=1) 
        old_importance_score = 0.
        for tt in range(0, training_states['current_task']):
            for i, p in enumerate(model.parameters()):
                I_n = self.lambda_l * training_states['cls_important_score'][tt][i] + self.lambda_t * training_states['topology_important_score'][tt][i]
                I_n = I_n * (p - training_states['optpar'][tt][i]).pow(2)
                old_importance_score += I_n.sum()            
        cls_loss += old_importance_score + self.beta * cur_importance_score
        cls_loss.backward()
        optimizer.step()
        del cur_importance_score, old_importance_score
        
        return {'_num_items': preds.shape[0], 'loss': cls_loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
    
    def processEvalIteration(self, model, _curr_batch):
        graphs, labels = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        return preds, {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
    
    def initTrainingStates(self, scenario, model, optimizer):
        return {'current_task':0, 'fisher_loss':{}, 'fisher_att':{}, 'optpar':{}, 'mem_mask':None, 'cls_important_score':{}, 'topology_important_score':{}}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)
        
    def processAfterTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):        
        curr_model.load_state_dict(curr_training_states['best_weights'])
        
        optpars, cls_scores, topology_scores = [], [], []
        train_loader = self.prepareLoader(curr_dataset, curr_training_states)[0]
        total_num_items = 0
        for i, _curr_batch in enumerate(iter(train_loader)):
            curr_model.zero_grad()
            graphs, labels = _curr_batch
            preds, elist = curr_model(graphs.to(self.device),
                                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                                      return_elist = True)
            cls_loss = self.loss_fn(preds, labels.to(self.device))
            cls_loss.backward(retain_graph=True)
            
            curr_sz = labels.shape[0]
            total_num_items += curr_sz
            # print(curr_sz)
            
            if i == 0:
                for idx, (name, p) in enumerate(curr_model.named_parameters()):
                    optpars.append(p.data.clone().detach())
                    cls_scores.append(p.grad.data.clone().pow(2).detach() * curr_sz)
            else:
                for idx, (name, p) in enumerate(curr_model.named_parameters()):
                    cls_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz
            
            eloss = torch.norm(elist[0]) 
            eloss.backward()
            
            if i == 0:
                for idx, (name, p) in enumerate(curr_model.named_parameters()):
                    topology_scores.append(p.grad.data.clone().pow(2).detach() * curr_sz)
            else:
                for idx, (name, p) in enumerate(curr_model.named_parameters()):
                    topology_scores[idx] += p.grad.data.clone().pow(2).detach() * curr_sz

            len_elist = len(elist)
            for _ in range(len_elist):
                del elist[0]
            del cls_loss, eloss, elist
        
        print('total:', total_num_items)
        for idx, (name, p) in enumerate(curr_model.named_parameters()):
            cls_scores[idx] /= total_num_items
            topology_scores[idx] /= total_num_items
            
        _idx = curr_training_states['current_task']
        curr_training_states['cls_important_score'][_idx] = cls_scores  
        curr_training_states['topology_important_score'][_idx] = topology_scores
        curr_training_states['optpar'][_idx] = optpars
        curr_training_states['current_task'] += 1  
        
class GCTimeILTWPTrainer(GCClassILTWPTrainer):
    pass