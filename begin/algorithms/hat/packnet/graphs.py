import sys
import numpy as np
import torch
import dgl
import torch.nn.functional as F
from begin.trainers.graphs import GCTrainer

class GCTaskILPackNetTrainer(GCTrainer):
    def initTrainingStates(self, scenario, model, optimizer):
        model_masks = {}
        with torch.no_grad():
            pr = 1. - np.exp((1. / (self.num_tasks - 1)) * np.log(1. / self.num_tasks))
            for name, p in model.named_parameters():
                if 'convs' in name or 'mlp_layers' in name or 'enc' in name:
                    model_masks[name] = (torch.ones_like(p.data) * (self.num_tasks + 1)).long()
        return {'pr': pr, 'packnet_masks': model_masks, 'class_to_task': -torch.ones(model.classifier.num_outputs, dtype=torch.long)}
    
    def inference(self, model, _curr_batch, training_states):
        """
            The event function to execute inference step.
        
            For task-IL, we need to additionally consider task information for the inference step.
        
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the inference results, such as prediction result and loss.
        """
        graphs, labels, masks = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                      task_masks = masks)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}
    
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            
            PackNet masks the parameters of the model depending on the current magnitude of parameters in this function.
                   
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        
        # fix batchnorm parameters
        def set_bn_eval(m):
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
        if self.curr_task == 0:
            # initialize masks
            curr_model.packnet_masks = curr_training_states.pop('packnet_masks')
            curr_model.class_to_task = curr_training_states.pop('class_to_task')
        else:
            # fix batchnorm parameters
            for name, p in curr_model.named_parameters():
                if 'norms' in name:
                    p.requires_grad_(False)
                    
        # detect new classes    
        trainloader, valloader, _ = self.prepareLoader(curr_dataset, curr_training_states)
        first_train_batch = next(iter(trainloader))
        curr_model.class_to_task[first_train_batch[-1][0]] = self.curr_task
        
        # pre-training (10% of number of training epochs)
        pre_scheduler = self.scheduler_fn(curr_optimizer)
        best_val_loss = 1e10
        pre_checkpoint = copy.deepcopy(curr_model.state_dict())
        for epoch_cnt in range(self.args.num_steps // 10):
            curr_model.train()
            if self.curr_task > 0:
                curr_model.apply(set_bn_eval)
                
            for curr_batch in trainloader:
                self.processTrainIteration(curr_model, curr_optimizer, curr_batch, curr_training_states, use_mask=False)
            
            curr_model.eval()
            val_stats = []
            for curr_batch in valloader:
                val_stats.append(self.processEvalIteration(curr_model, curr_batch, use_mask=False)[-1])
            val_loss = sum([vst['loss'] for vst in val_stats]) / sum([vst['n_samples'] for vst in val_stats])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                pre_checkpoint = copy.deepcopy(curr_model.state_dict())
            pre_scheduler.step(val_loss)
            if -1e-9 < (curr_optimizer.param_groups[0]['lr'] - pre_scheduler.min_lrs[0]) < 1e-9:
                break
        
        # select parameters and perform masking
        with torch.no_grad():
            curr_model.load_state_dict(pre_checkpoint)
            for name, p in curr_model.named_parameters():
                if 'convs' in name or 'mlp_layers' in name or 'enc' in name:
                    try:
                        candidates = torch.abs(p.data)[curr_model.packnet_masks[name] >= self.curr_task]
                        threshold = torch.topk(candidates, int((candidates.shape[0] * curr_training_states['pr']) + 0.5), largest=True).values.min()
                        accept_criteria = (curr_model.packnet_masks[name] >= self.curr_task)
                        if self.curr_task < self.num_tasks - 1:
                            accept_criteria &= (torch.abs(p.data) >= threshold)
                        curr_model.packnet_masks[name][accept_criteria] = self.curr_task
                        p.data[curr_model.packnet_masks[name] > self.curr_task] = 0.
                    except:
                        pass
                    
        for name, p in curr_model.named_parameters():
            if 'norms' in name:
                p.requires_grad_(False)
                
        super().processBeforeTraining(task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states)
        
    def afterInference(self, results, model, optimizer, _curr_batch, training_states, use_mask=True):
        """
            The event function to execute some processes right after the inference step (for training).
            We recommend performing backpropagation in this event function.
            
            In this function, PackNet weights gradients of parametrs according to 'packnet_masks'.
            For this, Packnet additionally needs 'use_mask' parameter.
            
            Args:
                results (dict): the returned dictionary from the event function `inference`.
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                use_mask (bool): whether model masks weights of the model.
                
            Returns:
                A dictionary containing the information from the `results`.
        """
        results['loss'].backward()
        if use_mask:
            # flow gradients only for masked ones
            for name, p in model.named_parameters():
                if 'convs' in name or 'mlp_layers' in name or 'enc' in name:
                    p.grad = p.grad * (model.packnet_masks[name] == self.curr_task).long()
        else:
            # flow gradients only for unmasked ones (pre-training)
            for name, p in model.named_parameters():
                if 'convs' in name or 'mlp_layers' in name or 'enc' in name:
                    p.grad = p.grad * (model.packnet_masks[name] >= self.curr_task).long()
        optimizer.step()
        return {'_num_items': results['preds'].shape[0], 'loss': results['loss'].item(), 'acc': self.eval_fn(results['preds'].argmax(-1), _curr_batch[1].to(self.device))}
    
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states, use_mask=True):
        """
            The event function to handle every training iteration.
        
            PackNet additionally needs 'use_mask' parameter.
        
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                use_mask (bool): whether model masks weights of the model.
                
            Returns:
                A dictionary containing the outcomes (stats) during the training iteration.
        """
        
        # freeze running variables of batchnorms
        if use_mask:
            def set_bn_eval(m):
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
            model.apply(set_bn_eval)
        
        optimizer.zero_grad()
        self.beforeInference(model, optimizer, _curr_batch, training_states)
        inference_results = self.inference(model, _curr_batch, training_states)
        return self.afterInference(inference_results, model, optimizer, _curr_batch, training_states, use_mask=use_mask)
    
    def processEvalIteration(self, model, _curr_batch, use_mask=True):
        """
            The event function to handle every evaluation iteration.
            
            We need to extend the base function since the output format is slightly different from the base trainer.
            
            PackNet additionally needs 'use_mask' parameter.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                use_mask (bool): whether model masks weights of the model.
                
            Returns:
                A dictionary containing the outcomes (stats) during the evaluation iteration.
        """
        
        # classify each sample by its task-id
        eval_model = copy.deepcopy(model)
        graphs, labels, mask = _curr_batch
        each_graph = dgl.unbatch(graphs)
        task_ids = torch.max(mask * model.class_to_task, dim=-1).values
        task_checks = torch.min(mask * model.class_to_task, dim=-1).values
        task_ids[task_checks < 0] = self.curr_task
        
        num_samples = torch.bincount(task_ids.detach().cpu(), minlength=self.curr_task+1)
        total_results = torch.zeros(mask.shape[0], dtype=torch.long).to(self.device)
        total_loss = 0.
        
        # handle the samples with different tasks separately
        for i in range(self.curr_task, -1, -1):
            if num_samples[i].item() == 0: continue
            if use_mask:
                for name, p in eval_model.named_parameters():
                    if 'convs' in name or 'mlp_layers' in name or 'enc' in name:
                        p.data = p.data * (model.packnet_masks[name] <= i).long()

            eval_mask = (task_ids == i)
            eval_nodes = torch.nonzero(eval_mask, as_tuple=True)[0]
            target_graphs = dgl.batch([each_graph[graph_id] for graph_id in eval_nodes.tolist()])
            results = self.inference(eval_model, (target_graphs, labels[eval_mask], mask[eval_mask]), None)
            total_results[eval_nodes] = torch.argmax(results['preds'], dim=-1)
            total_loss += results['loss'].item() * num_samples[i].item()
        n_samples = torch.sum(num_samples).item()
        total_loss /= n_samples
        
        return total_results, {'loss': total_loss, 'n_samples': n_samples}