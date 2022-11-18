import sys
from begin.trainers.nodes import NCTrainer, NCMinibatchTrainer

class NCTaskILBareTrainer(NCTrainer):
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
        curr_batch, mask = _curr_batch
        # use task_masks as additional input
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        return {'preds': preds, 'loss': loss}
    
class NCClassILBareTrainer(NCTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass

class NCClassILBareMinibatchTrainer(NCMinibatchTrainer):
    """
        This trainer has the same behavior as `NCMinibatchTrainer`.
    """
    pass

class NCDomainILBareTrainer(NCTrainer):
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        """
            The event function to handle every training iteration.
            We need to extend the base function since the output format is slightly different from the base trainer.
            
            Args:
                model (torch.nn.Module): the current trained model.
                optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_batch (object): the data (or minibatch) for the current iteration.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A dictionary containing the outcomes (stats) during the training iteration.
        """
        curr_batch, mask = _curr_batch
        optimizer.zero_grad()
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].float().to(self.device))
        loss.backward()
        optimizer.step()
        # output format is slightly different from the base NCTrainer
        return {'loss': loss.item(), 'acc': self.eval_fn(preds, curr_batch.ndata['label'][mask].to(self.device))}
    
    def processEvalIteration(self, model, _curr_batch):
        """
            The event function to handle every evaluation iteration.
            We need to extend the base function since the output format is slightly different from the base trainer.
            
            Args:
                model (torch.nn.Module): the current trained model.
                curr_batch (object): the data (or minibatch) for the current iteration.
                
            Returns:
                A dictionary containing the outcomes (stats) during the evaluation iteration.
        """
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].float().to(self.device))
        # output format is slightly different from the base NCTrainer
        return preds, {'loss': loss.item()}
        
    def processBeforeTraining(self, task_id, curr_dataset, curr_model, curr_optimizer, curr_training_states):
        """
            The event function to execute some processes before training.
            We need to extend the base function since the output format is slightly different from the base trainer.
            
            Args:
                task_id (int): the index of the current task
                curr_dataset (object): The dataset for the current task.
                curr_model (torch.nn.Module): the current trained model.
                curr_optimizer (torch.optim.Optimizer): the current optimizer function.
                curr_training_states (dict): the dictionary containing the current training states.
        """
        curr_training_states['scheduler'] = self.scheduler_fn(curr_optimizer)
        curr_training_states['best_val_acc'] = -1.
        curr_training_states['best_val_loss'] = 1e10
        # it uses all outputs for every task
        curr_model.observe_labels(torch.arange(curr_dataset.ndata['label'].shape[-1]))
        self._reset_optimizer(curr_optimizer)
        
class NCTimeILBareTrainer(NCTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass