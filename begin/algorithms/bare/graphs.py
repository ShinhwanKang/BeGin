import sys
from begin.trainers.graphs import GCTrainer

class GCTaskILBareTrainer(GCTrainer):
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
        # use task_masks as additional input
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None,
                      task_masks = masks)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}

class GCClassILBareTrainer(GCTrainer):
    pass

class GCDomainILBareTrainer(GCTrainer):
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
        graphs, labels = _curr_batch
        optimizer.zero_grad()
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        loss.backward()
        optimizer.step()
        # output format is slightly different from the base GCTrainer
        return {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
        
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
        graphs, labels = _curr_batch
        preds = model(graphs.to(self.device),
                      graphs.ndata['feat'].to(self.device) if 'feat' in graphs.ndata else None,
                      edge_attr = graphs.edata['feat'].to(self.device) if 'feat' in graphs.edata else None,
                      edge_weight = graphs.edata['weight'].to(self.device) if 'weight' in graphs.edata else None)
        loss = self.loss_fn(preds, labels.to(self.device))
        # output format is slightly different from the base GCTrainer
        return preds, {'_num_items': preds.shape[0], 'loss': loss.item(), 'acc': self.eval_fn(preds, labels.to(self.device))}
        
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
        # it is logistic regression (binary classification) problem
        curr_model.observe_labels(torch.LongTensor([0]))
        self._reset_optimizer(curr_optimizer)
        
class GCTimeILBareTrainer(GCClassILBareTrainer):
    pass