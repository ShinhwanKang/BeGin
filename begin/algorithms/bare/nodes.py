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
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass
        
class NCTimeILBareTrainer(NCTrainer):
    """
        This trainer has the same behavior as `NCTrainer`.
    """
    pass