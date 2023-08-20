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

class GCDomainILBareTrainer(GCClassILBareTrainer):
    """
        This trainer has the same behavior as `GCClassILBareTrainer`.
    """
    pass
        
class GCTimeILBareTrainer(GCClassILBareTrainer):
    """
        This trainer has the same behavior as `GCClassILBareTrainer`.
    """
    pass