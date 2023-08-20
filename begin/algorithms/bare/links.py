import sys
import copy
import dgl
from begin.trainers.links import LCTrainer, LPTrainer

class LCTaskILBareTrainer(LCTrainer):
    def prepareLoader(self, _curr_dataset, curr_training_states):
        """
            The event function to generate dataloaders from the given dataset for the current task.
            For task-IL, we need to additionally consider task information.
            
            Args:
                curr_dataset (object): The dataset for the current task. Its type is dgl.graph for node-level and link-level problem, and dgl.data.DGLDataset for graph-level problem.
                curr_training_states (dict): the dictionary containing the current training states.
                
            Returns:
                A tuple containing three dataloaders.
                The trainer considers the first dataloader, second dataloader, and third dataloader
                as dataloaders for training, validation, and test, respectively.
        """
        curr_dataset = copy.deepcopy(_curr_dataset)
        srcs, dsts = curr_dataset.edges()
        labels = curr_dataset.edata.pop('label')
        train_mask = curr_dataset.edata.pop('train_mask')
        val_mask = curr_dataset.edata.pop('val_mask')
        test_mask = curr_dataset.edata.pop('test_mask')
        task_mask = curr_dataset.edata.pop('task_specific_mask')
        curr_dataset = dgl.add_self_loop(curr_dataset)
        return [(curr_dataset, srcs[train_mask], dsts[train_mask], task_mask[train_mask], labels[train_mask])], [(curr_dataset, srcs[val_mask], dsts[val_mask], task_mask[val_mask], labels[val_mask])], [(curr_dataset, srcs[test_mask], dsts[test_mask], task_mask[test_mask], labels[test_mask])]
    
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
        # use task_masks as additional input
        curr_batch, srcs, dsts, task_masks, labels = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), srcs, dsts, task_masks=task_masks)
        loss = self.loss_fn(preds, labels.to(self.device))
        return {'preds': preds, 'loss': loss}
    
class LCClassILBareTrainer(LCTrainer):
    pass

class LCTimeILBareTrainer(LCTrainer):
    """
        This trainer has the same behavior as `LCTrainer`.
    """

class LPTimeILBareTrainer(LPTrainer):
    """
        This trainer has the same behavior as `LPTrainer`.
    """
    pass

class LPDomainILBareTrainer(LPTrainer):
    """
        This trainer has the same behavior as `LPTrainer`.
    """
    pass