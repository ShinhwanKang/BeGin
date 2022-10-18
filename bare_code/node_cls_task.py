from .trainer_code.node_level import BaseNodeIncrementalBenchmark
class BaseNodeTaskIncrementalBenchmark(BaseNodeIncrementalBenchmark):
    def processTrainIteration(self, model, optimizer, _curr_batch, training_states):
        curr_batch, mask = _curr_batch
        optimizer.zero_grad()
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'acc': self.eval_fn(preds.argmax(-1), curr_batch.ndata['label'][mask].to(self.device))}
        
    def processEvalIteration(self, model, _curr_batch):
        curr_batch, mask = _curr_batch
        preds = model(curr_batch.to(self.device), curr_batch.ndata['feat'].to(self.device), task_masks=curr_batch.ndata['task_specific_mask'].to(self.device))[mask]
        loss = self.loss_fn(preds, curr_batch.ndata['label'][mask].to(self.device))
        return torch.argmax(preds, dim=-1), {'loss': loss.item()}