===================================
Load Custom Scenarios
===================================

Since v0.3, BeGin supports loading custom scenarios with user-defined dataset for users who want to create custom benchmark scenarios.

In this material, we briefly describe how to load custom benchmark scenarios with examples.

-------------------------------------------------
Create custom dataset and its loader
-------------------------------------------------

Since v0.3, ScenarioLoader inputs additional argument `dataset_load_func` for loading custom scenarios. Below, we provide an example for loading custom dataset with NCScenarioLoader. 

.. code-block:: python

  >>> from begin.scenarios.nodes import NCScenarioLoader
  >>> scenario = NCScenarioLoader(dataset_name='custom_dataset_name', dataset_load_func=name_of_custom_function, ...)

Currently, BeGin requires different outputs of the loader function, depending on the target problem:

- Node Classification (NC): The loader function should output a dictionary with keys `graph`, `num_feats`, and `num_classes`.

   + `graph` (`dgl.DGLGraph`) : It should contain node features in `graph.ndata['feat']` and ground-truth labels in `graph.ndata['label']. For Time-IL, time information for constructing tasks in `graph.ndata['time']` is additionally needed. For Domain-IL, domain information for constructing tasks in `graph.ndata['domain']`. The nodes with values in `graph.ndata['time']` or `graph.ndata['domain']` greater than or equal to `num_tasks` will be ignored during the training and evaluation process.
   + `num_feats` (`int`) : Number of node features. `graph.ndata['feat']` should be matched with this value.
   + `num_tasks` (`int`) : Number of tasks for constructing benchmark scenario.
   
- Link Classification (LC): The loader function should output a dictionary with keys `graph`, `num_feats`, and `num_classes`.

   + `graph` (`dgl.DGLGraph`) : It should contain node features in `graph.ndata['feat']` and ground-truth labels in `graph.edata['label']. For Time-IL, time information for constructing tasks in `graph.edata['time']` is additionally needed. For Domain-IL, domain information for constructing tasks in `graph.edata['domain']`. The nodes with values in `graph.edata['time']` or `graph.edata['domain']` greater than or equal to `num_tasks` will be ignored during the training and evaluation process.
   + `num_feats` (`int`) : Number of node features. `graph.ndata['feat']` should be matched with this value.
   + `num_tasks` (`int`) : Number of tasks for constructing benchmark scenario.

- Link Prediction (LP): The loader function should output a dictionary with keys `graph`, `num_feats`, `tvt_splits`, and `neg_edges`.

   + `graph` (`dgl.DGLGraph`) : It should contain node features in `graph.ndata['feat']`. For Time-IL, time information for constructing tasks in `graph.edata['time']` is additionally needed. For Domain-IL, domain information for constructing tasks in `graph.edata['domain']`. The nodes with values in `graph.edata['time']` or `graph.edata['domain']` greater than or equal to `num_tasks` will be ignored during the training and evaluation process. Currently, BeGin only supports undirected graphs for this problem. Therefore, edges in `graph` should satisfy ``graph.edges()[0][0::2] == graph.edges()[1][1::2]`` and ``graph.edges()[0][1::2] == graph.edges()[1][0::2]``.
   
   + `num_feats` (`int`) : Number of node features. `graph.ndata['feat']` should be matched with this value.
   + `tvt_splits` (`torch.LongTensor`) : The information for train/val/test splits. In this tensor, the value `0`, `1`, and `2` indicates the corresponding edge should be used for train, validation, and test, respectively. Its shape should be `(graph.num_edges,)`.
   + `neg_edges` (`dict`) : The dictionary contains negative edges. It should contain keys `val` and `test`. the corresponding value of the key `val` is used as negative edges for validation, and that of `test` is used as negative edges for test. The types of the values and shapes of the values should be `torch.LongTensor`, and `(*, 2)`, respectively.

- Graph Classification (GC): The loader function should output a dictionary with keys `graphs`, `num_feats`, and `num_classes`, `domain_info` (optional, for Domain-IL) and `time_info` (optional, for Time-IL).

   + `graphs` (`Iterable[dgl.DGLGraph, int]`) : It should be the iterable object, which outputs graph object with type `dgl.DGLGraph` and its corresponding label for each iteration. Each graph should contain node features in `graph.ndata['feat']`.
   + `num_feats` (`int`) : Number of node features. `graph.ndata['feat']` should be matched with this value.
   + `num_tasks` (`int`) : Number of tasks for constructing benchmark scenario.
   + `domain_info` (`torch.LongTensor`) : For domain-IL, it should contain domain information for constructing the tasks. Its shape should be `(len(graphs),)`. The graphs with values in `domain_info` greater than or equal to `num_tasks` will be ignored during the training and evaluation process.
   + `time_info` (`torch.LongTensor`) : For time-IL, it should contain time information for constructing the tasks. Its shape should be `(len(graphs),)`. The graphs with values in `time_info` greater than or equal to `num_tasks` will be ignored during the training and evaluation process.

-------------------------------------------------
Example with Cora dataset
-------------------------------------------------

For example, consider we need to load Cora dataset. One of the easiest way to write the function is to extract graphs and from `dgl.data.CoraGraphDataset`.

.. code-block:: python

    def dataset_load_func(save_path):
        dataset = dgl.data.CoraGraphDataset(raw_dir=save_path, verbose=False)

Then, we can extract `graph`, `num_classes`, and `num_tasks` in the `dataset` object.

.. code-block:: python

    def dataset_load_func(save_path):
        dataset = dgl.data.CoraGraphDataset(raw_dir=save_path, verbose=False)
        graph = dataset._g
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes

Now, All you need is just adding one line to return the dictionary contains `graph`, `num_feats`, and `num_classes`!

.. code-block:: python

    def dataset_load_func(save_path):
        dataset = dgl.data.CoraGraphDataset(raw_dir=save_path, verbose=False)
        graph = dataset._g
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        return {'graph': graph, 'num_classes': num_classes, 'num_feats': num_feats}