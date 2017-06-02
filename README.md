# PReP

This repository provides a reference implementation of *PReP* as described in the paper:<br>
> PReP: Path-Based Relevance from a Probabilistic Perspective in Heterogeneous Information Networks<br>
> Yu Shi, Po-Wei Chan, Honglei Zhuang, Huan Gui, and Jiawei Han.<br>
> In Proceedings of the 23nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ACM, 2017.<br>

### Basic Usage

#### Input

The supported input HIN should be provided with concerned meta-paths. Note that our implementation further assumes (1) each concerned meta-path has at least one path instance in the given HIN, and (2) the given HIN has not dangling nodes, _i.e.,_ nodes not attaching to any path instances under any given meta-paths.

There are three required input files.

1. **Matrix file** contains the matrix for path counts between node pairs under each meta-paht, which is kept in a dok-based matrix format. The first line specifies the shape of the matrix:

		num_node_pair num_path_type

	The following lines are all the nonzero entries in the matrix, in the format of:

		node_pair_id meta_path_id path_count

2. **Node2pair file** contains the indices of nodes that each node pair is made of, in the order of node pair indices. That is, the i-th line gives indices of nodes that form the node pair with id == i:

		node1_id node2_id

3. **Truth file** contains the ground truth of each node pair, in the order of node pair indices. Each line should contain a single digit of 1 or 0. (For evaluation only.)

Hyperparameters to be specified as part of input include the number of clusters and beta for the *PReP* model (optional).

#### Execute

_All the commands are executed from the project home directory._<br/>

To train *PReP* model:<br/>
``python src/train.py matrix_file pair2node_file output_model_file num_clus [optional: beta] ``

To evaluate the ouput *PReP* model:<br/>
``python eval/eval.py matrix_file pair2node_file model_file truth_file num_clus [optional: beta] ``

Alternatively, to run a shell script that first trains the model and then evaluates it:<br/>
``./run.sh matrix_file pair2node_file model_file truth_file num_clus [optional: beta] ``

#### Example command

An example dataset can be found under the ``data`` folder, which is a subset of the Facebook dataset we used for evaluation reported in the paper.

To run the previous three commands on example dataset, execute respectively:<br/>
	``python src/train.py data/matrix.txt data/pair2node.txt data/example.model 15 1.e-4``<br/>
	``python eval/eval.py data/matrix.txt data/pair2node.txt data/example.model data/truth.txt 15 1.e-4``<br/>
	``./run.sh data/matrix.txt data/pair2node.txt data/example.model data/truth.txt 15 1.e-4``

### Citing
If you find *PReP* useful for your research, please consider citing the following paper:

	@inproceedings{shi2017prep,
	author = {Shi, Yu and Chan, Po-Wei and Zhuang, Honglei and Gui, Huan and Han, Jiawei},
	 title = {PReP: Path-Based Relevance from a Probabilistic Perspective in Heterogeneous Information Networks},
	 booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
	 year = {2017},
	 organization={ACM}
	}


### Miscellaneous

Please send any questions you might have about the codes and/or the algorithm to <yushi2@illinois.edu> or <poweichan0902@gmail.com>.

*Note:* This is only a reference implementation of the *PReP* algorithm and could benefit from several performance enhancement schemes, some of which are discussed in the paper.
