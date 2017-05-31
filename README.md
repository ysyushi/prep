# PReP

This repository provides a reference implementation of *PReP* as described in the paper:<br>
> PReP: Path-Based Relevance from a Probabilistic Perspective in Heterogeneous Information Networks<br>
> Yu Shi, Po-Wei Chan, Honglei Zhuang, Huan Gui, and Jiawei Han.<br>
> Knowledge Discovery and Data Mining, 2016.<br>

### Basic Usage

#### Input

The supported input should be an HIN with multiple nodes and path-types. The HIN should be a connected graph, _i.e.,_ each node has at least 1 edge, and all path-types should not be redundant, _i.e.,_ with no such edge existing. Among all the node pairs, we are only interested in those that has least 1 edge in between. All the nodes, node pairs, and path-types in care should be properly numbered starting from 0, respectively.

There are 3 required input files.

1. **Matrix file**: contains the weight of edges of all path-types between all node pairs. To avoid wasting memory, we take the input in a dok-based matrix format. The first line should contain the shape of the matrix:

		num_node_pair num_path_type

	The following lines are all the nonzero entries in the matrix, in the format of:

		node_pair_id path_type_id edge_weight

2. **Node2pair file**: contains the indices of nodes that each node pair is made of, in the order of node pair indices. Each line should be in the format of:

		node1_id node2_id

3. **Truth file**: contains the ground truth of each node pair, in the order of node pair indices. Each line should contain a single digit of 1 or 0. (For evaluation only)

Inputs also include destination for output model file, number of clusters, and beta for *PReP* model (optional).

#### Execute

_All the commands are executed from the project home directory._<br/>

To train *PReP* model:<br/>
``python src/train.py matrix_file pair2node_file output_model_file num_clus [optional: beta] ``

To evaluate the ouput *PReP* model:<br/>
``python eval/eval.py matrix_file pair2node_file model_file truth_file num_clus [optional: beta] ``

Alternatively, to run a shell script that first trains the model and then evaluates it:<br/>
``./run.sh matrix_file pair2node_file model_file truth_file num_clus [optional: beta] ``

#### Example command

There is an example dataset under ``data`` folder. 

To run the previous three commands on example dataset, execute respectively:<br/>
	``python src/train.py data/matrix.txt data/pair2node.txt data/example.model 15 1.e-4``<br/>
	``python eval/eval.py data/matrix.txt data/pair2node.txt data/example.model data/truth.txt 15 1.e-4``<br/>
	``./run.sh data/matrix.txt data/pair2node.txt data/example.model data/truth.txt 15 1.e-4``

### Citing
If you find *PReP* useful for your research, please consider citing the following paper:

	@inproceedings{PReP-kdd2017,
	author = {Shi, Yu and Chan, Po-Wei and Zhuang, Honglei and Gui, Huan and Han, Jiawei},
	 title = {PReP: Path-Based Relevance from a Probabilistic Perspective in Heterogeneous Information Networks},
	 booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
	 year = {2017}
	}


### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <yushi2@illinois.edu>.

*Note:* This is only a reference implementation of the *PReP* algorithm and could benefit from several performance enhancement schemes, some of which are discussed in the paper.