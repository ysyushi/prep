import sys
from os import path
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from src.PReP import PReP
from src.load import load_matrix, load_pair2node, load_truth


def calc_metric(truth, score):
    print "ROCAUC:", roc_auc_score(truth, score)
    precision, recall, _ = precision_recall_curve(truth, score)
    print "AUPRC: ", auc(recall, precision)


matrix_file = sys.argv[1]
pair2node_file = sys.argv[2]
model_file = sys.argv[3]
truth_file = sys.argv[4]
num_clus = int(sys.argv[5])
beta = float(sys.argv[6]) if len(sys.argv) > 6 else None

w = load_matrix(matrix_file)
pair2node = load_pair2node(pair2node_file)
assert w.shape[0] == len(pair2node)
truth = load_truth(truth_file)

prep = PReP(w, pair2node, model_file, num_clus=num_clus, beta=beta)
prep.load_model()
np.set_printoptions(precision=4, suppress=True)
score = prep.new_all_pairs_neg_log_likelihood()
calc_metric(truth, score)
