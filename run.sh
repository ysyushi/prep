#!/bin/bash

# Argument 1: matrix_file
# Argument 2: pair2node_file
# Argument 3: output_model_file
# Argument 4: truth_file
# Argument 5: num_clus
# Argument 6: beta [optional]

echo "Start training..."
python src/train.py $1 $2 $3 $5 ${6:-}
echo "Start evaluation..."
python eval/eval.py $1 $2 $3 $4 $5 ${6:-}
