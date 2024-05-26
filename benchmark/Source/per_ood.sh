# BZR  69+(19) 17  64-19
# PROTEINS 360+(40) 90 (133-40)
# COX2 81+(22) 21 （73-22）

#CVTGOD
python benchmark/per_ood.py -exp_type ad -DS PROTEINS   -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.0 -num_layer 3 -eval_freq 5   -per 0.1 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS BZR   -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.0 -num_layer 3 -eval_freq 5  -per 0.2 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS COX2    -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.0 -num_layer 3  -per 0.3 -eval_freq 5  -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool



#GLADC
python benchmark/per_ood.py -exp_type ad -DS PROTEINS   -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256  -num_layer 2 -dropout 0.1  -lr 0.0001  -per 0.1 -model GLADC GLADC

python benchmark/per_ood.py -exp_type ad -DS BZR    -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256  -num_layer 2 -dropout 0.1  -lr 0.0001   -per 0.2 -model GLADC GLADC

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256  -num_layer 2 -dropout 0.1  -lr 0.0001   -per 0.3 -model GLADC GLADC


#GLocalKD
python benchmark/per_ood.py -exp_type ad -DS PROTEINS    -num_epoch 50  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -num_layer 3 -eval_freq 2   -per 0.1 -model GLocalKD GLocalKD -max-nodes 0

python benchmark/per_ood.py -exp_type ad -DS BZR     -num_epoch 50  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -num_layer 3 -eval_freq 2   -per 0.2 -model GLocalKD GLocalKD -max-nodes 0

python benchmark/per_ood.py -exp_type ad -DS COX2   -exp_type add    -num_epoch 50  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -num_layer 3 -eval_freq 2   -per 0.3 -model GLocalKD GLocalKD -max-nodes 0


#GOOD-D
python benchmark/per_ood.py -exp_type ad -DS PROTEINS    -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.2   -per 0.1 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS BZR    -num_epoch 400 -num_cluster 2 -alpha 0   -per 0.2 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 150 -num_cluster 15 -alpha 0.2   -per 0.3 -model GOOD-D


#GraphDE

python benchmark/per_ood.py -exp_type ad -DS PROTEINS   -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.2   -per 0.1 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS BZR    -batch_size 64 -num_epoch 400 -alpha 0   -per 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS COX2   -batch_size 64 -num_epoch 150  -alpha 0.2   -per 0.3 -model GraphDE -num_layer 2 -hidden_dim 64


# OCGIN
python benchmark/per_ood.py -exp_type ad -DS PROTEINS       -batch_size  128 -batch_size_test 9999 -eval_freq 5 -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.1 -model OCGIN OCGIN

python benchmark/per_ood.py -exp_type ad -DS BZR       -batch_size  128 -batch_size_test 9999 -eval_freq 5 -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.2 -model OCGIN OCGIN

python benchmark/per_ood.py -exp_type ad -DS COX2      -batch_size  128 -batch_size_test 9999 -eval_freq 5 -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.3 -model OCGIN OCGIN


# OCGTL
python benchmark/per_ood.py -exp_type ad -DS PROTEINS      -batch_size  128 -batch_size_test 9999 -eval_freq 5 -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.1 -model OCGTL OCGTL

python benchmark/per_ood.py -exp_type ad -DS BZR        -batch_size  128 -batch_size_test 9999 -eval_freq 5 -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.2 -model OCGTL OCGTL

python benchmark/per_ood.py -exp_type ad -DS COX2     -batch_size  128 -batch_size_test 9999 -eval_freq 5 -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.3 -model OCGTL OCGTL


# SIGNET

python benchmark/per_ood.py -exp_type ad -DS PROTEINS       -batch_size  128 -batch_size_test 9999 -eval_freq 5 -num_epoch 300 -lr 0.001 -hidden_dim 64    -per 0.1 -model SIGNET SIGNET

python benchmark/per_ood.py -exp_type ad -DS BZR      -batch_size  128 -batch_size_test 9999 -eval_freq 5 -num_epoch 300 -lr 0.001 -hidden_dim 64   -per 0.2 -model SIGNET SIGNET

python benchmark/per_ood.py -exp_type ad -DS COX2       -batch_size  128 -batch_size_test 9999 -eval_freq 5 -num_epoch  300 -lr 0.001 -hidden_dim 64    -per 0.3 -model SIGNET SIGNET



