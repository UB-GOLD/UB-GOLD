# BZR  69+(19) 17  64-19
# PROTEINS_full 360+(40) 90 (133-40)
# COX2 81+(22) 21 （73-22）

# CVTGOD
python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full   -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 600 -num_cluster 2 -alpha 0.2 -num_layer 5  -lr 0.001   -per 0.1 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS BZR   -rw_dim 8 -dg_dim 8 -hidden_dim 32 -num_epoch 1000 -num_cluster 2 -alpha 0.45 -num_layer 5  -lr 0.001  -per 0.1 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS COX2    -rw_dim 16 -dg_dim 16 -hidden_dim 32 -num_epoch 1000 -num_cluster 3 -alpha 0.6 -num_layer 5  -lr 0.001  -per 0.1  -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full   -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 600 -num_cluster 2 -alpha 0.2 -num_layer 5  -lr 0.001   -per 0.2 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS BZR   -rw_dim 8 -dg_dim 8 -hidden_dim 32 -num_epoch 1000 -num_cluster 2 -alpha 0.45 -num_layer 5  -lr 0.001  -per 0.2 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS COX2    -rw_dim 16 -dg_dim 16 -hidden_dim 32 -num_epoch 1000 -num_cluster 3 -alpha 0.6 -num_layer 5  -lr 0.001  -per 0.2  -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full   -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 600 -num_cluster 2 -alpha 0.2 -num_layer 5  -lr 0.001   -per 0.3 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS BZR   -rw_dim 8 -dg_dim 8 -hidden_dim 32 -num_epoch 1000 -num_cluster 2 -alpha 0.45 -num_layer 5  -lr 0.001  -per 0.3 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/per_ood.py -exp_type ad -DS COX2    -rw_dim 16 -dg_dim 16 -hidden_dim 32 -num_epoch 1000 -num_cluster 3 -alpha 0.6 -num_layer 5  -lr 0.001  -per 0.3  -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool


#GLADC
python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full   -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 128   -dropout 0.2  -lr 0.0001  -per 0.1 -model GLADC GLADC -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS BZR      -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.3  -lr 0.001   -per 0.1 -model GLADC GLADC -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.2  -lr 0.0001   -per 0.1 -model GLADC GLADC -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full   -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 128   -dropout 0.2  -lr 0.0001  -per 0.2 -model GLADC GLADC -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS BZR      -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.3  -lr 0.001   -per 0.2 -model GLADC GLADC -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.2  -lr 0.0001   -per 0.2 -model GLADC GLADC -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full   -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 128   -dropout 0.2  -lr 0.0001  -per 0.3 -model GLADC GLADC -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS BZR      -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.3  -lr 0.001   -per 0.3 -model GLADC GLADC -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.2  -lr 0.0001   -per 0.3 -model GLADC GLADC -output_dim 128

#GLocalKD
python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full    -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.001 -num_layer 4   -per 0.1 -model GLocalKD GLocalKD -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS BZR    -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.0001 -num_layer 4   -per 0.1 -model GLocalKD GLocalKD -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.0001 -num_layer 3    -per 0.1 -model GLocalKD GLocalKD -output_dim 256

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full    -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.001 -num_layer 4   -per 0.2 -model GLocalKD GLocalKD -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS BZR    -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.0001 -num_layer 4   -per 0.2 -model GLocalKD GLocalKD -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.0001 -num_layer 3    -per 0.2 -model GLocalKD GLocalKD -output_dim 256

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full    -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.001 -num_layer 4   -per 0.3 -model GLocalKD GLocalKD -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS BZR    -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.0001 -num_layer 4   -per 0.3 -model GLocalKD GLocalKD -output_dim 128

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.0001 -num_layer 3    -per 0.3 -model GLocalKD GLocalKD -output_dim 256


#GOOD-D
python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full    -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.2   -per 0.1 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS BZR    -num_epoch 400 -num_cluster 2 -alpha 0   -per 0.1 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 150 -num_cluster 15 -alpha 0.2   -per 0.1 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full    -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.2   -per 0.2 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS BZR    -num_epoch 400 -num_cluster 2 -alpha 0   -per 0.2 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 150 -num_cluster 15 -alpha 0.2   -per 0.2 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full    -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.3   -per 0.3 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS BZR    -num_epoch 400 -num_cluster 2 -alpha 0   -per 0.3 -model GOOD-D

python benchmark/per_ood.py -exp_type ad -DS COX2   -num_epoch 150 -num_cluster 15 -alpha 0.2   -per 0.3 -model GOOD-D

#GraphDE

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full   -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.2   -per 0.1 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS BZR    -batch_size 64 -num_epoch 400 -alpha 0   -per 0.1 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS COX2   -batch_size 64 -num_epoch 150  -alpha 0.2   -per 0.1 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full   -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.2   -per 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS BZR    -batch_size 64 -num_epoch 400 -alpha 0   -per 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS COX2   -batch_size 64 -num_epoch 150  -alpha 0.2   -per 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full   -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.2   -per 0.3-model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS BZR    -batch_size 64 -num_epoch 400 -alpha 0   -per 0.3 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/per_ood.py -exp_type ad -DS COX2   -batch_size 64 -num_epoch 150  -alpha 0.2   -per 0.3 -model GraphDE -num_layer 2 -hidden_dim 64

# OCGIN
python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full       -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.1 -model OCGIN OCGIN

python benchmark/per_ood.py -exp_type ad -DS BZR       -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.1 -model OCGIN OCGIN

python benchmark/per_ood.py -exp_type ad -DS COX2      -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 3   -per 0.1 -model OCGIN OCGIN


python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full       -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.2 -model OCGIN OCGIN

python benchmark/per_ood.py -exp_type ad -DS BZR       -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.2 -model OCGIN OCGIN

python benchmark/per_ood.py -exp_type ad -DS COX2      -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 3   -per 0.2 -model OCGIN OCGIN


python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full       -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.3 -model OCGIN OCGIN

python benchmark/per_ood.py -exp_type ad -DS BZR       -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.3 -model OCGIN OCGIN

python benchmark/per_ood.py -exp_type ad -DS COX2      -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 3   -per 0.3 -model OCGIN OCGIN



# OCGTL
python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full      -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4   -per 0.1 -model OCGTL OCGTL

python benchmark/per_ood.py -exp_type ad -DS BZR         -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.0001 -hidden_dim 64 -num_layer 4   -per 0.1 -model OCGTL OCGTL

python benchmark/per_ood.py -exp_type ad -DS COX2     -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 64 -num_layer 3   -per 0.1 -model OCGTL OCGTL



# SIGNET

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.001 -hidden_dim 128    -per 0.1 -model SIGNET SIGNET  -encoder_layers 3

python benchmark/per_ood.py -exp_type ad -DS BZR       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.001 -hidden_dim 128    -per 0.1 -model SIGNET SIGNET -encoder_layers 4

python benchmark/per_ood.py -exp_type ad -DS COX2       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 64    -per 0.1 -model SIGNET SIGNET -encoder_layers 5

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.001 -hidden_dim 128    -per 0.2 -model SIGNET SIGNET  -encoder_layers 3

python benchmark/per_ood.py -exp_type ad -DS BZR       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.001 -hidden_dim 128    -per 0.2 -model SIGNET SIGNET -encoder_layers 4

python benchmark/per_ood.py -exp_type ad -DS COX2       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 64    -per 0.2 -model SIGNET SIGNET -encoder_layers 5

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.001 -hidden_dim 128    -per 0.3 -model SIGNET SIGNET  -encoder_layers 3

python benchmark/per_ood.py -exp_type ad -DS BZR       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.001 -hidden_dim 128    -per 0.3 -model SIGNET SIGNET -encoder_layers 4

python benchmark/per_ood.py -exp_type ad -DS COX2       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 64    -per 0.3 -model SIGNET SIGNET -encoder_layers 5


# GCL-IF

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5


# GCL-OCSVM
python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


# IFGraph-IF

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5


# IFGraph-OCSVM

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.1  -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.1 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.2 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/per_ood.py -exp_type ad -DS PROTEINS_full  -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/per_ood.py -exp_type ad -DS BZR -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/per_ood.py -exp_type ad -DS COX2 -per 0.3 -batch_size  128 -batch_size_test 9999 -eval_freq 2 -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

