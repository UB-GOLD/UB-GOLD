# AIDS
# BZR
# ENZYMES

#CVTGOD
python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 5   -lr 0.0001    -near far -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2 -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.32 -num_layer 5   -lr 0.001   -near far -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS  -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 3   -lr 0.0001 -near far   -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 5   -lr 0.0001    -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2 -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.32 -num_layer 5   -lr 0.001   -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS  -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 3   -lr 0.0001   -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool




#GLADC
python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 128   -dropout 0.1 -near far -lr 0.001 -model GLADC GLADC -output_dim 64

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2  -num_epoch 100  -batch_size 300 -batch_size_test 1  -hidden_dim 128   -dropout 0.1 -near far -lr 0.001 -model GLADC GLADC -output_dim 64

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS  -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 64   -dropout 0.1 -near far -lr 0.0001 -model GLADC GLADC -output_dim 32

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 128   -dropout 0.1  -lr 0.001 -model GLADC GLADC -output_dim 64

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2  -num_epoch 100  -batch_size 300 -batch_size_test 1  -hidden_dim 128   -dropout 0.1  -lr 0.001 -model GLADC GLADC -output_dim 64

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS  -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 64   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 32



#GLocalKD
python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR   -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.0001 -num_layer 3  -near far -model GLocalKD GLocalKD -output_dim 256

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2   -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 64 -num_layer 4  -near far -model GLocalKD GLocalKD -output_dim 128
python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.001 -num_layer 3  -near far -model GLocalKD GLocalKD -output_dim 256


python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR   -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.0001 -num_layer 3  -model GLocalKD GLocalKD -output_dim 256

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2   -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 64 -num_layer 4  -model GLocalKD GLocalKD -output_dim 128
python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS -num_epoch 150  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -lr 0.001 -num_layer 3  -model GLocalKD GLocalKD -output_dim 256


#GOOD-D
python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR  -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.2  -near far -model GOOD-D

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2  -num_epoch 400 -num_cluster 2 -alpha 0  -near far -model GOOD-D

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS -num_epoch 150 -num_cluster 15 -alpha 0.2  -near far -model GOOD-D

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR  -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.2 -model GOOD-D

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2  -num_epoch 400 -num_cluster 2 -alpha 0 -model GOOD-D

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS -num_epoch 150 -num_cluster 15 -alpha 0.2 -model GOOD-D

# #GraphDE

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.2  -near far -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2  -batch_size 64 -num_epoch 400 -alpha 0  -near far -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS -batch_size 64 -num_epoch 150  -alpha 0.2  -near far -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2  -batch_size 64 -num_epoch 400 -alpha 0 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS -batch_size 64 -num_epoch 150  -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

# OCGIN
python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR      -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4  -near far -model OCGIN OCGIN

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2     -batch_size  128 -batch_size_test 9999   -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.0001 -hidden_dim 32 -num_layer 4 -near far -model OCGIN OCGIN

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS    -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4  -near far -model OCGIN OCGIN

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR     -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4 -model OCGIN OCGIN

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2      -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.0001 -hidden_dim 32 -num_layer 4 -model OCGIN OCGIN

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS    -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4 -model OCGIN OCGIN

# OCGTL
python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR    -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 128 -num_layer 3 -near far -model OCGTL OCGTL

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2       -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.0001 -hidden_dim 32 -num_layer 4  -near far -model OCGTL OCGTL

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS   -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 64 -num_layer 3  -near far -model OCGTL OCGTL

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR    -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 128 -num_layer 3 -model OCGTL OCGTL

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2      -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.0001 -hidden_dim 32 -num_layer 4 -model OCGTL OCGTL

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS    -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 64 -num_layer 3 -model OCGTL OCGTL

# SIGNET

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR      -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.0001 -hidden_dim 64 -near far -model SIGNET SIGNET  -readout concat  -encoder_layers 3

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2    -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.001 -hidden_dim 64 -near far -model SIGNET SIGNET  -readout concat  -encoder_layers 3

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS    -batch_size  128 -batch_size_test 9999  -num_epoch  300 -lr 0.0001 -hidden_dim 64 -near far -model SIGNET SIGNET  -readout concat  -encoder_layers 5

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR      -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.0001 -hidden_dim 64  -model SIGNET SIGNET  -readout concat  -encoder_layers 3

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2    -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.001 -hidden_dim 64 -model SIGNET SIGNET  -readout concat  -encoder_layers 3

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS    -batch_size  128 -batch_size_test 9999  -num_epoch  300 -lr 0.0001 -hidden_dim 64  -model SIGNET SIGNET  -readout concat  -encoder_layers 5



# # GCL-IF
python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR   -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2    -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS   -near far  -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR    -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2      -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS     -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

# GCL-OCSVM

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR    -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2    -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS   -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR    -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2     -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS    -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1


# IFGraph-IF
python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR   -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2    -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS    -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR    -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2     -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS    -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5


# IFGraph-OCSVM

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR   -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2    -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS   -near far -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS AIDS -DS_pair AIDS+DHFR    -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS BZR -DS_pair BZR+COX2      -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/near_far_ood.py -exp_type ood -DS ENZYMES -DS_pair ENZYMES+PROTEINS    -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1




