# 11 Tudataset

python3 benchmark/mymain.py -exp_type ad -DS AIDS            -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 200 -num_cluster 3 -alpha 1.0 -num_layer 2  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS BZR            -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 1000 -num_cluster 2 -alpha 0.45 -num_layer 5  -lr 0.001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS COLLAB         -batch_size 64 -batch_size_test 64 -rw_dim 32 -dg_dim 32 -hidden_dim 16 -num_epoch 800 -num_cluster 2 -alpha 0.32 -num_layer 5  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS COX2           -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 1000 -num_cluster 3 -alpha 0.6 -num_layer 5  -lr 0.001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS DD             -rw_dim 16 -dg_dim 16 -hidden_dim 16 -batch_size 16 -batch_size_test 16 -num_epoch 100 -num_cluster 2 -alpha 1.0 -num_layer 5  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS DHFR           -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.0 -num_layer 3  -lr 0.001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS ENZYMES       -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 200 -num_cluster 10 -alpha 0.2 -num_layer 4  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS IMDB-BINARY   -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 800 -num_cluster 10 -alpha 0.2 -num_layer 4  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS NCI1          -rw_dim 16 -dg_dim 16 -hidden_dim 16 -batch_size 64 -batch_size_test 64 -num_epoch 400 -num_cluster 100 -alpha 1.0 -num_layer 4  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS PROTEINS_full  -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 600 -num_cluster 2 -alpha 0.2 -num_layer 5  -lr 0.001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS REDDIT-BINARY  -rw_dim 8 -dg_dim 8 -hidden_dim 16 -batch_size 16 -batch_size_test 16 -num_epoch 800 -num_cluster 30 -alpha 0.2 -num_layer 5  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

# 4 true dataset
python3 benchmark/mymain.py -exp_type ad -DS Tox21_MMP        -rw_dim 16 -dg_dim 16 -hidden_dim 8 -num_epoch 600 -num_cluster 5 -alpha 0.42 -num_layer 5  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS Tox21_PPAR-gamma -rw_dim 4 -dg_dim 4 -num_epoch 400 -num_cluster 10 -alpha 0.2 -num_layer 6 -lr 0.00001 -eval_freq 1 -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS Tox21_p53        -rw_dim 12 -dg_dim 12 -hidden_dim 24 -num_epoch 400 -num_cluster 5 -alpha 0.2 -num_layer 5  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type ad -DS Tox21_HSE        -rw_dim 4 -dg_dim 4 -hidden_dim 4 -num_epoch 400 -num_cluster 2 -alpha 0.32 -num_layer 5  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool


# double dataset

python3 benchmark/mymain.py -exp_type oodd -DS_pair AIDS+DHFR -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 5   -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type oodd -DS_pair BZR+COX2 -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.32 -num_layer 5   -lr 0.001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type oodd -DS_pair ENZYMES+PROTEINS -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 3   -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 5   -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type oodd -DS_pair PTC_MR+MUTAG -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.0 -num_layer 4   -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molbbbp+ogbg-molbace -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.32 -num_layer 3   -lr 0.001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molfreesolv+ogbg-moltoxcast -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 4   -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molclintox+ogbg-mollipo -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 3   -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -rw_dim 8 -dg_dim 8 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 4  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool

python3 benchmark/mymain.py -exp_type oodd -DS_pair ogbg-moltox21+ogbg-molsider -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.2 -num_layer 3  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool




