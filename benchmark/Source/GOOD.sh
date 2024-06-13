#DS = [GOODHIV+size+concept, GOODHIV+scaffold+concept, GOODZINC+size+concept, GOODZINC+scaffold+concept]
python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -rw_dim 16 -dg_dim 16 -hidden_dim 16 -num_epoch 300 -num_cluster 2 -alpha 0.0 -num_layer 3   -model CVTGAD CVTGAD -GNN_Encoder GIN -graph_level_pool global_mean_pool
python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256  -num_layer 2 -dropout 0.1  -lr 0.0001  -model GLADC GLADC
python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -num_epoch 50  -batch_size 300 -batch_size_test 1 -hidden_dim 32 -num_layer 3   -model GLocalKD GLocalKD -max-nodes 0
python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -num_epoch 200 -num_cluster 10 -alpha 0.2  -model GOOD-D
python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -batch_size 64 -num_epoch 200 -alpha 0.2 -model GraphDE -num_layer 2  -hidden_dim 64
python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -batch_size  128 -batch_size_test 128  -num_epoch 500 -lr 0.001 -hidden_dim 32  -num_layer 4 -model OCGIN OCGIN 
python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -batch_size  128 -batch_size_test 128  -num_epoch 500 -lr 0.001 -hidden_dim 32  -num_layer 4 -model OCGTL OCGTL
python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -batch_size  128 -batch_size_test 128 -num_epoch 300 -lr 0.001  -hidden_dim 16  -model SIGNET SIGNET 
python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept   -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_IF GraphCL_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_IF InfoGraph_IF -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -batch_size  128 -batch_size_test 9999  -num_epoch 30 -num_layer 2  -model InfoGraph_OCSVM InfoGraph_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -batch_size  128 -batch_size_test 9999  -num_epoch 30  -model GraphCL_OCSVM GraphCL_OCSVM -detector OCSVM -gamma 'scale' -nuOCSVM 0.1

python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept  -batch_size  128 -batch_size_test 1  -num_epoch 30  -model KernelGLAD KernelGLAD -detector IF -IF_n_trees 200  -IF_sample_ratio 0.5

python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept   -batch_size  128 -batch_size_test 1  -num_epoch 30 -gpu 3 -model KernelGLAD KernelGLAD -detector OCSVM -kernel WL

python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept   -batch_size  128 -batch_size_test 1  -num_epoch 30 -gpu 3 -model KernelGLAD KernelGLAD -detector OCSVM -kernel PK

python benchmark/mymain.py -exp_type ood -DS GOODHIV+size+concept   -batch_size  128 -batch_size_test 1  -num_epoch 30 -gpu 3 -model KernelGLAD KernelGLAD -detector IF -kernel PK
