python benchmark/mymain.py -exp_type ad -DS AIDS           -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 128   -dropout 0.1  -lr 0.001 -model GLADC GLADC -output_dim 64

python benchmark/mymain.py -exp_type ad -DS BZR            -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.3  -lr 0.001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS COLLAB         -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS COX2           -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.2  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS DD             -num_epoch 100   -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS DHFR           -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.2  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS ENZYMES        -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 128   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 64

python benchmark/mymain.py -exp_type ad -DS IMDB-BINARY    -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS NCI1           -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.2  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS PROTEINS_full  -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 128   -dropout 0.2  -lr 0.0001 -model GLADC GLADC -output_dim 64

python benchmark/mymain.py -exp_type ad -DS REDDIT-BINARY  -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 128

# 4 true dataset
python benchmark/mymain.py -exp_type ad -DS Tox21_MMP        -num_epoch 100  -batch_size 2000 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.00001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS Tox21_PPAR-gamma -num_epoch 100  -batch_size 2000 -batch_size_test 1 -hidden_dim 256   -dropout 0.2  -lr 0.00001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS Tox21_p53        -num_epoch 100  -batch_size 2000 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.00001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type ad -DS Tox21_HSE        -num_epoch 100  -batch_size 2000 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.00001 -model GLADC GLADC -output_dim 128


# double dataset

python benchmark/mymain.py -exp_type oodd -DS_pair AIDS+DHFR -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 128   -dropout 0.1  -lr 0.001 -model GLADC GLADC -output_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair BZR+COX2 -num_epoch 100  -batch_size 300 -batch_size_test 1  -hidden_dim 128   -dropout 0.1  -lr 0.001 -model GLADC GLADC -output_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair ENZYMES+PROTEINS -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 64   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 32

python benchmark/mymain.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.2  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type oodd -DS_pair PTC_MR+MUTAG -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molbbbp+ogbg-molbace -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molfreesolv+ogbg-moltoxcast -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molclintox+ogbg-mollipo -num_epoch 100  -batch_size 300 -batch_size_test 1  -hidden_dim 128   -dropout 0.3  -lr 0.001 -model GLADC GLADC -output_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.3  -lr 0.0001 -model GLADC GLADC -output_dim 128

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-moltox21+ogbg-molsider -num_epoch 100  -batch_size 300 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.0001 -model GLADC GLADC -output_dim 128


