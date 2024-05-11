# 11 Tudataset
python benchmark/mymain.py -exp_type ad -DS AIDS -batch_size 64 -num_epoch 400  -alpha 1.0 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type ad -DS BZR -batch_size 64 -num_epoch 400  -alpha 0.8 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type ad -DS COLLAB -batch_size 64 -batch_size_test 16 -num_epoch 100 -alpha 0.8 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS COX2 -batch_size 64 -num_epoch 150 -alpha 0.4 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS DD -batch_size 64 -batch_size_test 16 -num_epoch 50  -alpha 1.0 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type ad -DS DHFR -batch_size 64 -num_epoch 20 -alpha 0 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS ENZYMES -batch_size 64 -num_epoch 400  -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS IMDB-BINARY -batch_size 64 -num_epoch 400 -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS NCI1 -batch_size 64 -batch_size_test 64 -num_epoch 400  -alpha 1.0 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS PROTEINS_full -batch_size 64 -num_epoch 20  -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS REDDIT-BINARY -batch_size 64 -batch_size_test  -num_epoch 80  -alpha 0.8 -model GraphDE -num_layer 2 -hidden_dim 64

# 4 true dataset
python benchmark/mymain.py  -exp_type ad -DS Tox21_MMP -batch_size 64 -num_epoch 400 -alpha 0.0 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS Tox21_PPAR-gamma -batch_size 64 -num_epoch 200  -alpha 0.8 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS Tox21_p53 -batch_size 64 -num_epoch 150  -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py  -exp_type ad -DS Tox21_HSE -batch_size 64 -num_epoch 300   -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64


# double dataset

python benchmark/mymain.py -exp_type oodd -DS_pair AIDS+DHFR -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair BZR+COX2 -batch_size 64 -num_epoch 400 -alpha 0 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair ENZYMES+PROTEINS -batch_size 64 -num_epoch 150  -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY -batch_size 64 -num_epoch 20 -alpha 0.8 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair PTC_MR+MUTAG -batch_size 64 -num_epoch 400  -alpha 0.8 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molbbbp+ogbg-molbace -batch_size 64 -batch_size_test 128 -num_epoch 400  -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molfreesolv+ogbg-moltoxcast -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.6 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molclintox+ogbg-mollipo -batch_size 64 -batch_size_test 128 -num_epoch 300  -alpha 1.0 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.4 -model GraphDE -num_layer 2 -hidden_dim 64

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-moltox21+ogbg-molsider -batch_size 64 -batch_size_test 128 -num_epoch 400 -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64

# drugood

python benchmark/mymain.py -exp_type ood -DS DrugOOD -batch_size 64 -num_epoch 200 -alpha 0.2 -model GraphDE -num_layer 2 -hidden_dim 64
