
# 11 Tudataset
python benchmark/mymain.py -exp_type ad -DS AIDS -num_epoch 400 -num_cluster 3 -alpha 1.0 -model GOOD-D

python benchmark/mymain.py -exp_type ad -DS BZR -num_epoch 400 -num_cluster 2 -alpha 0.8 -model GOOD-D

python benchmark/mymain.py -exp_type ad -DS COLLAB -batch_size 64 -batch_size_test 64 -num_epoch 400 -num_cluster 2 -alpha 0.8 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS COX2 -num_epoch 400 -num_cluster 3 -alpha 0.4 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS DD -batch_size 16 -batch_size_test 16 -num_epoch 50 -num_cluster 2 -alpha 1.0 -model GOOD-D

python benchmark/mymain.py -exp_type ad -DS DHFR -num_epoch 400 -num_cluster 2 -alpha 0 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS ENZYMES -num_epoch 400 -num_cluster 10 -alpha 0.2 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS IMDB-BINARY -num_epoch 400 -num_cluster 10 -alpha 0.2 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS NCI1 -batch_size 64 -batch_size_test 64 -num_epoch 400 -num_cluster 20 -alpha 1.0 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS PROTEINS_full -num_epoch 400 -num_cluster 2 -alpha 0.2 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS REDDIT-BINARY -batch_size 16 -batch_size_test 16 -num_epoch 80 -num_cluster 30 -alpha 0.8 -model GOOD-D

# 4 true dataset
python benchmark/mymain.py  -exp_type ad -DS Tox21_MMP -num_epoch 400 -num_cluster 5 -alpha 0.0 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS Tox21_PPAR-gamma -num_epoch 4000 -num_cluster 10 -alpha 0.8 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS Tox21_p53 -num_epoch 400 -num_cluster 5 -alpha 0.2 -model GOOD-D

python benchmark/mymain.py  -exp_type ad -DS Tox21_HSE -num_epoch 400 -num_cluster 2 -alpha 0.2 -model GOOD-D


# double dataset

python benchmark/mymain.py -exp_type oodd -DS_pair AIDS+DHFR -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.2 -model GOOD-D

python benchmark/mymain.py -exp_type oodd -DS_pair BZR+COX2 -num_epoch 400 -num_cluster 2 -alpha 0 -model GOOD-D

python benchmark/mymain.py -exp_type oodd -DS_pair ENZYMES+PROTEINS -num_epoch 400 -num_cluster 15 -alpha 0.2 -model GOOD-D

python benchmark/mymain.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY -num_epoch 400 -num_cluster 5 -alpha 0.8 -model GOOD-D

python benchmark/mymain.py -exp_type oodd -DS_pair PTC_MR+MUTAG -num_epoch 400 -num_cluster 2 -alpha 0.8 -model GOOD-D

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molbbbp+ogbg-molbace -batch_size_test 128 -num_epoch 400 -num_cluster 30 -alpha 0.2 -model GOOD-D

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molfreesolv+ogbg-moltoxcast -batch_size_test 128 -num_epoch 400 -num_cluster 2 -alpha 0.6 -model GOOD-D

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molclintox+ogbg-mollipo -batch_size_test 128 -num_epoch 400 -num_cluster 30 -alpha 1.0 -model GOOD-D

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -batch_size_test 128 -num_epoch 400 -num_cluster 20 -alpha 0.4 -model GOOD-D

python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-moltox21+ogbg-molsider -batch_size_test 128 -num_epoch 400 -num_cluster 5 -alpha 0.2 -model GOOD-D



