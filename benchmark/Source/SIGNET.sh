python benchmark/mymain.py -exp_type ad -DS AIDS           -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 128 -model SIGNET SIGNET  --readout concat  --encoder_layers 3

python benchmark/mymain.py -exp_type ad -DS BZR               -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.001 -hidden_dim 128 -model SIGNET SIGNET  --readout concat  --encoder_layers 4

python benchmark/mymain.py -exp_type ad -DS COLLAB          -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 64 -model SIGNET SIGNET  --readout concat  --encoder_layers 5

python benchmark/mymain.py -exp_type ad -DS COX2             -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 64 -model SIGNET SIGNET  --readout concat  --encoder_layers 5

python benchmark/mymain.py -exp_type ad -DS DD                 -batch_size  128 -batch_size_test 9999  -num_epoch 100 -lr 0.0001 -hidden_dim 128 -model SIGNET SIGNET  --readout concat  --encoder_layers 3

python benchmark/mymain.py -exp_type ad -DS DHFR             -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 128 -model SIGNET SIGNET  --readout concat  --encoder_layers 4

python benchmark/mymain.py -exp_type ad -DS ENZYMES           -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.001 -hidden_dim 128 -model SIGNET SIGNET  --readout concat  --encoder_layers 4

python benchmark/mymain.py -exp_type ad -DS IMDB-BINARY       -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 64 -model SIGNET SIGNET  --readout concat  --encoder_layers 5

python benchmark/mymain.py -exp_type ad -DS NCI1              -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 128 -model SIGNET SIGNET  --readout concat  --encoder_layers 5

python benchmark/mymain.py -exp_type ad -DS PROTEINS_full     -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.001 -hidden_dim 128 -model SIGNET SIGNET  --readout concat  --encoder_layers 3

python benchmark/mymain.py -exp_type ad -DS REDDIT-BINARY     -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 128 -model SIGNET SIGNET  --readout concat  --encoder_layers 3

# 4 true dataset
python benchmark/mymain.py -exp_type ad -DS Tox21_MMP           -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.0001 -hidden_dim 64 -model SIGNET SIGNET  --readout concat  --encoder_layers 3

python benchmark/mymain.py -exp_type ad -DS Tox21_PPAR-gamma    -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.001 -hidden_dim 64 -model SIGNET SIGNET  --readout concat  --encoder_layers 3

python benchmark/mymain.py -exp_type ad -DS Tox21_p53           -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.0001 -hidden_dim 128 -model SIGNET SIGNET  --readout concat  --encoder_layers 3

python benchmark/mymain.py -exp_type ad -DS Tox21_HSE           -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.001 -hidden_dim 64 -model SIGNET SIGNET  --readout concat  --encoder_layers 5


# # double dataset

# python benchmark/mymain.py -exp_type oodd -DS_pair AIDS+DHFR     -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.0001 -hidden_dim 64  -model SIGNET SIGNET  -readout concat  -encoder_layers 3

# python benchmark/mymain.py -exp_type oodd -DS_pair BZR+COX2     -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.001 -hidden_dim 64 -model SIGNET SIGNET  -readout concat  -encoder_layers 3

# python benchmark/mymain.py -exp_type oodd -DS_pair ENZYMES+PROTEINS     -batch_size  128 -batch_size_test 9999  -num_epoch  300 -lr 0.0001 -hidden_dim 64  -model SIGNET SIGNET  -readout concat  -encoder_layers 5

# python benchmark/mymain.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY     -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.001 -hidden_dim 64  -model SIGNET SIGNET  -readout concat  -encoder_layers 5

# python benchmark/mymain.py -exp_type oodd -DS_pair PTC_MR+MUTAG     -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.001 -hidden_dim 64  -model SIGNET SIGNET  -readout concat  -encoder_layers 5

# python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molbbbp+ogbg-molbace     -batch_size  128 -batch_size_test 9999  -num_epoch 300 -lr 0.001 -hidden_dim 128 -model SIGNET SIGNET  -readout concat  -encoder_layers 3

# python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molfreesolv+ogbg-moltoxcast     -batch_size  128 -batch_size_test 128  -num_epoch 300 -lr 0.001 -hidden_dim 64 -model SIGNET SIGNET  -readout concat  -encoder_layers 5

# python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molclintox+ogbg-mollipo     -batch_size  128 -batch_size_test 128  -num_epoch 300 -lr 0.0001 -hidden_dim 64 -model SIGNET SIGNET  -readout concat  -encoder_layers 4

# python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv    -batch_size  128 -batch_size_test 128 -num_epoch 300  -lr 0.001 -hidden_dim 64 -model SIGNET SIGNET  -readout concat  -encoder_layers 5

# python benchmark/mymain.py -exp_type oodd -DS_pair ogbg-moltox21+ogbg-molsider    -batch_size  128 -batch_size_test 128 -num_epoch 300  -lr 0.001 -hidden_dim 64 -model SIGNET SIGNET  -readout concat  -encoder_layers 5



