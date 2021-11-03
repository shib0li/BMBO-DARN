#!/bin/bash

# cd ../..

domain=$1
horizon=$2

ntrials=1
burn=10
ns=10

n_init_1=10
n_init_2=10
n_init_3=10

if [[ "$domain" == "NewsGroup" ]];
then
    c1=1
    c2=10
    c3=50
elif [[ "$domain" == "Diabetes" ]];
then
    c1=2
    c2=50
    c3=100
elif [[ "$domain" == "BurgersShock" ]];
then
    c1=10
    c2=100
    c3=50000
elif [[ "$domain" == "Cifar10" ]];
then
    c1=1
    c2=1
    c3=1
else
    echo "Error: unrecognized domains"
    exit 1
fi


python main.py \
-algorithm_name='ao_batch_hmc_cs' -horizon=$horizon -num_trials=$ntrials -init_i_trial=0 \
-constraint=True -batch_size=5 \
-hidden_widths=[40,40,40] -hidden_depths=[2,2,2] -activation='tanh' -surrogate_placement='cpu' \
-step_size=0.012 -L=10 -burn=$burn -Ns=$ns \
-domain_name=$domain -Nfid=3 -num_inits=[$n_init_1,$n_init_2,$n_init_3] \
-penalty=[$c1,$c2,$c3] -domain_placement='cpu'














