#!/bin/bash

bash runner.sh Diabetes seq mtbo cpu 2 &&
bash runner.sh Diabetes seq dnn_mfbo cpu 2 &&
bash runner.sh Diabetes seq mfhmc_cs cpu 2 &&
bash runner.sh Diabetes seq mfhmc_ucs cpu 2 && 
bash runner.sh Diabetes seq mfhmc_par_cs cpu 2 && 
bash runner.sh Diabetes seq mfhmc_greedy_cs cpu 2 &&
bash runner.sh Diabetes seq mfhmc_ao_cs cpu 2 &&  
bash runner.sh Diabetes seq mfhmc_full_random cpu 2 &&
bash runner.sh Diabetes seq mf_mes cpu 2 &&
bash runner.sh Diabetes seq mf_gp_ucb cpu 2 &&  
bash runner.sh Diabetes seq par_mf_mes cpu 2 &&
bash runner.sh Diabetes seq gp_kernel cpu 2 &&
bash runner.sh Diabetes seq smac cpu 2 &&  
bash runner.sh Diabetes seq hyperband cpu 2 &&
bash runner.sh Diabetes seq bohb cpu 2 &&  

# bash runner.sh Diabetes par mtbo cpu 2 &&  
# bash runner.sh Diabetes par dnn_mfbo cpu 2 &&
# bash runner.sh Diabetes par mfhmc_cs cpu 2 &&
# bash runner.sh Diabetes par mfhmc_ucs cpu 2 &&
# bash runner.sh Diabetes par mfhmc_par_cs cpu 2 &&
# bash runner.sh Diabetes par mfhmc_greedy_cs cpu 2 &&
# bash runner.sh Diabetes par mfhmc_ao_cs cpu 2 &&  
# bash runner.sh Diabetes par mfhmc_full_random cpu 2 &&
# bash runner.sh Diabetes par mf_mes cpu 2 &&
# bash runner.sh Diabetes par mf_gp_ucb cpu 2 &&  
# bash runner.sh Diabetes par par_mf_mes cpu 2 &&
# bash runner.sh Diabetes par gp_kernel cpu 2 &&
# bash runner.sh Diabetes par smac cpu 2 &&  
# bash runner.sh Diabetes par hyperband cpu 2 &&
# bash runner.sh Diabetes par bohb cpu 2

