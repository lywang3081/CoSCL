


#EXP1: CIFAR-100-SC

#EWC
python3 ./main.py --experiment split_cifar100_sc_5 --approach ewc --lamb 40000 --seed 0

#EWC+CPR
python3 ./main.py --experiment split_cifar100_sc_5 --approach ewc --lamb 40000 --lamb1 1.5 --seed 0

#EWC+AFEC
python3 ./main.py --experiment split_cifar100_sc_5 --approach afec_ewc --lamb 40000 --lamb_emp 1 --seed 0

#EWC+CoSCL(w/o TG)
python3 ./main.py --experiment split_cifar100_sc_5 --approach ewc_coscl --lamb 40000 --lamb1 0.02 --seed 0

#EWC+CoSCL(w/ TG)
python3 ./main.py --experiment split_cifar100_sc_5 --approach ewc_coscl --lamb 40000 --lamb1 0.02 --use_TG --s_gate 100 --seed 0

#MAS
python3 ./main.py --experiment split_cifar100_sc_5 --approach mas --lamb 16 --seed 0

#MAS+CPR
python3 ./main.py --experiment split_cifar100_sc_5 --approach mas --lamb 16 --lamb1 1.5 --seed 0

#MAS+AFEC
python3 ./main.py --experiment split_cifar100_sc_5 --approach afec_MAS --lamb 16 --lamb_emp 1 --seed 0

#MAS+CoSCL(w/o TG)
python3 ./main.py --experiment split_cifar100_sc_5 --approach mas_coscl --lamb 16 --lamb1 0.02 --seed 0

#MAS+CoSCL(w/ TG)
python3 ./main.py --experiment split_cifar100_sc_5 --approach mas_coscl --lamb 16 --lamb1 0.02 --use_TG --s_gate 100 --seed 0

#RWALK
python3 ./main.py --experiment split_cifar100_sc_5 --approach rwalk --lamb 128 --seed 0

#SI
python3 ./main.py --experiment split_cifar100_sc_5 --approach si --lamb 8 --seed 0

#HAT
python3 ./main.py --experiment split_cifar100_sc_5 --approach hat --gamma 500 --smax 200 --seed 0

#HAT+CoSCL
python3 ./main.py --experiment split_cifar100_sc_5 --approach hat_coscl --gamma 500 --smax 200 --seed 0

#AGS-CL
python3 ./main.py --experiment split_cifar100_sc_5 --approach gs --lamb 3200 --mu 10 --rho 0.3 --eta 0.9 --seed 0

#AGS-CL+CoSCL
python3 ./main.py --experiment split_cifar100_sc_5 --approach gs_coscl --lamb 3200 --mu 10 --rho 0.3 --eta 0.9 --seed 0

#Experience Replay(ER)
python3 ./main.py --experiment split_cifar100_sc_5 --approach er --seed 0

#ER+CoSCL
python3 ./main.py --experiment split_cifar100_sc_5 --approach er_coscl --seed 0

#Finetune
python3 ./main.py --experiment split_cifar100_sc_5 --approach ft --seed 0


#EXP2: CIFAR-100-RS

#EWC
python3 ./main.py --experiment split_cifar100_rs_5 --approach ewc --lamb 10000 --seed 0

#EWC+CPR
python3 ./main.py --experiment split_cifar100_rs_5 --approach ewc --lamb 10000 --lamb1 1.5 --seed 0

#EWC+AFEC
python3 ./main.py --experiment split_cifar100_rs_5 --approach afec_ewc --lamb 10000 --lamb_emp 1 --seed 0

#EWC+CoSCL(w/o TG)
python3 ./main.py --experiment split_cifar100_rs_5 --approach ewc_coscl --lamb 10000 --lamb1 0.02 --seed 0

#EWC+CoSCL(w/ TG)
python3 ./main.py --experiment split_cifar100_rs_5 --approach ewc_coscl --lamb 10000 --lamb1 0.02  --use_TG --s_gate 100 --seed 0

#MAS
python3 ./main.py --experiment split_cifar100_rs_5 --approach mas --lamb 4 --seed 0

#MAS+CPR
python3 ./main.py --experiment split_cifar100_rs_5 --approach mas --lamb 4 --lamb1 1.5 --seed 0

#MAS+AFEC
python3 ./main.py --experiment split_cifar100_rs_5 --approach afec_MAS --lamb 4 --lamb_emp 1 --seed 0

#MAS+CoSCL(w/o TG)
python3 ./main.py --experiment split_cifar100_rs_5 --approach mas_coscl --lamb 4 --lamb1 0.02 --seed 0

#MAS+CoSCL(w/ TG)
python3 ./main.py --experiment split_cifar100_rs_5 --approach mas_coscl --lamb 4 --lamb1 0.02 --use_TG --s_gate 100 --seed 0

#RWALK
python3 ./main.py --experiment split_cifar100_rs_5 --approach rwalk --lamb 6 --seed 0

#SI
python3 ./main.py --experiment split_cifar100_rs_5 --approach si --lamb 10 --seed 0

#HAT
python3 ./main.py --experiment split_cifar100_rs_5 --approach hat --gamma 500 --smax 200 --seed 0

#HAT+CoSCL
python3 ./main.py --experiment split_cifar100_rs_5 --approach hat_coscl --gamma 500 --smax 200 --seed 0

#AGS-CL
python3 ./main.py --experiment split_cifar100_rs_5 --approach gs --lamb 1600 --mu 10 --rho 0.3 --eta 0.9 --seed 0

#AGS-CL+CoSCL
python3 ./main.py --experiment split_cifar100_rs_5 --approach gs_coscl --lamb 1600 --mu 10 --rho 0.3 --eta 0.9 --seed 0

#Experience Replay(ER)
python3 ./main.py --experiment split_cifar100_rs_5 --approach er --seed 0

#ER+CoSCL
python3 ./main.py --experiment split_cifar100_rs_5 --approach er_coscl --seed 0

#Finetune
python3 ./main.py --experiment split_cifar100_rs_5 --approach ft --seed 0



#EXP3: CUB-200-2011

#EWC
python3 ./main.py --dataset CUB200 --trainer ewc --lamb 1 --tasknum 10 --seed 0

#EWC+CPR
python3 ./main.py --dataset CUB200 --trainer ewc --lamb 1 --lamb1 1 --tasknum 10 --seed 0

#EWC+AFEC
python3 ./main.py --dataset CUB200 --trainer afec_ewc --lamb 1 --lamb_emp 0.001 --tasknum 10 --seed 0

#EWC+CoSCL
python3 ./main.py --dataset CUB200 --trainer ewc_coscl --lamb 1 --lamb1 0.0001 --s_gate 100 --tasknum 10 --seed 0

#MAS
python3 ./main.py --dataset CUB200 --trainer mas --lamb 0.01 --tasknum 10 --seed 0

#MAS+CPR
python3 ./main.py --dataset CUB200 --trainer mas --lamb 0.01 --lamb1 0.01 --tasknum 10 --seed 0

#MAS+AFEC
python3 ./main.py --dataset CUB200 --trainer afec_mas --lamb 0.01 --lamb_emp 0.001 --tasknum 10 --seed 0

#MAS+CoSCL
python3 ./main.py --dataset CUB200 --trainer mas_coscl --lamb 0.01 --lamb1 0.0001 --s_gate 100 --tasknum 10 --seed 0

#RWALK
python3 ./main.py --dataset CUB200 --trainer rwalk --lamb 25 --tasknum 10 --seed 0

#SI
python3 ./main.py --dataset CUB200 --trainer si --lamb 0.4 --tasknum 10 --seed 0



#EXP4: Tiny-ImageNet

#EWC
python3 ./main.py --dataset tinyImageNet --trainer ewc --lamb 80 --tasknum 10 --seed 0

#EWC+CPR
python3 ./main.py --dataset tinyImageNet --trainer ewc --lamb 80 --lamb1 0.6 --tasknum 10 --seed 0

#EWC+AFEC
python3 ./main.py --dataset tinyImageNet --trainer afec_ewc --lamb 80 --lamb_emp 0.1 --tasknum 10 --seed 0

#EWC+CoSCL
python3 ./main.py --dataset tinyImageNet --trainer ewc_coscl --lamb 320 --lamb1 0.001 --s_gate 100 --tasknum 10 --seed 0

#MAS
python3 ./main.py --dataset tinyImageNet --trainer mas --lamb 0.1 --tasknum 10 --seed 0

#MAS+CPR
python3 ./main.py --dataset tinyImageNet --trainer mas --lamb 0.1 --lamb1 0.1 --tasknum 10 --seed 0

#MAS+AFEC
python3 ./main.py --dataset tinyImageNet --trainer afec_mas --lamb 0.1 --lamb_emp 0.1 --tasknum 10 --seed 0

#MAS+CoSCL
python3 ./main.py --dataset tinyImageNet --trainer mas_coscl --lamb 0.1 --lamb1 0.001 --s_gate 100 --tasknum 10 --seed 0

#RWALK
python3 ./main.py --dataset tinyImageNet --trainer rwalk --lamb 5 --tasknum 10 --seed 0

#SI
python3 ./main.py --dataset tinyImageNet --trainer si --lamb 0.8 --tasknum 10 --seed 0
