#!/bin/bash
#source ~/.zshrc

#conda activate LogPPT
#export CUDA_VISIBLE_DEVICES=4
set -ex
for k in {0..1}; do
    echo $k
    rm -rf datasets
	python fewshot_sampling.py
	./train_2k.sh

    mv result_LogPPT_2k result-LogPPT-2k-9-28-$k

	cd ../evaluation/
	#conda activate logevaluate
	python LogPPT_eval.py -otc --output-dir ../LogPPT/result-LogPPT-2k-9-28-$k

	cd ../LogPPT
done
