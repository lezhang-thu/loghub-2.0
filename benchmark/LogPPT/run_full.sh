#!/bin/bash
#source ~/.zshrc

#conda activate LogPPT
#export CUDA_VISIBLE_DEVICES=4
set -ex
for k in {0..1}; do
	echo $k
	rm -rf datasets
	python fewshot_sampling.py
	python convert_fewshot_label.py datasets
	./train_full.sh

	mv result_LogPPT_full result-LogPPT-full-9-28-$k

	cd ../evaluation/
	#conda activate logevaluate
	python LogPPT_eval.py -full --output-dir ../LogPPT/result-LogPPT-full-9-28-$k

	cd ../LogPPT
done
