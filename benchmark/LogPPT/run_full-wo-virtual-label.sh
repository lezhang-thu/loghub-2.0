#!/bin/bash
#source ~/.zshrc

#conda activate LogPPT
#export CUDA_VISIBLE_DEVICES=4
set -ex
for k in {0..1}; do
	echo $k
	rm -rf datasets-wo-virtual-label
	python x_fewshot_sampling.py
	python convert_fewshot_label.py datasets-wo-virtual-label
	./x_train_full.sh

	mv result_LogPPT_full-wo-virtual-label result-LogPPT-full-wo-virtual-label-9-28-$k

	cd ../evaluation/
	#conda activate logevaluate
	python LogPPT_eval.py -full --output-dir ../LogPPT/result-LogPPT-full-wo-virtual-label-9-28-$k

	cd ../LogPPT
done
