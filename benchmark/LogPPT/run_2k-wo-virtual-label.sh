#!/bin/bash
#source ~/.zshrc

#conda activate LogPPT
#export CUDA_VISIBLE_DEVICES=4
set -ex
for k in {0..1}; do
	echo $k
	rm -rf datasets-wo-virtual-label
	python x_fewshot_sampling.py
	./x_train_2k_wo_virtual_label.sh

	mv result_LogPPT_2k-wo-virtual-label result-LogPPT-2k-wo-virtual-label-9-28-$k

	cd ../evaluation/
	#conda activate logevaluate
	python LogPPT_eval.py -otc --output-dir ../LogPPT/result-LogPPT-2k-wo-virtual-label-9-28-$k

	cd ../LogPPT
done
