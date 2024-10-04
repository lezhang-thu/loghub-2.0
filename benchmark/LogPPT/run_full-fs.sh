#!/bin/bash
#source ~/.zshrc

#conda activate LogPPT
#export CUDA_VISIBLE_DEVICES=4
set -ex
for k in {1..4}; do
	echo $k
	rm -rf datasets-fs-full-wo-vl
	python fs-full-wo-vl.py ../Brain-exp/result-Brain-full-9-28-2
	./fs-train_full.sh

	mv result_LogPPT-fs_full result-LogPPT-fs_full-10-1-$k

	cd ../evaluation/
	python LogPPT_eval.py -full --output-dir ../LogPPT/result-LogPPT-fs_full-10-1-$k

	cd ../LogPPT
done
