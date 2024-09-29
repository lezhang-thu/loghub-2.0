#!/bin/bash
#source ~/.zshrc

#conda activate LogPPT
#export CUDA_VISIBLE_DEVICES=4
set -ex
for k in {0..1}; do
	echo $k
	rm -rf datasets-brain-help-sample-from-full
	python brain-sample-from-full.py ../Brain-exp/result-Brain-full-9-28-2
	./brain-logppt-train_full.sh

	mv result_LogPPT-Brain-sample_full result-LogPPT-Brain-sample-full-9-28-$k

	cd ../evaluation/
	python LogPPT_eval.py -full --output-dir ../LogPPT/result-LogPPT-Brain-sample-full-9-28-$k

	cd ../LogPPT
done
