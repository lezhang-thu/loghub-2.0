#!/bin/bash

set -ex
# Brain alg. is deterministic, so only keep ONE
brain_path="../Brain-exp/result-Brain-full-9-28-2"
brain_sample_prefix="../LogPPT/result-LogPPT-Brain-sample-full-9-28-"
for k in {0..1}; do
	echo $k
	python ../logparser-Brain/custom-eval/merge.py $brain_path $brain_sample_prefix$k brain-logppt-merge-9-30-$k
	cd ../evaluation/
	python LogPPT_eval.py -full --output-dir ../Brain-LogPPT/brain-logppt-merge-9-30-$k
	cd ../Brain-LogPPT
done
