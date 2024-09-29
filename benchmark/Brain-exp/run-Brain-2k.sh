#!/bin/bash

set -ex
for k in {0..0}; do
	echo $k

	cd ../evaluation/
	python Brain_eval.py -otc --output-dir ../Brain-exp/result-Brain-2k-9-28-$k

	cd ../Brain-exp
done
