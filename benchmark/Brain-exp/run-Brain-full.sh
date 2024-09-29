#!/bin/bash

set -ex
for k in {2..2}; do
	echo $k

	cd ../evaluation/
	python Brain_eval.py -full --output-dir ../Brain-exp/result-Brain-full-9-28-$k

	cd ../Brain-exp
done
