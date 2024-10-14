
#!/bin/bash
set -ex
cd ../evaluation/
prefix="../LogPPT/result-LogPPT-full-wo-virtual-label-9-28-"
for frequent in 10 -10; do
	echo $frequent
	for k in {0..4}; do
		echo $k
		#conda activate logevaluate
		python LogPPT_eval.py -full --output-dir $prefix$k --frequent $frequent
		mv $prefix$k/frequent/* $prefix$k/
	done
	python ../average-over-seeds.py 5 $prefix$k/LogPPT_full_complex-0_frequent-$frequent.csv logppt-wo-vlw-$frequent.csv
done

