# export CUDA_VISIBLE_DEVICES=5
set -ex
for dataset in Apache BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Zookeeper; do
	shot=32
	trf="datasets-wo-virtual-label/${dataset}/${shot}shot/1.json"
	python x_train.py --train_file ${trf} \
		--model_name_or_path "./pretrained_models/roberta-base" \
		--per_device_train_batch_size 8 \
		--learning_rate 5e-5 \
		--lr_scheduler_type polynomial \
		--num_warmup_steps 20 \
		--max_train_steps 200 \
		--log_file datasets-wo-virtual-label/${dataset}/${dataset}_2k.log_structured.csv \
		--shot $shot \
		--dataset_name ${dataset} \
		--task_output_dir "result_LogPPT_2k-wo-virtual-label"
done
