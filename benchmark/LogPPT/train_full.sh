# export CUDA_VISIBLE_DEVICES=5

#for dataset in Apache BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Zookeeper;
for dataset in Apache; do
	shot=32
	trf="datasets/${dataset}/${shot}shot/2.json"
	tef="datasets/${dataset}/test.json"
	python train.py --mode prompt-tuning --train_file ${trf} \
		--validation_file ${tef} \
		--model_name_or_path "./pretrained_models/roberta-base" \
		--per_device_train_batch_size 8 \
		--learning_rate 5e-5 \
		--lr_scheduler_type polynomial \
		--task_name log-parsing \
		--num_warmup_steps 20 \
		--max_train_steps 200 \
		--log_file ../../full_dataset/${dataset}/${dataset}_full.log_structured.csv \
		--shot $shot \
		--dataset_name ${dataset} \
		--task_output_dir "result_LogPPT_full"
done
