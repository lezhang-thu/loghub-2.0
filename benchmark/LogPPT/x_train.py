import logging
import math

import datasets
import torch
import transformers

from torch.utils.data import DataLoader
#from tqdm import tqdm
from transformers import (HfArgumentParser, AutoConfig, default_data_collator,
                          get_scheduler, set_seed)
from logppt.models import load_model
from accelerate import Accelerator
import copy

from logppt.utils import MainArguments, ModelArguments, TrainArguments, TaskArguments, find_labels
from logppt.data import load_data_parsing, load_data_anomaly_detection, x_CustomDataCollator
from logppt.models import add_label_token
# debug
from logppt.tokenization import x_parsing_tokenize_dataset
from logppt.tasks.log_parsing import template_extraction

logger = logging.getLogger(__name__)
accelerator = Accelerator()

filter_list = ["and", "or", "the", "a", "of", "to", "at"]


def train():
    total_batch_size = train_args.per_device_train_batch_size * accelerator.num_processes * train_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {train_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {train_args.max_train_steps}")
    completed_steps = 0
    for epoch in range(train_args.num_train_epochs):
        model.train()
        total_loss = []
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss.append(float(loss))
            loss = loss / train_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % train_args.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            if completed_steps >= train_args.max_train_steps:
                break
    # hacker - start
    if main_args.brain is not None:
        import os
        print('Use brain to calibrate. Good!!!')
        task_args.log_file = os.path.join(main_args.brain,
                                          os.path.basename(task_args.log_file))
        print('task_args.log_file: {}'.format(task_args.log_file))
    # hacker - end
    template_extraction(tokenizer,
                        model,
                        accelerator,
                        task_args.log_file,
                        max_length=main_args.max_length,
                        model_name=model_type,
                        shot=main_args.shot,
                        dataset_name=task_args.dataset_name,
                        o_dir=task_args.task_output_dir,
                        mode=main_args.mode)


if __name__ == '__main__':
    parser = HfArgumentParser(
        (MainArguments, ModelArguments, TrainArguments, TaskArguments))
    main_args, model_args, train_args, task_args = parser.parse_args_into_dataclasses(
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if main_args.seed is not None:
        set_seed(main_args.seed)

    # Get the datasets: the data file are JSON files
    raw_datasets, text_column_name, label_name = load_data_parsing(main_args)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model_type = config.model_type
    tokenizer, model = load_model(model_args.model_name_or_path,
                                  model_type,
                                  mode=main_args.mode)

    tokenizer.add_tokens([
        "i-val",
    ])

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if main_args.pad_to_max_length else False

    processed_raw_datasets = x_parsing_tokenize_dataset(
        tokenizer, raw_datasets, text_column_name,
        label_name, main_args.max_length, padding,
        tokenizer.convert_tokens_to_ids("i-val"), model_type)
    train_dataset = processed_raw_datasets["train"]

    # `x_CustomDataCollator` will apply dynamic padding for us (by padding to the
    # maximum length of the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all
    # tensors to multiple of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute
    # capability >= 7.5 (Volta).
    data_collator = x_CustomDataCollator(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_args.per_device_train_batch_size)
    model = add_label_token(model_type,
                            model,
                            tokenizer, {"i-val": []},
                            wo_label_words=True)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": train_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=train_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_args.gradient_accumulation_steps)
    if train_args.max_train_steps is None:
        train_args.max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch
    else:
        train_args.num_train_epochs = math.ceil(train_args.max_train_steps /
                                                num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=train_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=train_args.num_warmup_steps,
        num_training_steps=train_args.max_train_steps,
    )
    train()
