import re


def get_parameter_list(s, template_regex):
    """
    :param s: log message
    :param template_regex: template regex with <*> indicates parameters
    :return: list of parameters
    """
    # template_regex = re.sub(r"<.{1,5}>", "<*>", template_regex)
    if "<*>" not in template_regex:
        return []
    template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
    template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
    template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
    parameter_list = re.findall(template_regex, s)
    parameter_list = parameter_list[0] if parameter_list else ()
    parameter_list = list(parameter_list) if isinstance(
        parameter_list, tuple) else [parameter_list]
    return parameter_list


def x_parsing_tokenize_dataset(
    tokenizer,
    dataset,
    text_column_name,
    label_column_name,
    max_length,
    padding,
    var_id,
    model_type="roberta",
):

    def tokenize_and_align_labels(examples):
        examples[text_column_name] = [
            " ".join(x.strip().split()) for x in examples[text_column_name]
        ]
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=max_length,
            padding=padding,
            truncation=True,
            is_split_into_words=False,
        )
        target_tokens = []
        for i, label in enumerate(examples[label_column_name]):
            content = examples[text_column_name][i]
            label = " ".join(label.strip().split())
            variable_list = get_parameter_list(content, label)
            input_ids = tokenized_inputs.input_ids[i]
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

            target_token = []
            processing_variable = False
            variable_token = ""
            input_tokens = [
                tokenizer.convert_tokens_to_string([x]) for x in input_tokens
            ]
            # pos = 0
            for ii, (input_idx,
                     input_token) in enumerate(zip(input_ids, input_tokens)):
                if input_idx in tokenizer.all_special_ids:
                    target_token.append(-100)
                    continue
                # Set target token for the first token of each word.
                if (label[:3] == "<*>" or label[:len(input_token.strip())] != input_token.strip()) \
                        and processing_variable is False:
                    processing_variable = True
                    variable_token = variable_list.pop(0).strip()
                    pos = label.find("<*>")
                    label = label[label.find("<*>") + 3:].strip()
                    input_token = input_token.strip()[pos:]

                if processing_variable:
                    input_token = input_token.strip()
                    if input_token == variable_token[:len(input_token)]:
                        target_token.append(var_id)
                        variable_token = variable_token[len(input_token
                                                           ):].strip()
                        # print(variable_token, "+++", input_token)
                    elif len(input_token) > len(variable_token):
                        target_token.append(var_id)
                        label = label[len(input_token) -
                                      len(variable_token):].strip()
                        variable_token = ""
                    else:
                        raise ValueError(
                            f"error at {variable_token} ---- {input_token}")
                    if len(variable_token) == 0:
                        processing_variable = False
                else:
                    input_token = input_token.strip()
                    if input_token == label[:len(input_token)]:
                        target_token.append(input_idx)
                        label = label[len(input_token):].strip()
                    else:
                        raise ValueError(
                            f"error at {content} ---- {input_token}")

            target_tokens.append(target_token)
            tokenized_inputs.input_ids[i] = input_ids
        tokenized_inputs["labels"] = target_tokens
        return tokenized_inputs

    processed_raw_datasets = {}
    processed_raw_datasets['train'] = dataset['train'].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    return processed_raw_datasets
