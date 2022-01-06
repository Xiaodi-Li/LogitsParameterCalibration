# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE with LPC optimizer."""

def TRANSFORMERS(args, model_name_or_path):
    import argparse
    import glob
    import logging
    import os
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
    from torch.utils.data.distributed import DistributedSampler
    from tqdm import tqdm, trange

    from transformers import (
        WEIGHTS_NAME,
        AdamW,
        AlbertConfig,
        AlbertForSequenceClassification,
        AlbertTokenizer,
        AlbertForMaskedLM,
        BertConfig,
        BertForSequenceClassification,
        BertForPreTraining,
        BertTokenizer,
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizer,
        XLMConfig,
        XLMForSequenceClassification,
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
        XLMRobertaTokenizer,
        XLMTokenizer,
        XLNetConfig,
        XLNetForSequenceClassification,
        XLNetTokenizer,
        get_linear_schedule_with_warmup,
    )
    from transformers import glue_compute_metrics as compute_metrics
    from transformers import glue_convert_examples_to_features as convert_examples_to_features
    from transformers import glue_output_modes as output_modes
    from transformers import glue_processors as processors

    MODEL_CLASSES = {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
        "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
        "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    }

    # Prepare GLUE task
    args['task_name'] = args['task_name'].lower()
    if args['task_name'] not in processors:
        raise ValueError("Task not found: %s" % (args['task_name']))
    processor = processors[args['task_name']]()
    args['output_mode'] = output_modes[args['task_name']]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['sub_model_type']]
    config = config_class.from_pretrained(
        args['config_name'] if 'config_name' in args else model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args['task_name'],
        cache_dir=args['cache_dir'] if args['cache_dir'] else None,
    )
    pretrained_model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=args['cache_dir'] if args['cache_dir'] else None,
    )
    return pretrained_model

def bert_base_uncased(args):
    return TRANSFORMERS(args, 'bert-base-uncased')

def albert_xxlarge_v2(args):
    return TRANSFORMERS(args, 'albert-xxlarge-v2')