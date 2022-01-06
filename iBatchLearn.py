import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen
import agents
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
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
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}



def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    if args.dataset == 'glue':
        task_names = ['sts-b', 'cola', 'mrpc', 'qnli', 'rte', 'sst-2', 'wnli']
        data_task_names = ['STS-B', 'CoLA', 'MRPC', 'QNLI', 'RTE', 'SST-2', 'WNLI']

    if args.model_type == 'transformer_models':

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.sub_model_type]

        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path.replace('_', '-'),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        # Prepare dataloaders
        if args.dataset == 'glue':
            train_datasets = {}
            val_datasets = {}
            for task_name, data_task_name in zip(task_names, data_task_names):
                train_datasets[task_name] = dataloaders.base.__dict__['GlueData'](args, task_name, data_task_name, tokenizer)
                val_datasets[task_name] = dataloaders.base.__dict__['GlueData'](args, task_name, data_task_name, tokenizer, evaluate=True)
        else:
            train_dataset = dataloaders.base.__dict__['GlueData'](args, args.task_name, tokenizer)
            val_dataset = dataloaders.base.__dict__['GlueData'](args, args.task_name, tokenizer, evaluate=True)
    else:
        # Prepare dataloaders
        train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.train_aug)


    if args.n_permutation>0:
        if args.model_type == 'transformer_models':
            if args.dataset == 'glue':
                train_datasets_splits = {}
                val_datasets_splits = {}
                task_output_spaces = {}
                for task_name in task_names:
                    train_datasets_splits[task_name], val_datasets_splits[task_name], task_output_spaces[
                        task_name] = PermutedGen(train_datasets[task_name],
                                                 val_datasets[task_name],
                                                 args.n_permutation,
                                                 remap_class=not args.no_class_remap,
                                                 is_transformer=True)
            else:
                train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                          args.n_permutation,
                                                                                          remap_class=not args.no_class_remap, is_transformer=True)
        else:
             train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                             args.n_permutation,
                                                                             remap_class=not args.no_class_remap)
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                          first_split_sz=args.first_split_size,
                                                                          other_split_sz=args.other_split_size,
                                                                          rand_split=args.rand_split,
                                                                          remap_class=not args.no_class_remap)

    if args.model_type == 'transformer_models':
        # Prepare the Agent (model)
        agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
                        'model_type':args.model_type, 'model_name': args.model_name_or_path, 'model_weights':args.model_weights,
                        'out_dim':{'All':args.force_out_dim} if args.force_out_dim>0 else task_output_space,
                        'optimizer':args.optimizer,
                        'print_freq':args.print_freq, 'gpuid': args.gpuid,
                        'reg_coef':args.reg_coef, 'task_name': args.task_name,
                        'cache_dir': args.cache_dir, 'sub_model_type': args.sub_model_type}
    else:
        # Prepare the Agent (model)
        agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
                        'model_type':args.model_type, 'model_name': args.model_name_or_path, 'model_weights':args.model_weights,
                        'out_dim':{'All':args.force_out_dim} if args.force_out_dim>0 else task_output_space,
                        'optimizer':args.optimizer,
                        'print_freq':args.print_freq, 'gpuid': args.gpuid,
                        'reg_coef':args.reg_coef}
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](args, agent_config)
    print(agent.model)
    print('#parameter of model:',agent.count_parameter())

    # Decide split ordering
    if args.dataset != 'glue':
        task_names = sorted(list(task_output_space.keys()), key=int)
        print('Task order:', task_names)
    else:
        task_ids = {}
        for task_name in task_names:
            task_ids[task_name] = sorted(list(task_output_spaces[task_name].keys()), key=int)
            print('Task_name {}, Task order {}:'.format(task_name, task_ids[task_name]))
    if args.rand_split_order:
        if args.dataset != 'glue':
            shuffle(task_names)
            print('Shuffled task order:', task_names)
        else:
            shuffle(task_ids)
            print('Shuffled task order:', task_ids)
    if args.offline_training:
        if args.output_mode == 'classification':
            acc_table = OrderedDict()
            mcc_table = OrderedDict()
        elif args.output_mode == 'regression':
            corr_table = OrderedDict()
    else:
        metrics_table = OrderedDict()
    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        task_names = ['All']
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                   batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader)

        if args.output_mode == 'classification':
            acc_table['All'] = {}
            acc_table['All']['All'] = agent.validation(val_loader)[0]
            mcc_table['All'] = {}
            mcc_table['All']['All'] = agent.validation(val_loader)[1]
        elif args.output_mode == 'regression':
            corr_table['All'] = {}
            corr_table['All']['All'] = agent.validation(val_loader)
        if args.output_mode == 'classification':
            return acc_table, mcc_table, task_names
        elif args.output_mode == 'regression':
            return corr_table, task_names

    else:  # Incremental learning
        # Feed data to agent and evaluate agent's performance
        for k in range(len(task_names)):
            train_name = task_names[k]
            if args.dataset == 'glue':
                train_dataset_all = torch.utils.data.ConcatDataset(train_datasets_splits[train_name].values())
                val_dataset_all = torch.utils.data.ConcatDataset(val_datasets_splits[train_name].values())
                train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                           batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.workers)
                val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                         batch_size=args.batch_size, shuffle=False,
                                                         num_workers=args.workers)
                print('======================',train_name,'=======================')
            else:
                train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                            batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
                val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                          batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            if args.incremental_class:
                if args.dataset == 'glue':
                    agent.add_valid_output_dim(task_output_spaces[train_name]['1'])
                else:
                    agent.add_valid_output_dim(task_output_space[train_name]['1'])

            # Learn
            agent.learn_batch(train_name, train_loader, val_loader)

            # Evaluate
            metrics_table[train_name] = OrderedDict()
            for j in range(k+1):
                val_name = task_names[j]
                print('validation split name:', val_name)
                val_data = val_dataset_all if not args.eval_on_train_set else train_dataset_all
                val_loader = torch.utils.data.DataLoader(val_data,
                                                         batch_size=args.batch_size, shuffle=False,
                                                         num_workers=args.workers)
                metrics_table[val_name][train_name] = agent.validation(val_name, val_loader)

        return metrics_table, task_names

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--log_path",
        default=None,
        type=str,
        required=False,
        help="Path to the logging file.",)
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--sub_model_type', type=str, default='bert',
                        help="The type (mlp|lenet|vgg|resnet|bert) of backbone network")
    parser.add_argument('--model_name_or_path', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=2, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=3, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument("--tokenizer_name",default="",type=str,help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=False,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0.], help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")


    parser.add_argument("--train_logging_steps", type=int, default=100, help="Log training info every X updates steps.")
    parser.add_argument("--eval_logging_steps", type=int, default=500, help="Evaluate every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--logging_Euclid_dist", action="store_true",
                        help="Whether to log the Euclidean distance between the pretrained model and fine-tuning model")
    parser.add_argument("--start_from_pretrain", action="store_true",
                        help="Whether to initialize the model with pretrained parameters")
    parser.add_argument('--reg_lambda', default=1.0, type=float,
                        help='Regularization parameter')
    parser.add_argument("--update_epoch", default=1, type=int, help="per update_epoch update omega")
    parser.add_argument("--logits_calibraion_degree", default=1.0, type=float, help="the degree of logits calibration")
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    if args.dataset == 'glue':
        task_names = ['sts-b', 'cola', 'mrpc', 'qnli', 'rte', 'sst-2', 'wnli']
        data_task_names = ['STS-B', 'CoLA', 'MRPC', 'QNLI', 'RTE', 'SST-2', 'WNLI']
    if args.offline_training:
        args.output_mode = output_modes[args.task_name]
        reg_coef_list = args.reg_coef
        if args.task_name == 'cola':
            avg_final_mcc = {}
        elif args.output_mode == 'classification':
            avg_final_acc = {}
        elif args.output_mode == 'regression':
            avg_final_corr = {}

        # The for loops over hyper-paramerters or repeats
        for reg_coef in reg_coef_list:
            args.reg_coef = reg_coef
            if args.task_name == 'cola':
                avg_final_mcc[reg_coef] = np.zeros(args.repeat)
            elif args.output_mode == 'classification':
                avg_final_acc[reg_coef] = np.zeros(args.repeat)
            elif args.output_mode == 'regression':
                avg_final_corr[reg_coef] = np.zeros(args.repeat)
            for r in range(args.repeat):

                # Run the experiment
                if args.task_name == 'cola':
                    mcc_table, task_names = run(args)
                    print(mcc_table)
                elif args.output_mode == 'classification':
                    acc_table, task_names = run(args)
                    print(acc_table)
                elif args.output_mode == 'regression':
                    corr_table, task_names = run(args)
                    print(corr_table)


                # Calculate average performance across tasks
                # Customize this part for a different performance metric
                if args.output_mode == 'classification':
                    avg_acc_history = [0] * len(task_names)
                    avg_mcc_history = [0] * len(task_names)
                    for i in range(len(task_names)):
                        train_name = task_names[i]
                        cls_acc_sum = 0
                        cls_mcc_sum = 0
                        for j in range(i + 1):
                            val_name = task_names[j]
                            cls_acc_sum += acc_table[val_name][train_name]
                            cls_mcc_sum += mcc_table[val_name][train_name]
                        avg_acc_history[i] = cls_acc_sum / (i + 1)
                        avg_mcc_history[i] = cls_mcc_sum / (i + 1)
                        print('Task', train_name, 'average acc:', avg_acc_history[i], 'average mcc:', avg_mcc_history[i])
                elif args.output_mode == 'regression':
                    avg_corr_history = [0] * len(task_names)
                    for i in range(len(task_names)):
                        train_name = task_names[i]
                        cls_corr_sum = 0
                        for j in range(i + 1):
                            val_name = task_names[j]
                            cls_corr_sum += corr_table[val_name][train_name]
                        avg_corr_history[i] = cls_corr_sum / (i + 1)
                        print('Task', train_name, 'average corr:', avg_corr_history[i])

                # Gather the final avg accuracy
                if args.output_mode == 'classification':
                    avg_final_acc[reg_coef][r] = avg_acc_history[-1]
                    avg_final_mcc[reg_coef][r] = avg_mcc_history[-1]
                elif args.output_mode == 'regression':
                    avg_final_corr[reg_coef][r] = avg_corr_history[-1]

                # Print the summary so far
                print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
                print('The regularization coefficient:', args.reg_coef)
                if args.output_mode == 'classification':
                    print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
                    print('The last avg mcc of all repeats:', avg_final_mcc[reg_coef])
                    print('mean acc:', avg_final_acc[reg_coef].mean(), 'std acc:', avg_final_acc[reg_coef].std())
                    print('mean mcc:', avg_final_mcc[reg_coef].mean(), 'std mcc:', avg_final_mcc[reg_coef].std())
                elif args.output_mode == 'regression':
                    print('The last avg corr of all repeats:', avg_final_corr[reg_coef])
                    print('mean corr:', avg_final_corr[reg_coef].mean(), 'std corr:', avg_final_corr[reg_coef].std())
        if args.output_mode == 'classification':
            for reg_coef,v in avg_final_acc.items():
                print('reg_coef:', reg_coef,'mean acc:', avg_final_acc[reg_coef].mean(), 'std acc:', avg_final_acc[reg_coef].std())
            for reg_coef,v in avg_final_mcc.items():
                print('reg_coef:', reg_coef,'mean mcc:', avg_final_mcc[reg_coef].mean(), 'std mcc:', avg_final_mcc[reg_coef].std())
        elif args.output_mode == 'regression':
            for reg_coef,v in avg_final_corr.items():
                print('reg_coef:', reg_coef,'mean corr:', avg_final_corr[reg_coef].mean(), 'std corr:', avg_final_corr[reg_coef].std())
    else:
        args.output_mode = output_modes[args.task_name]
        reg_coef_list = args.reg_coef
        avg_final_metrics = {}

        # The for loops over hyper-paramerters or repeats
        for reg_coef in reg_coef_list:
            args.reg_coef = reg_coef
            avg_final_metrics[reg_coef] = np.zeros(args.repeat)
            for r in range(args.repeat):

                # Run the experiment
                metrics_table, task_names = run(args)
                print(metrics_table)

                # Calculate average performance across tasks
                # Customize this part for a different performance metric
                avg_metrics_history = [0] * len(task_names)
                for i in range(len(task_names)):
                    train_name = task_names[i]
                    cls_metrics_sum = 0
                    for j in range(i + 1):
                        val_name = task_names[j]
                        cls_metrics_sum += metrics_table[val_name][train_name]
                    avg_metrics_history[i] = cls_metrics_sum / (i + 1)
                    print('Task', train_name, 'average metric:', avg_metrics_history[i])

                # Gather the final avg metrics
                avg_final_metrics[reg_coef][r] = avg_metrics_history[-1]

                # Print the summary so far
                print('===Summary of experiment repeats:', r + 1, '/', args.repeat, '===')
                print('The regularization coefficient:', args.reg_coef)
                print('The last avg corr of all repeats:', avg_final_metrics[reg_coef])
                print('mean metrics:', avg_final_metrics[reg_coef].mean(), 'std metrics:', avg_final_metrics[reg_coef].std())

        for reg_coef, v in avg_final_metrics.items():
            print('reg_coef:', reg_coef, 'mean metric:', avg_final_metrics[reg_coef].mean(), 'std metric:',
                  avg_final_metrics[reg_coef].std())
