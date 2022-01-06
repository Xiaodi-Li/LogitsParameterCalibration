from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, matthews, pearson_and_spearman, AverageMeter, Timer
from transformers.modeling_outputs import SequenceClassifierOutput
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
from transformers import glue_output_modes as output_modes

class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, args, agent_config):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super(NormalNN, self).__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        self.args = args
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(self.config['out_dim'])>1 else False  # A convenience flag to indicate multi-head/task
        self.model = self.create_model()
        # if self.args.output_mode == 'classification':
        #     self.criterion_fn = nn.CrossEntropyLoss()
        # elif self.args.output_mode == 'regression':
        #     self.criterion_fn = nn.MSELoss()
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()
        self.reset_optimizer = False
        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
                                    # Set a interger here for the incremental class scenario

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=0.1)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        ##models.__dict__['transformer_models'].__dict__['TRANSFORMERS']
        if cfg['model_type'] == 'transformer_models':
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](cfg)
        else:
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()

        if cfg['model_type']!='transformer_models':
            # Apply network surgery to the backbone
            # Create the heads for tasks (It can be single task or multi-task)
            n_feat = model.last.in_features

            # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
            # For a single-headed model the output will be {'All':output}
            model.last = nn.ModuleDict()
            for task,out_dim in cfg['out_dim'].items():
                model.last[task] = nn.Linear(n_feat,out_dim)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def forward(self, x):
        if isinstance(self.model, BertForSequenceClassification):
            #print('it is in default.py line 123')
            #print(x)
            return self.model.forward(**x)
        elif isinstance(self.model, AlbertForSequenceClassification):
            return self.model.forward(**x)
        else:
            return self.model.forward(x)

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return out

    def validation(self, val_name, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        output_mode = output_modes[val_name]
        if val_name == 'cola':
            mcc = AverageMeter()
            self.criterion_fn = nn.CrossEntropyLoss().cuda()
        elif output_mode == 'classification':
            acc = AverageMeter()
            self.criterion_fn = nn.CrossEntropyLoss().cuda()
        elif output_mode == 'regression':
            corr = AverageMeter()
            self.criterion_fn = nn.MSELoss().cuda()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    target = target.cuda()
                    if isinstance(input, list):
                        input = tuple(t.cuda() for t in input)
                        input = {"input_ids": input[0], "attention_mask": input[1], "token_type_ids": (input[2]),  "labels": target}
                        #input["token_type_ids"] = (input[2])  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    else:
                        input = input.cuda()
            output = self.predict(input)

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            if val_name == 'cola':
                mcc = accumulate_mcc(output, target, task, mcc)
            elif output_mode == 'classification':
                acc = accumulate_acc(output, target, task, acc)
            elif output_mode == 'regression':
                corr = accumulate_corr(output, target, task, corr)

        self.train(orig_mode)
        if val_name == 'cola':
            self.log(' * Val mcc {mcc.avg:.3f}, Total time {time:.2f}'
                     .format(mcc=mcc, time=batch_timer.toc()))
        elif self.args.output_mode == 'classification':
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                  .format(acc=acc, time=batch_timer.toc()))
        elif self.args.output_mode == 'regression':
            self.log(' * Val corr {corr.avg:.3f}, Total time {time:.2f}'
                     .format(corr=corr, time=batch_timer.toc()))
        if val_name == 'cola':
            return mcc.avg
        elif self.args.output_mode == 'classification':
            return acc.avg
        elif self.args.output_mode == 'regression':
            return corr.avg

    def criterion(self, preds, targets, tasks, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.

        if self.multihead:
            loss = 0
            for t,t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i]==t]  # The index of inputs that matched specific task
                if len(inds)>0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            if 'All' in preds:
                pred = preds['All']
                if isinstance(self.valid_out_dim, int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                    pred = preds['All'][:,:self.valid_out_dim]
                loss = self.criterion_fn(pred, targets)
            else:
                loss = self.criterion_fn(preds[1], targets)
        return loss

    def update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

    def learn_batch(self, train_name, train_loader, val_loader=None):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        output_mode = output_modes[train_name]
        if output_mode == 'classification':
            self.criterion_fn = nn.CrossEntropyLoss().cuda()
        elif output_mode == 'regression':
            self.criterion_fn = nn.MSELoss().cuda()

        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            if train_name == 'cola':
                mcc = AverageMeter()
            elif output_mode == 'classification':
                acc = AverageMeter()
            elif output_mode == 'regression':
                corr = AverageMeter()

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            self.scheduler.step(epoch)
            for param_group in self.optimizer.param_groups:
                self.log('LR:',param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            for i, (input, target, task) in enumerate(train_loader):
                data_time.update(data_timer.toc())  # measure data loading time

                if self.gpu:
                    target = target.cuda()
                    if isinstance(input, list):
                        input = tuple(t.cuda() for t in input)
                        input = {"input_ids": input[0], "attention_mask": input[1], "token_type_ids": (input[2]),  "labels": target}
                        #input["token_type_ids"] = (input[2])  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    else:
                        input = input.cuda()

                loss, output = self.update_model(input, target, task)
                if isinstance(input, dict):
                    sample = input['input_ids'].detach()
                    losses.update(loss, sample.size(0))
                else:
                    input = input.detach()
                    losses.update(loss, input.size(0))
                target = target.detach()

                # measure accuracy, mcc, corr, and record loss
                if train_name == 'cola':
                    mcc = accumulate_mcc(output, target, task, mcc)
                elif output_mode == 'classification':
                    acc = accumulate_acc(output, target, task, acc)
                elif output_mode == 'regression':
                    corr = accumulate_corr(output, target, task, corr)

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

                if train_name == 'cola':
                    if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                        self.log('[{0}/{1}]\t'
                              '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                              '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                              '{loss.val:.3f} ({loss.avg:.3f})\t'
                              '{mcc.val:.2f} ({mcc.avg:.2f})'.format(
                            i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, mcc=mcc))
                elif output_mode == 'classification':
                    if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                        self.log('[{0}/{1}]\t'
                              '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                              '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                              '{loss.val:.3f} ({loss.avg:.3f})\t'
                              '{acc.val:.2f} ({acc.avg:.2f})'.format(
                            i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, acc=acc))
                elif output_mode == 'regression':
                    if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                        self.log('[{0}/{1}]\t'
                              '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                              '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                              '{loss.val:.3f} ({loss.avg:.3f})\t'
                              '{corr.val:.2f} ({corr.avg:.2f})'.format(
                            i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, corr=corr))
            if train_name == 'cola':
                self.log(' * Train mcc {mcc.avg:.3f}'.format(mcc=mcc))
            elif output_mode == 'classification':
                self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))
            elif output_mode == 'regression':
                self.log(' * Train corr {corr.avg:.3f}'.format(corr=corr))

            # Evaluate the performance of current task
            if val_loader != None:
                self.validation(train_name, val_loader)

    def learn_stream(self, data, label):
        assert False,'No implementation yet'

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model,torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        # self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

def accumulate_acc(output, target, task, meter):
    #print(output)
    if isinstance(output, SequenceClassifierOutput):
        tmp_eval_loss, logits = output[:2]
        meter.update(accuracy(logits, target), len(target))
    else:
        if 'All' in output.keys(): # Single-headed model
            meter.update(accuracy(output['All'], target), len(target))
        else:  # outputs from multi-headed (multi-task) model
            for t, t_out in output.items():
                inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
                if len(inds) > 0:
                    t_out = t_out[inds]
                    t_target = target[inds]
                    meter.update(accuracy(t_out, t_target), len(inds))

    return meter

def accumulate_mcc(output, target, task, meter):
    #print(output)
    if isinstance(output, SequenceClassifierOutput):
        tmp_eval_loss, logits = output[:2]
        meter.update(matthews(logits, target), len(target))
    else:
        if 'All' in output.keys(): # Single-headed model
            meter.update(matthews(output['All'], target), len(target))
        else:  # outputs from multi-headed (multi-task) model
            for t, t_out in output.items():
                inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
                if len(inds) > 0:
                    t_out = t_out[inds]
                    t_target = target[inds]
                    meter.update(matthews(t_out, t_target), len(inds))

    return meter

def accumulate_corr(output, target, task, meter):
    #print(output)
    if isinstance(output, SequenceClassifierOutput):
        tmp_eval_loss, logits = output[:2]
        meter.update(pearson_and_spearman(logits, target), len(target))
    else:
        if 'All' in output.keys(): # Single-headed model
            meter.update(pearson_and_spearman(output['All'], target), len(target))
        else:  # outputs from multi-headed (multi-task) model
            for t, t_out in output.items():
                inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
                if len(inds) > 0:
                    t_out = t_out[inds]
                    t_target = target[inds]
                    meter.update(pearson_and_spearman(t_out, t_target), len(inds))

    return meter
