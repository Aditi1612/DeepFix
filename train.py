# Copyright 2022 Hyeon-Tae Seo, Su-Hyeon Kim, Sang-Ki Ko
# Kangwon National University
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


import argparse

import os
import time
import json
import shutil
import logging

import sklearn.metrics
from collections import Counter

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

import numpy as np

# torch utils
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim

# tensorboard
from torch.utils.tensorboard import SummaryWriter

# for distributed trainingd
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer, T5ForConditionalGeneration, RobertaTokenizer

# data utilis script
from utils.data_utils import DrRepairDatasetForCode2Code

# metrics
def f1_str_base(target, prediction):
    target = target.split()
    prediction = prediction.split()

    same = Counter(target) & Counter(prediction)
    num_same = sum(same.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(target)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def f1_score_dict(targets, predictions, inputs=None):
    return {"f1_score": 
             {'score': 100 * np.average(
                 np.array([f1_str_base(x, y) for x, y in zip(targets, predictions)], dtype=np.float64)),
                 'count': len(targets)}}

# seq acc
def accuracy_dict(targets, predictions, inputs=None):
    return {"accuracy":
             {'score': 100*sklearn.metrics.accuracy_score(
                 targets, predictions),
                 'count': len(targets)}}

# 1 - afterERR / beforeERR   (afterERR : edit-distance of prediction-target, beforeERR : edit-distance of inputs-target)
def fix_ratio_dict(targets, predictions, inputs):
    scores = [1 - levenshteinDistance(prediction, target) / levenshteinDistance(input_, target) for target, prediction, input_ in zip(targets, predictions, inputs)]
    scores = [max(x, 0) for x in scores]
    return {"fix_ratio":
             {'score': 100 * sum(scores) / len(scores) ,
                 'count': len(targets)}}

TASKS = {
    "drrepair_code2code": {
        "model": T5ForConditionalGeneration,
        "dataset": DrRepairDatasetForCode2Code,
        "metric": [accuracy_dict, f1_score_dict, fix_ratio_dict],
        "metric_key": ["accuracy", "f1_score", "fix_ratio"]
    },
    "drrepair_code2align": {
        "model": T5ForConditionalGeneration,
        "dataset": DrRepairDatasetForCode2Align,
        "metric": [accuracy_dict, f1_score_dict, fix_ratio_dict],
        "metric_key": ["accuracy", "f1_score", "fix_ratio"]
    }
}

def make_dir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def create_directory_info(args, create_dir=True):
    model_dir = os.path.join(args.output_dir,
        "{pre_trained_model}_{task}".format(
            pre_trained_model=args.pre_trained_model.replace('/', '_'),
            task = args.task))
    if args.dir_suffix is not None:
        model_dir = '_'.join([model_dir, args.dir_suffix])
    weights_dir = os.path.join(model_dir, "weights")
    logs_dir = os.path.join(model_dir, "logs")

    path_info = {
        'model_dir': model_dir,
        'weights_dir': weights_dir,
        'logs_dir': logs_dir,
    }

    if create_dir:
        for k, v in path_info.items():
            make_dir_if_not_exist(v)

    path_info['best_model_path'] = os.path.join(weights_dir, "best_model.pth")
    path_info['ckpt_path'] = os.path.join(weights_dir, "checkpoint.pth")
    return path_info


def get_env_var(env_var, type_cls, default_val):
    if env_var in os.environ:
        return type_cls(os.environ[env_var])
    return default_val

"""
For DDP
**Definitions:**
    1. ``Node`` - A physical instance or a container; maps to the unit that the job manager works with.
    2. ``Worker`` - A worker in the context of distributed training.
    3. ``WorkerGroup`` - The set of workers that execute the same function (e.g. trainers).
    4. ``LocalWorkerGroup`` - A subset of the workers in the worker group running on the same node.
    5. ``RANK`` - The rank of the worker within a worker group.
    6. ``WORLD_SIZE`` - The total number of workers in a worker group.
    7. ``LOCAL_RANK`` - The rank of the worker within a local worker group.
    8. ``LOCAL_WORLD_SIZE`` - The size of the local worker group.
    9. ``rdzv_id`` - A user-defined id that uniquely identifies the worker group for a job. This id is
    used by each node to join as a member of a particular worker group.
    9. ``rdzv_backend`` - The backend of the rendezvous (e.g. ``c10d``). This is typically a strongly
    consistent key-value store.
    10. ``rdzv_endpoint`` - The rendezvous backend endpoint; usually in form ``<host>:<port>``.
    A ``Node`` runs ``LOCAL_WORLD_SIZE`` workers which comprise a ``LocalWorkerGroup``. The union of
    all ``LocalWorkerGroups`` in the nodes in the job comprise the ``WorkerGroup``.
**Environment Variables:**
The following environment variables are made available to you in your script:
    1. ``LOCAL_RANK`` -  The local rank.
    2. ``RANK`` -  The global rank.
    3. ``GROUP_RANK`` - The rank of the worker group. A number between 0 and ``max_nnodes``. When
    running a single worker group per node, this is the rank of the node.
    4. ``ROLE_RANK`` -  The rank of the worker across all the workers that have the same role. The role
    of the worker is specified in the ``WorkerSpec``.
    5. ``LOCAL_WORLD_SIZE`` - The local world size (e.g. number of workers running locally); equals to
    ``--nproc_per_node`` specified on ``torch.distributed.run``.
    6. ``WORLD_SIZE`` - The world size (total number of workers in the job).
    7. ``ROLE_WORLD_SIZE`` - The total number of workers that was launched with the same role specified
    in ``WorkerSpec``.
    8. ``MASTER_ADDR`` - The FQDN of the host that is running worker with rank 0; used to initialize
    the Torch Distributed backend.
    9. ``MASTER_PORT`` - The port on the ``MASTER_ADDR`` that can be used to host the C10d TCP store.
    10. ``TORCHELASTIC_RESTART_COUNT`` - The number of worker group restarts so far.
    11. ``TORCHELASTIC_MAX_RESTARTS`` - The configured maximum number of restarts.
    12. ``TORCHELASTIC_RUN_ID`` - Equal to the rendezvous ``run_id`` (e.g. unique job id).
"""

def main():
    parser = argparse.ArgumentParser()

    # default settings
    parser.add_argument("--pre_trained_model",
                        default="Salesforce/codet5-base", type=str)
    parser.add_argument("--data_path",
                        default="/data", type=str)
    parser.add_argument("--output_dir",
                        default="output", type=str)
    parser.add_argument("--task",
                        default="drrepair_code2code", type=str)
    parser.add_argument("--dir_suffix", default=None,
                        type=str, help="suffix for output directory")

    # training configuration                    
    parser.add_argument("--batch_size", default=4, type=int,
                        help="batch size")
    parser.add_argument("--start_epoch", default=0, type=int,
                        help="start epoch")
    parser.add_argument("--epochs", default=10, type=int,
                        help="epochs")

    parser.add_argument("--print_freq", default=50, type=int,
                        help="print frequency")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="beam size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")


    # resume
    parser.add_argument("--resume", default=None, type=str,
                        help="path to checkpoint.")
    parser.add_argument("--hf_path", default=None, type=str,
                        help="path to score huggingface model")

    # hyper-parameters for optimizer
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=1e-3,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    # distributed settings
    parser.add_argument("--local_rank", type=int, default=0,
                        help="The rank of the worker within a local worker group.")
    parser.add_argument("--local_world_size", type=int, default=1,
                        help="The size of the local worker group.")
    parser.add_argument("--rank", type=int, default=0,
                        help="The rank of the worker within a worker group.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="The total number of workers in a worker group.")
    parser.add_argument("--distributed", action='store_true',
                        help="is distributed training")
    args = parser.parse_args()


    args.local_rank = get_env_var('LOCAL_RANK', int, args.local_rank)
    args.local_world_size = get_env_var('LOCAL_WORLD_SIZE', int, args.local_world_size)
    args.rank = get_env_var('RANK', int, args.rank)
    args.world_size = get_env_var('WORLD_SIZE', int, args.world_size)

     # check world size
    args.distributed = False
    args.distributed = args.world_size > 1

    # if the world size is bigger than 1, init process group(sync)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    device = torch.device("cuda")

    # create directory and summary logger
    summary_logger = None
    if args.local_rank == 0 or not args.distributed:
        path_info = create_directory_info(args)
        summary_logger = SummaryWriter(path_info["logs_dir"])
    path_info = create_directory_info(args, create_dir=False)

    # update batch_size per a device
    args.batch_size = int(
        args.batch_size / args.gradient_accumulation_steps)

    # get tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.pre_trained_model)
    tokenizer = RobertaTokenizer.from_pretrained(args.pre_trained_model)

    # get model
    model = TASKS[args.task]["model"].from_pretrained(args.pre_trained_model)

    # add special token
    if TASKS[args.task] == 'drrepair_code2align':
        special_tokens_dict = {"additional_special_tokens": ['[unchange]', '[insert]', '[delete]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)

    # get optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
    )

    # load dataset
    # train_ds = TASKS[args.task]["dataset"](tokenizer, split="train")
    # valid_ds = TASKS[args.task]["dataset"](tokenizer, split="val")
    train_ds = TASKS[args.task]["dataset"](tokenizer, split="train", data_len=1000)
    valid_ds = TASKS[args.task]["dataset"](tokenizer, split="val", data_len=10)
    print("train_ds: {}".format(len(train_ds)))
    print("valid_ds: {}".format(len(valid_ds)))

    collate_fn = train_ds.get_collate_fn()

    # sampler
    train_sampler = None
    valid_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_ds
        )

    # create data loader
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=0,
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    test_loader = DataLoader(valid_ds,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0,
                             sampler=valid_sampler,
                             collate_fn=collate_fn)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        last_epoch=-1,
        steps_per_epoch=len(train_loader)*args.gradient_accumulation_steps,
        pct_start=0.1,
        anneal_strategy="linear"
    )

    # wrap model using DDP
    if args.distributed:
        model = DDP(model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank)

    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logging.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))
                if 'epoch' in checkpoint: args.start_epoch = checkpoint['epoch']
                if 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])
                if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler'])

                if args.local_rank == 0 or not args.distributed:
                    if 'best_score' in checkpoint: best_score = checkpoint['best_score']
                logging.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            elif args.resume.lower()=='true':
                args.resume = path_info['ckpt_path']
                resume()
            elif args.resume.lower()=='best':
                args.resume = path_info['best_model_path']
                resume()
            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # save model as huggingface model
    if args.hf_path:
        if args.hf_path == "default":
            args.hf_path = os.path.join(path_info["model_dir"], "hf")
        if args.local_rank == 0 and args.distributed:
            model.module.save_pretrained(args.hf_path)
            logging.info('hf model is saved in {}'.format(args.hf_path))
        elif not args.distributed:
            model.save_pretrained(args.hf_path)
            logging.info('hf model is saved in {}'.format(args.hf_path))
        exit()

    best_score = float('-inf')

    for epoch in range(args.start_epoch, args.epochs):
        # set epoch to train sampler 
        # for different order of examples between epochss
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # training
        train(train_loader, model, tokenizer, optimizer, scheduler, epoch, device, args, summary_logger=summary_logger)

        # evaluation
        scores = validate(test_loader, model, tokenizer, epoch, device, args)

        if args.local_rank == 0 or not args.distributed:
            curr_best = max(
                sum(scores[key].item() for key in TASKS[args.task]["metric_key"])/len(TASKS[args.task]["metric_key"]),
                best_score)

            is_best = curr_best > best_score

            if is_best or epoch==0:
                best_score = curr_best
                best_result = {k: v.item() for k, v in scores.items()}
                with open(os.path.join(path_info["model_dir"], "best_score.json"), "w") as f:
                    json.dump(best_result, f, indent=4)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best,
                path_info["ckpt_path"],
                path_info["best_model_path"])

            # write summary
            summary_logger.add_scalar('eval/loss',
                                    scores['loss'], epoch)
            for key in TASKS[args.task]["metric_key"]:
                summary_logger.add_scalar(f'eval/{key}',
                                    scores[key], epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def train(train_loader, model, tokenizer, optimizer, scheduler, epoch, device, args, summary_logger=None):

    # calc batch time
    batch_time = AverageMeter()
    losses = AverageMeter()

    steps_per_epoch = len(train_loader)

    # switch to train mode
    model.train()
    end = time.time()

    # zero grad
    optimizer.zero_grad()

    for step_inbatch, batch_in in enumerate(train_loader):
        # compute loss

        outputs = model(
            input_ids = batch_in['input']['input_ids'].cuda(),
            attention_mask = batch_in['input']['attention_mask'].cuda(),
            labels = batch_in['target']['input_ids'].cuda()
        )

        loss = outputs.loss

        # backward pass  
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (step_inbatch + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # schedule learning rate
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():   
            losses.update(loss)

        global_step = epoch*steps_per_epoch + step_inbatch
        if global_step % args.print_freq == (args.print_freq - 1):
            with torch.no_grad():
                batch_time.update((time.time() - end)/args.print_freq)
                end = time.time()

                if args.distributed:
                    avg_loss = reduce_tensor(losses.avg, args)
                else:
                    avg_loss = losses.avg

                if args.local_rank == 0 or not args.distributed:

                    if summary_logger is not None:
                        summary_logger.add_scalar('train/loss',
                                        avg_loss, global_step)

                    score_log = "loss\t{:.3f}\n".format(
                        avg_loss.item())

                    logging.info('-----Training----- \nEpoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Speed {3:.3f} ({4:.3f})\t'.format(
                              epoch, step_inbatch, steps_per_epoch,
                              args.batch_size/batch_time.val,
                              args.batch_size/batch_time.avg,
                              batch_time=batch_time)+score_log)


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def reduce_sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def validate(test_loader, model, tokenizer, epoch, device, args):
    steps_per_epoch = len(test_loader)

    # score meters
    batch_time = AverageMeter()
    losses = AverageMeter()

    metric_key = TASKS[args.task]["metric_key"]
    metric_meter = {k:AverageMeter() for k in metric_key}

    pad_token_id=tokenizer.pad_token_id

    # switch to evaluate mode
    model.eval()

    model_ptr = model if not args.distributed else model.module

    with torch.no_grad():
        end = time.time()

        for step_inbatch, batch_in in enumerate(test_loader):
            # compute loss

            outputs = model(
                input_ids = batch_in['input']['input_ids'].cuda(),
                attention_mask = batch_in['input']['attention_mask'].cuda(),
                labels = batch_in['target']['input_ids'].cuda()
            )

            loss = outputs.loss
            losses.update(loss)

            # generate sequence (beam)
            output_seq = model_ptr.generate(
                input_ids=batch_in['input']['input_ids'].cuda(),
                pad_token_id=pad_token_id,
                max_length=512,
                num_beams=args.num_beams,
                early_stopping=True
            )

            # decode best sequence
            inputs = [tokenizer.decode(tseq, skip_special_tokens=True) for tseq in batch_in['input']["input_ids"]]
            targets = [tokenizer.decode(tseq, skip_special_tokens=True) for tseq in batch_in['target']["input_ids"]]
            predictions = [tokenizer.decode(tseq, skip_special_tokens=True) for tseq in output_seq]

            # try:
            #     # calc score & update score
            #     for metric in TASKS[args.task]["metric"]:
            #         score_dict = metric(targets, predictions)
            #         for k, v in score_dict.items():
            #             metric_meter[k].update(torch.tensor(v["score"], device=device), v["count"])

            # except Exception as e:
            #     print(e)
            #     print("targets: {}\n predictions: {}\n".format(targets, predictions))

            for metric in TASKS[args.task]["metric"]:
                score_dict = metric(targets, predictions, inputs)
                for k, v in score_dict.items():
                    metric_meter[k].update(torch.tensor(v["score"], device=device), v["count"])

            # global_steps for validation process
            global_step = epoch*steps_per_epoch + step_inbatch
            if global_step % args.print_freq == (args.print_freq - 1):
                batch_time.update((time.time() - end)/args.print_freq)
                end = time.time()

                # reduce score over gpus
                if args.distributed:
                    avg_loss = reduce_tensor(losses.avg, args)
                    avg_score = {k:reduce_tensor(metric_meter[k].avg, args) for k in metric_key}
                else:
                    avg_loss = losses.avg
                    avg_score = {k:metric_meter[k].avg for k in metric_key}

                # create log for shell
                score_log = "loss\t{:.3f}\t ".format(
                        avg_loss.item())
                for k, v in avg_score.items():
                    score_log += "{}\t{:.3f}\t".format(
                        k,
                        avg_score[k].item(),
                    )
                score_log += "\n"

                # print log
                if args.local_rank == 0 or not args.distributed:
                    logging.info('-----Evaluation----- \nEpoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Speed {3:.3f} ({4:.3f})\t'.format(
                              epoch, step_inbatch, steps_per_epoch,
                              args.batch_size/batch_time.val,
                              args.batch_size/batch_time.avg,
                              batch_time=batch_time)+score_log)

        # reduce scores
        if args.distributed:
            loss_sum, loss_cnt = reduce_sum_tensor(losses.sum), reduce_sum_tensor(torch.tensor(losses.count, device=device))
            score_sum = {k:reduce_sum_tensor(metric_meter[k].sum) for k in metric_key}
            score_cnt = {k:reduce_sum_tensor(torch.tensor(metric_meter[k].count, device=device)) for k in metric_key}
        else:
            loss_sum, loss_cnt = losses.sum, torch.tensor(losses.count, device=device)
            score_sum = {k:metric_meter[k].sum for k in metric_key}
            score_cnt = {k:torch.tensor(metric_meter[k].count, device=device) for k in metric_key}

        # create score dict
        result_dict = {
            "loss": loss_sum/loss_cnt
        }
        for k in metric_key:
            result_dict[k] = score_sum[k]/score_cnt[k]

        # create log
        score_log = "loss\t{:.3f}\t ".format(
                result_dict['loss'].item(),
            )
        for k, v in result_dict.items():
            score_log += "{}\t{:.3f}\t".format(
                k,
                result_dict[k].item(),
            )
        score_log += "\n"

        # print log
        if args.local_rank == 0 or not args.distributed:
            logging.info('-----Evaluation----- \n'+score_log)

        return result_dict

if __name__ == "__main__":
    main()

# Using Multi-GPU Training
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=8 train.py

# Resume & Huggingface model save
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=8 train.py --resume [true or best] --hf_path default
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=8 train.py --resume true --hf_path default
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=3 train.py --resume true --hf_path default