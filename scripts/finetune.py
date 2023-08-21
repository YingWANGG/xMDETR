# This code is based on https://github.com/ashkamath/mdetr
import argparse
import json
import os
import random
import sys
from collections import namedtuple, defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List
import datetime
import time

import numpy as np
import torch
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
torch.cuda.empty_cache()
from itertools import chain

import datasets
from datasets import load_dataset, load_metric, Dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    get_scheduler,
    AdapterConfig
)

sys.path.append("mdetr")
import main as detection
import util.dist as dist
import util.misc as utils
from dataset import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
from util.metrics import MetricLogger, accuracy
from util.misc import targets_to
from dataset.coco_eval import CocoEvaluator
from models.postprocessors import build_postprocessors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args_parser():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser(
        "Train XGQA", parents=[detection_parser], add_help=False
    )
    parser.add_argument("--max_seq_length", type=int, default=None,
        help = "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
    )
    parser.add_argument("--preprocessing_num_workers", type=int,
        default=None,
        help = "The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--lang", type=str, help="Language code"
    )
    parser.add_argument("--gradient_accumulation", type=int, help="number of warmup steps"
    )
    parser.add_argument("--distributed", action="store_true", help="whether or not use mutltiple gpu"
    )
    parser.add_argument("--reduction_factor", type=int, help="reduction factor for the lm adapter"
    )
    parser.add_argument("--n_shot", type=int, help="number of images used in the training"
    )
    return parser


def main(args):

    # Init distributed mode
    if args.distributed:
        dist.init_distributed_mode(args)  

    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    print(args)

    device = torch.device(args.device)
    rank = dist.get_rank() if args.distributed else 0
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build the model
    tokenizer =  AutoTokenizer.from_pretrained(args.text_tokenizer_type)
    #print("Loading pretrained weights ...")
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)
    
    # Insert adapter
    adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=args.reduction_factor)
    model.transformer.text_encoder.add_adapter("lm_adapter", config=adapter_config)
    model.transformer.text_encoder.set_active_adapters("lm_adapter")

    # Load weights
    checkpoint = torch.load(args.load, map_location=device)["model"]
    model.load_state_dict(checkpoint, strict=False)
    
    # Wrap models by DDP
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)# broadcast_buffers=False
    
    # Freeze weights
    for param in model.named_parameters():
        if "embeddings" not in param[0] and "adapters" not in param[0]:
            param[1].requires_grad = False
        else:
            print("Required Gradients: ", param[0])
            
         
    # Optimizer
    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr) 
    
    # QGA - Dataset
    dataset_train = ConcatDataset([build_dataset(name, image_set="fewshot_train", args=args) for name in args.combine_datasets])
    sampler_train = DistributedSampler(dataset_train) if args.distributed else torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(
                    dataset_train,
                    batch_sampler=batch_sampler_train,
                    collate_fn=partial(utils.collate_fn, False),
                    num_workers=args.num_workers,
                )
                
    # Val dataset
    if len(args.combine_datasets_val) == 0:
        raise RuntimeError("Please provide at least one validation dataset")

    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])

    val_tuples = []
    for dset_name in args.combine_datasets_val:
        dset = build_dataset(dset_name, image_set="fewshot_dev", args=args)
        sampler = (
            DistributedSampler(dset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dset)
        )
        dataloader = DataLoader(
            dset,
            args.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(utils.collate_fn, False),
            num_workers=args.num_workers,
        )
        base_ds = get_coco_api_from_dataset(dset)
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=dataloader, base_ds=base_ds, evaluator_list=None))
    
    # learning rate scheduler
    lr_scheduler = get_scheduler(name="constant", optimizer=optimizer)
    
    print("Training begins")
    
    t = time.time()
    criterion.train()
    qa_criterion.train()

    best_metric = 0
    
    for i in range(0, args.epochs):
        model.train()
        for batch_dict in data_loader_train:
            optimizer.zero_grad(set_to_none=True)
            samples = batch_dict["samples"].to(device)
            positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
            targets = batch_dict["targets"]
            answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
            captions = [t["caption"] for t in targets]
            targets = targets_to(targets, device)
            outputs = model(samples, captions, forward_type="gqa")
            loss_dict = {}
            loss_dict.update(criterion(outputs, targets, positive_map))
            loss_dict.update(qa_criterion(outputs, answers))
            qa_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            qa_loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        qa_loss_avg = losses_reduced_scaled.item()

        print(f"Evaluating........")
        model.eval()
        test_stats = {}
        for j, item in enumerate(val_tuples):
            evaluator_list = []
            iou_types = ["bbox"]
            evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
            item = item._replace(evaluator_list=evaluator_list)
            postprocessors = build_postprocessors(args, item.dataset_name)
            curr_test_stats = evaluate(
                model=model,
                criterion=criterion,
                contrastive_criterion=contrastive_criterion,
                qa_criterion=qa_criterion,
                postprocessors=postprocessors,
                weight_dict=weight_dict,
                data_loader=item.dataloader,
                evaluator_list=item.evaluator_list,
                device=device,
                args=args,
                print_freq=1000
            )
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})
            
        metric = test_stats["gqa_accuracy_answer_total_unscaled"]
        
        if metric > best_metric:
            best_metric = metric
            dist.save_on_master(
                    {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": args,
                    },
                    f"{args.output_dir}/{args.lang}_{args.n_shot}shot_final"
                )

        
    print("Training ends")
    print(f"{args.n_shot}: {best_metric}", flush=True)
    print("----------------------------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("XGQA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    
    
