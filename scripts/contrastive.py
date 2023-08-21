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
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from itertools import chain

import datasets
from datasets import load_dataset, load_metric, Dataset, load_from_disk
import transformers
from transformers import (
    AutoTokenizer,
    get_scheduler,
    AdapterConfig
)

sys.path.append("mdetr")
import main as detection
import util.dist as dist
import util.misc as utils
from dataset import build_dataset, get_coco_api_from_dataset
from dataset.contrastive_dataset import ImageMultiTextDataset
from engine import evaluate
from models import build_model
from util.metrics import MetricLogger, accuracy
from util.misc import targets_to
from dataset.coco_eval import CocoEvaluator
from models.postprocessors import build_postprocessors
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_DATASETS_CACHE']="cache"



def get_args_parser():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser(
        "Train XGQA", parents=[detection_parser], add_help=False
    )
    parser.add_argument("--contrastive_path", type=str, help="path to contrastive annotation data"
    )
    parser.add_argument("--lang", type=str, help="Language code"
    )
    parser.add_argument("--max_steps", type=int, help="total steps"
    )
    parser.add_argument("--warmup_steps", type=int, help="number of warmup steps"
    )
    parser.add_argument("--distributed", action="store_true", help="whether or not use mutltiple gpu"
    )
    parser.add_argument("--reduction_factor", type=int, help="reduction factor for the lm adapter"
    )
    return parser


def main(args):
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
    print("Loading pretrained weights ...")
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
        if "embeddings" not in param[0] and "lm_head" not in param[0] and "adapters" not in param[0] and "contrastive" not in param[0]:
            param[1].requires_grad = False
        else:
            print("Gradients: ", param[0])
         
    # Optimizer
    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr) 

    # QGA - Dataset
    image_text_dataset = ImageMultiTextDataset(args.contrastive_path, batch_size=args.batch_size,
                                                    tokenizer=tokenizer)

    image_text_loader = torch.utils.data.DataLoader(image_text_dataset, batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=image_text_dataset.collate_fn)
    multik_size = len(image_text_loader)
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
    lr_scheduler = get_scheduler(name="constant", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    
    #Training 
    print("Training begins")
    
    t = time.time()
    
    for k in range(10):
        print(f"epoch: {k}")
        for i, batch in enumerate(image_text_loader):
            model.train()    
            optimizer.zero_grad()
            batch = [k.to(device) for k in batch]
            images, text1, text2 = batch[0], batch[1:3], batch[3:5]
            text_out1, text_out2, image_out = model(forward_type="contrastive", samples=images, captions=[text1, text2])
            itc_loss1 = contrastive_criterion(text_out1, image_out)
            itc_loss2 = contrastive_criterion(text_out2, image_out)
            itc_loss = (itc_loss1 + itc_loss2)/2
            itc_loss = contrastive_criterion(text_out1, text_out2)
            loss = itc_loss + itc_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if i%500==0:   
                # reduce losses over all GPUs for logging purposes
                loss_avg = dist.reduce(loss)

                # evaluate
                print(f"Evaluating........")
                model.eval()
                test_stats = {}
                for j, item in enumerate(val_tuples):
                    evaluator_list = []
                    iou_types = ["bbox"]
                    evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
                    item = item._replace(evaluator_list=evaluator_list)
                    postprocessors = build_postprocessors(args, item.dataset_name)
                    print(f"Evaluating {item.dataset_name}")
                    curr_test_stats = evaluate(
                        model=model,
                        criterion=criterion,
                        contrastive_criterion=None, #temp
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
                
                print(f"{i} - accuracy:{metric}")
        
    print("Training ends")
    dist.save_on_master(
                    {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": args,
                    },
                    f"{args.output_dir}/{args.lang}_0shot_contrastive"
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("XGQA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    
    
