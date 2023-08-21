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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args_parser():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser(
        "Evaluate XGQA", parents=[detection_parser], add_help=False
    )
    parser.add_argument("--lang", type=str, help="Language code"
    )
    return parser  

def main(args):
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)
    print(args)

    device = torch.device(args.device)
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)
    
    adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=16)
    model.transformer.text_encoder.add_adapter("lm_adapter", config=adapter_config)
    model.transformer.text_encoder.set_active_adapters("lm_adapter")

    checkpoint = torch.load(args.load, map_location=device)["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.to(args.device)
    model.eval();
    
    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])

    val_tuples = []
    for dset_name in args.combine_datasets_val:
        dset = build_dataset(dset_name, image_set="fewshot_test", args=args)
        sampler = (
            torch.utils.data.SequentialSampler(dset)
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
                contrastive_criterion=contrastive_criterion,
                qa_criterion=qa_criterion,
                postprocessors=postprocessors,
                weight_dict=weight_dict,
                data_loader=item.dataloader,
                evaluator_list=item.evaluator_list,
                device=device,
                args=args,
                print_freq=500
        )
        test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})   
    #print(test_stats)
    metric = test_stats["gqa_accuracy_answer_total_unscaled"]
    print("-----------------------------------------------------------------------")
    print("|", args.load)
    print("|", metric, flush=True)
    print("-----------------------------------------------------------------------")
if __name__ == "__main__":
    parser = argparse.ArgumentParser("XGQA evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
