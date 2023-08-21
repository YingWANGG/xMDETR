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
from datasets import load_dataset, load_metric, Dataset, load_from_disk
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
    parser.add_argument("--mlm_dir", type=str, help="MLM language data directory"
    )
    parser.add_argument("--lang", type=str, help="Language code"
    )
    parser.add_argument("--code_switch_path", type=str, help="path to code switch dic"
    )
    parser.add_argument("--code_switch_ratio", default=1.0, type=float, help="ratio of code switch"
    )
    parser.add_argument("--max_steps", type=int, help="total steps"
    )
    parser.add_argument("--warmup_steps", type=int, help="number of warmup steps"
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

    # Load code switch dictionary
    code_switch_dict = defaultdict(list)
    code_switch_dict_re = defaultdict(list)
    with open(args.code_switch_path) as f:
        for line in f:
            k, v = line.split()[0], " ".join(line.split()[1:])
            code_switch_dict[k].append(v.lower())
            code_switch_dict_re[v].append(k.lower())
    print("Loading bilingual dictionary:", len(code_switch_dict))

    # Build the model
    tokenizer =  AutoTokenizer.from_pretrained(args.text_tokenizer_type)
    print("Loading pretrained weights ...")
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)
    
      # Insert adapter
    adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=args.reduction_factor)
    model.transformer.text_encoder.add_adapter("lm_adapter", config=adapter_config)
    model.transformer.text_encoder.set_active_adapters("lm_adapter")

    # Load weights
    checkpoint = torch.hub.load_state_dict_from_url(args.load)["model_ema"] if args.device == 'gpu' \
                                else torch.hub.load_state_dict_from_url(args.load, map_location=torch.device('cpu'))["model_ema"]
    model_dict = model.state_dict()
    checkpoint_sub = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == checkpoint[k].shape}
    model_dict.update(checkpoint_sub)
    print("Initialization with Lexical Overlap ...")
    # copy word embedding for shared tokens
    en_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    transformed_en_vocab = set(en_tokenizer.vocab.keys()) #set([x.replace('Ġ', '▁') for x in en_tokenizer.vocab.keys()])
    shared_tokens = set(tokenizer.vocab.keys()).intersection(transformed_en_vocab)
    print(f"XLMR has {len(shared_tokens)} same tokens as the English vocabulary")
    print("Copying word embeddings of overlapping tokens")
    count = 0
    for token in list(shared_tokens):
        count += 1
        en_idx = en_tokenizer.vocab[token] #en_tokenizer.vocab[token.replace('▁', 'Ġ')]
        new_idx = tokenizer.vocab[token]
        model_dict['transformer.text_encoder.embeddings.word_embeddings.weight'][new_idx] = checkpoint['transformer.text_encoder.embeddings.word_embeddings.weight'][en_idx]
    print(f"count: {count}")

    # copy word embedding for target language tokens
    # optional
    #transformed_vocab =  set([x.strip('▁') for x in tokenizer.vocab.keys()])
    #shared_tokens_target = transformed_vocab.intersection(set(code_switch_dict_re.keys()))
    #print(f"XLMR has {len(shared_tokens_target)} same tokens as the {args.lang} vocabulary")
    #print("Copying word embeddings of overlapping tokens")
    #en_count = 0
    #for token in list(shared_tokens_target):
    #    trans_list = code_switch_dict_re[token]
    #    em_list = []
    #    for en_token in trans_list:
    #        if en_token in en_tokenizer.vocab:
    #            en_idx = en_tokenizer.vocab[en_token]
    #            em_list.append(checkpoint['transformer.text_encoder.embeddings.word_embeddings.weight'][en_idx])
    #        if 'Ġ'+en_token in en_tokenizer.vocab:
    #            en_idx = en_tokenizer.vocab['Ġ'+en_token]
    #            em_list.append(checkpoint['transformer.text_encoder.embeddings.word_embeddings.weight'][en_idx])
    #    if len(em_list) > 0 :
    #        en_count += 1
    #        if token in tokenizer.vocab:
    #            new_idx = tokenizer.vocab[token]
    #            model_dict['transformer.text_encoder.embeddings.word_embeddings.weight'][new_idx] = sum(em_list)/len(em_list)
    #        if '_'+token in tokenizer.vocab:
    #            new_idx = tokenizer.vocab['_'+token]
    #            model_dict['transformer.text_encoder.embeddings.word_embeddings.weight'][new_idx] = sum(em_list)/len(em_list)
    #print(f"Found {en_count} words overlaping words in roberta-base") 
    #model.load_state_dict(model_dict)

    # Wrap models by DDP
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)# broadcast_buffers=False
    
    # Freeze weights
    for param in model.named_parameters():
        if "embeddings" not in param[0] and "lm_head" not in param[0] and "adapters" not in param[0]:
            param[1].requires_grad = False
        else:
            print("Gradients: ", param[0])
         
    # Optimizer
    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr) 
    
    # QGA - Dataset
    dataset_train = ConcatDataset([build_dataset(name, image_set="train", args=args) for name in args.combine_datasets])
    gqa_size = len(dataset_train)
    gqa_num_epoch = gqa_size //10000 + 1
    chunks = torch.chunk(torch.arange(len(dataset_train)), gqa_num_epoch)
    datasets = [torch.utils.data.Subset(dataset_train, chunk.tolist()) for chunk in chunks]
    samplers_train = [DistributedSampler(ds) for ds in datasets] if args.distributed else [torch.utils.data.RandomSampler(ds) for ds in datasets]
    
    batch_samplers_train = [
                torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
                for sampler_train in samplers_train
         ]
    assert len(batch_samplers_train) == len(datasets)
    data_loaders_train = [
                DataLoader(
                    ds,
                    batch_sampler=batch_sampler_train,
                    collate_fn=partial(utils.collate_fn, False),
                    num_workers=args.num_workers,
                )
                for ds, batch_sampler_train in zip(datasets, batch_samplers_train)
            ]
    
    # OSCAR - DATASET    
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    raw_datasets = load_from_disk(args.mlm_dir)
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=False,
                truncation=True,
                max_length=args.max_seq_length,
                return_special_tokens_mask=True,
            )

    with dist.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets['train'].map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
            )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )
    
    mlm_trainset = tokenized_datasets
    mlm_size = len(mlm_trainset)
    mlm_num_epoch = mlm_size //10000 + 1
    mlm_chunks = torch.chunk(torch.arange(mlm_size), mlm_num_epoch)
    mlm_datasets = [torch.utils.data.Subset(mlm_trainset, chunk.tolist()) for chunk in mlm_chunks]
    mlm_samplers = [DistributedSampler(ds) for ds in mlm_datasets] if args.distributed else [torch.utils.data.RandomSampler(ds) for ds in mlm_datasets]
    mlm_batch_samplers = [
        torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True) 
        for sampler_train in mlm_samplers
    ]
    assert len(mlm_samplers) == len(mlm_datasets)
    mlm_data_loaders = [
        DataLoader(
            ds,
            batch_sampler=batch_sampler_train,
            collate_fn=data_collator,
            num_workers=args.num_workers,
        )
        for ds, batch_sampler_train in zip(mlm_datasets, mlm_batch_samplers)
    ]
    
    # Val dataset (xGQA)
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
    curr_gqa_iter_idx = 0
    gqa_iter = iter(data_loaders_train[curr_gqa_iter_idx])
    gqa_epoch = 0
    print("GQA dataset loaded:", gqa_size, gqa_num_epoch)
    
    curr_mlm_iter_idx = 0
    mlm_iter = iter(mlm_data_loaders[curr_mlm_iter_idx])
    mlm_epoch = 0
    print("MLM dataset loaded", mlm_size, mlm_num_epoch)
    
    # code switch
    def code_switch(s):
        #print(s)
        s = s.lower().strip('?').split()
        n = len(s)
        num_replaced =int(n * args.code_switch_ratio)
        for idx in random.sample(range(n), k=n):
            target = s[idx].strip(",").strip("'").replace("'s","")
            if num_replaced == 0:
                break
            if target in code_switch_dict:
                s[idx] = random.sample(code_switch_dict[target], k=1)[0]
                num_replaced -= 1   
        s = ' '.join(s)+'?'
        #print(s)
        return s
    
    print("Training begins")
    print("GQA epoch - 0")
    print("MLM epoch - 0")
    
    t = time.time()
    model.train()
    criterion.train()
    qa_criterion.train()

    best_metric = 0
    
    for i in range(0,args.max_steps+1,2):
        model.train()
        # mdetr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation):
            try:
                batch_dict = next(gqa_iter)
            except StopIteration:
                if curr_gqa_iter_idx != gqa_num_epoch-1:
                    curr_gqa_iter_idx += 1
                else:
                    gqa_epoch += 1 #new epoch
                    print(f"GQA epoch - {gqa_epoch}")
                    curr_gqa_iter_idx = 0
                    samplers_train[curr_gqa_iter_idx].set_epoch(gqa_epoch)
                gqa_iter = iter(data_loaders_train[curr_gqa_iter_idx])
                batch_dict = next(gqa_iter)
            
            samples = batch_dict["samples"].to(device)
            positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
            targets = batch_dict["targets"]
            answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
            captions = [code_switch(t["caption"], 0.3+0.7*i/(args.max_steps+1)) for t in targets]
            if i%10000==0: print(captions)
            targets = targets_to(targets, device)
            outputs = model(samples, captions, forward_type="gqa")#, encode_and_save=False, memory_cache=memory_cache)
            loss_dict = {}
            loss_dict.update(criterion(outputs, targets, positive_map))
            loss_dict.update(qa_criterion(outputs, answers))
            qa_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            qa_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        
        # MLM
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation):
            try:
                mlm_batch = next(mlm_iter)
            except StopIteration:
                if curr_mlm_iter_idx != mlm_num_epoch-1:
                    curr_mlm_iter_idx += 1
                else:
                    mlm_epoch += 1 #new epoch
                    print(f"MLM epoch - {mlm_epoch}")
                    curr_mlm_iter_idx = 0
                    mlm_samplers[curr_mlm_iter_idx].set_epoch(mlm_epoch)
                mlm_iter = iter(mlm_data_loaders[curr_mlm_iter_idx])
                mlm_batch = next(mlm_iter)
            batch = {k: v.to(device) for k, v in mlm_batch.items()}
            output = model(forward_type="mlm", **batch)
            mlm_loss = output.loss
            mlm_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i%500==0:   
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = dist.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            qa_loss_avg = losses_reduced_scaled.item()
            mlm_loss_avg = dist.reduce(mlm_loss)
            

        if i%2000==0:
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
            
            print(f"{i} - accuracy:{metric}")
            
        
            
            if metric > best_metric:
                best_metric = metric
                dist.save_on_master(
                                {
                                    "model": model.module.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "gqa_epoch": gqa_epoch,
                                    "mlm_epoch": mlm_epoch,
                                    "args": args,
                                },
                                f"{args.output_dir}/{args.lang}_{args.n_shot}shot"
                            )
                print(f"Best accuracy so far: {metric}; GQA epoch - {gqa_epoch}, MLM epoch - {mlm_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("XGQA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    
    
