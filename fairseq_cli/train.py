#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import ast
import os
import random
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch
import torch.distributed as dist

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed_utils import is_master
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



def main(cfg: FairseqConfig) -> None:
    ## IMP model architecture values are set here in te convert_namespace function. $$IMP
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    
    utils.import_user_module(cfg.common)

    if is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)


    # import ipdb; ipdb.set_trace()    
    # Print args
    logger.info(cfg)
    cfg.common.all_gather_list_size = 655360

    ## IMP
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    assert cfg.criterion, "Please specify criterion to train a model"

    ## IMP
    # Build model and criterion
    model = task.build_model(cfg.model)
    
    ## IMP
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
   

    logger.info(
        "num. shared model params: {} (num. trainable: {})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) ),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad),
        )
    )
    
    logger.info(
        "num. expert model params: {} (num. trainable: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) ),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )
    
    

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    
    # Build trainer
    if cfg.common.model_parallel_size == 1:
        # Path: dense
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per GPU = {} and batch size per GPU = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )
    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    # epoch itr here contains a dataset iterator which will further be converted into a data loader.
    # checkpoint_path = "/usr/project/xtmp/rt195/DEMIX/PT_Models/dense_4_GPUs_transformer_lm_gpt3_small_0812_IL_4_128_128_8_40k/checkpoint_3_40000.pt"
    # checkpoint_path = "/usr/project/xtmp/rt195/DEMIX/PT_Models/dense_4_GPUs_transformer_lm_gpt3_small_0912_IL_4_128_128_8_10k/checkpoint_1_10000.pt"
    # checkpoint_path = "../PT_Models/unbalanced_dense_4_GPUs_transformer_lm_gpt3_small_1001_IL_4_128_64_16_14k/checkpoint_4_14000.pt"
    # checkpoint_path = "../PT_Models/unbalanced_dense_4_GPUs_transformer_lm_gpt3_small_0802_ILbaseline_combined_4_128_16_64_12k_8e-4_0.1/checkpoint_best.pt"
    # checkpoint_path = "../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_1802_ILbaseline_combined_8_128_16_32_30k_8e-4_0.1/checkpoint_3_30000.pt"
    # checkpoint_path = "../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_1902_ILbaseline_combined_8_128_16_32_50k_8e-4_0.1/checkpoint_5_50000.pt"
    # checkpoint_path = "../PT_Models/unbalanced_dense_4_GPUs_transformer_lm_gpt3_small_2002_ILbaseline_combined_4_128_16_64_75k_8e-4_0.1/checkpoint_7_75000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_2402_ILbaseline_combined_8_128_16_32_100k_8e-4_0.1/checkpoint_last.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0301_ILbaseline_combined9010_8_128_16_32_10k_8e-4_0.1/checkpoint_3_10000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0301_ILbaseline_combined9010_8_128_16_32_50k_8e-4_0.1/checkpoint_14_50000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_4_GPUs_transformer_lm_gpt3_small_0303_ILbaseline_combined9010_4_128_16_64_75k_8e-4_0.1/checkpoint_21_75000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0313_ILbaseline_combined_8_128_16_32_100k_64e-4_0.1/checkpoint_1_10000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0313_ILbaseline_combined_8_128_16_32_100k_64e-4_0.1/checkpoint_last.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0313_ILbaseline_combined_8_128_16_32_100k_64e-4_0.1/checkpoint_5_50000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0313_ILbaseline_combined_8_128_16_32_100k_64e-4_0.1/checkpoint_3_30000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0313_ILbaseline_combined_8_128_16_32_100k_64e-4_0.1/checkpoint_7_65000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0313_ILbaseline_combined_8_128_16_32_100k_64e-4_0.1/checkpoint_8_85000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_4_GPUs_transformer_lm_gpt3_small_2002_ILbaseline_combined_4_128_16_64_75k_8e-4_0.1/checkpoint_5_50000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_4_GPUs_transformer_lm_gpt3_small_2002_ILbaseline_combined_4_128_16_64_75k_8e-4_0.1/checkpoint_3_30000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0322_ILbaseline_combined_8_128_16_32_75k_64e-4_0.1/checkpoint_7_75000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0322_ILbaseline_combined_8_128_16_32_75k_64e-4_0.1/checkpoint_5_50000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0322_ILbaseline_combined_8_128_16_32_75k_64e-4_0.1/checkpoint_3_30000.pt"
    # checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0322_ILbaseline_combined_8_128_16_32_75k_64e-4_0.1/checkpoint_6_60000.pt"
    checkpoint_path="../PT_Models/unbalanced_dense_8_GPUs_transformer_lm_gpt3_small_0322_ILbaseline_combined_8_128_16_32_75k_64e-4_0.1/checkpoint_4_40000.pt"
    
    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        reset_optimizer=True,
        reset_lr_scheduler=True,
        optimizer_overrides=True,
        reset_meters=True,
    )

    if getattr(cfg.model, "adaptation", False):
        for x,p in model.named_parameters():
            if hasattr(p, "expert"):
                delattr(p, "expert")
                delattr(p, "process_group")

    valid_subsets = cfg.dataset.valid_subset.split(",")
    end_of_epoch=True
    valid_losses = validate(cfg, trainer, task, None, valid_subsets)
    print(valid_losses)



def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False




def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(1)
    valid_losses = []
    subset = subsets[0]

    logger.info('begin validation on "{}" subset on rank {}'.format(
        subset, 0))

    # Initialize data iterator
    itr = trainer.get_valid_iterator(subset).next_epoch_itr(
        shuffle=False, set_dataset_epoch=False  # use a fixed valid set
    )

    # create a new root metrics aggregator so validation metrics
    # don't pollute other aggregators (e.g., train meters)
    kkk=0
    loss_array = []
    examples_array = []
    count=0
    


    with metrics.aggregate(new_root=True) as agg:
        for idx2, sample in enumerate(itr):
            if(kkk<100 or kkk%1000==0):
                print(f"### {kkk} of {len(itr)}", flush=True)
            kkk+=1

            if(sample is not None):
                # continue;
                # print(">", flush=True)
                stats = trainer.valid_step(sample)
                assert torch.sum(sample['net_input']['src_lengths'])==stats['ntokens']

            # print(">>", idx2, torch.distributed.get_rank(), flush=True)    
            torch.cuda.synchronize()
                
            if(sample is not None):
                len_list = [torch.tensor(0).cuda() for each in range(cfg.distributed_training.distributed_world_size)]
                nsentences = torch.tensor(sample['nsentences'])
                torch.distributed.all_gather(len_list, nsentences.cuda())    
                max_size = max(len_list).item()
                size_diff = max_size - nsentences
                if size_diff:
                    padding_loss = torch.zeros(size_diff*sample['net_input']['src_tokens'].shape[1], dtype=torch.float).cuda()
                    stats['loss'] = torch.cat((stats['loss'], padding_loss))

                    padding_input_ids = torch.zeros(size_diff, sample['net_input']['src_tokens'].shape[1], dtype=torch.long)
                    sample['net_input']['src_tokens'] = torch.cat((sample['net_input']['src_tokens'], padding_input_ids))

                
                if(torch.distributed.get_rank()==0):

                    loss_list = [torch.zeros_like(stats['loss'], dtype=torch.float).cuda() for  _ in  range(cfg.distributed_training.distributed_world_size)]
                    torch.distributed.gather(stats['loss'], loss_list)
                
                    input_list = [torch.zeros_like(sample['net_input']['src_tokens'], dtype=torch.long).cuda() for _ in  range(cfg.distributed_training.distributed_world_size)]
                    torch.distributed.gather(sample['net_input']['src_tokens'].cuda(), input_list)

                    if(max(len_list)!=min(len_list)):
                        loss_list = [each_val[:nsentences*sample['net_input']['src_tokens'].shape[1]] for nsentences, each_val in zip(len_list, loss_list)]
                        input_list = [each_val[:nsentences] for nsentences, each_val in zip(len_list, input_list)]

                    loss_array.append(torch.cat(loss_list).detach().cpu())
                    examples_array.append(torch.cat(input_list, 0).detach().cpu())
                else:
                    torch.distributed.gather(stats['loss'], None)
                    torch.distributed.gather(sample['net_input']['src_tokens'].cuda(), None)
            else:
                nsentences = torch.tensor(0)
                torch.distributed.all_gather(len_list, nsentences.cuda())  

                max_size = max(len_list).item()
                size_diff = max_size - nsentences

                if size_diff:
                    padding_loss = torch.zeros(size_diff*cfg.dataset.batch_size_valid, dtype=torch.float).cuda()
                    padding_input_ids = torch.zeros(size_diff, cfg.task.tokens_per_sample, dtype=torch.long).cuda()

                torch.distributed.gather(padding_loss, None)
                torch.distributed.gather(padding_input_ids, None)


    torch.cuda.synchronize()     
    
    if(torch.distributed.get_rank()==0):
        print("Concat", flush=True)
        all_examples = torch.cat(examples_array).view(-1, sample['net_input']['src_tokens'].shape[1])
        all_losses = torch.cat(loss_array).view(-1, sample['net_input']['src_tokens'].shape[1])
        print("Concat done", flush=True)

        try:
            torch.save(all_losses, cfg.task.data + cfg.task.eval_domains + "/IRL_losses_unrolled.pt" )
            torch.save(all_examples, cfg.task.data + cfg.task.eval_domains + "/IRL_inputs_unrolled.pt" )
        except:
            import ipdb; ipdb.set_trace()
        
        print("Save done", flush=True)


        # !something wierd between domain_datasets and multi corpus data. we need to left shift once
        print("Roll", flush=True)
        all_examples = roll_0(all_examples, 1)
        all_losses = roll_0(all_losses, 1)
        print("Roll done", flush=True)

        print("Save done", flush=True)


        assert len(all_examples)==len(trainer.task.domain_datasets[subset][0])
        torch.save(all_losses, cfg.task.data + cfg.task.eval_domains + "/IRL_losses.pt" )
        torch.save(all_examples, cfg.task.data + cfg.task.eval_domains + "/IRL_inputs.pt" )
    
        try:
            for idx1 in range(len(all_examples)):
                assert (all_examples[idx1]==trainer.task.domain_datasets[subset][0][idx1]['source']).all()
        except:
            assert idx1 == len(all_examples)-1
            assert (all_examples[idx1][:len(trainer.task.domain_datasets[subset][0][idx1]['source'])] == trainer.task.domain_datasets[subset][0][idx1]['source']).all()

        all_examples = all_examples[:,0:5]
        torch.save(all_examples, cfg.task.data + cfg.task.eval_domains + "/IRL_inputs_05.pt" )

        print("Save done", flush=True)

        # log validation stats
        # print(stats['loss'].shape)        
        
    return 0

def roll_0(x, n):  
    return torch.cat((x[n:], x[:n]), dim=0)

def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    # stats["num_updates"] = trainer.get_num_updates()
    # if hasattr(checkpoint_utils.save_checkpoint, "best"):
    #     key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
    #     best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
    #     stats[key] = best_function(
    #         checkpoint_utils.save_checkpoint.best,
    #         stats[cfg.checkpoint.best_checkpoint_metric],
    #     )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    # srun --label already creates ntasks number of parallel runs. ipdb will show all of them. Hence ntasks in salloc is very imp.
    # Path: dense

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                # Path: dense
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # main(cfg)

if __name__ == "__main__":
    cli_main()
