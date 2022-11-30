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

    # Print args
    logger.info(cfg)

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

    if torch.distributed.is_initialized() and getattr(cfg.model, "desynchronize", False):
        
        
        if getattr(cfg.model, "sync_type") == "none":
            groups = [[x] for x in range(torch.distributed.get_world_size())]
            
        elif getattr(cfg.model, "sync_type") == "manual":
            gps = getattr(cfg.model, "data_parallel_groups")
            gps = gps.split()
            gps = [gp.split(',') for gp in gps]
            groups = [[int(x) for x in y] for y in gps]

        process_groups = {}
        for group in groups:
            distributed_group = torch.distributed.new_group(group)
            for item in group:
                process_groups[item] = distributed_group


        logger.info(f"Data Parallel Groups: {groups}")
        process_group = process_groups[torch.distributed.get_rank(torch.distributed.group.WORLD)]
        
        if getattr(cfg.model, "untie_parameters"):
            for x, p in model.named_parameters():

                if getattr(cfg.model, "untie_parameters") == "transformer_l2":
                    every_other_layer = [f"layers.{z}" for z in range(0, getattr(cfg.model, "decoder_layers"), 2)]
                    if any(l in x for l in every_other_layer):
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "expert_layers":
                    if "expert_layers" in x:
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "expert_ffn":
                    ffns = ['expert_fc1', 'expert_fc2', 'gate']
                    if any(ffn in x for ffn in ffns):
                        p.expert = True
                        p.process_group = process_group


                elif getattr(cfg.model, "untie_parameters") == "transformer_output":
                    if "layers" in x or 'output_projection' in x:
                        p.expert = True
                        p.process_group = process_group
                
                elif getattr(cfg.model, "untie_parameters") == "input_output":
                    if "embed_tokens" in x or "embed_positions" in x or 'output_projection' in x:
                        p.expert = True
                        p.process_group = process_group
                
                elif getattr(cfg.model, "untie_parameters") == "transformer":
                    if "layers" in x:
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "transformer_l2":
                    every_other_layer = [f"layers.{z}" for z in range(0, getattr(cfg.model, "decoder_layers"), 2)]
                    if any(l in x for l in every_other_layer):
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "feedforward":
                    ffns = ['fc1', 'fc2']
                    if any(ffn in x for ffn in ffns) and 'expert' not in x:
                        p.expert = True
                        p.process_group = process_group
                
                elif getattr(cfg.model, "untie_parameters") == "feedforward_l2":
                    every_other_layer = [f"layers.{z}" for z in range(0, getattr(cfg.model, "decoder_layers"), 2)]
                    ffns = ['fc1', 'fc2']
                    if any(ffn in x for ffn in ffns) and any(l in x for l in every_other_layer):
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "feedforward_top":
                    top_layer = list(range(0, getattr(cfg.model, "decoder_layers")))[-1]
                    top_layer = [f"layers.{top_layer}"]
                    ffns = ['fc1', 'fc2']
                    if any(ffn in x for ffn in ffns) and any(l in x for l in top_layer):
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "all":
                    p.expert = True
                    p.process_group = process_group
                else:
                    raise Exception("value for untie_parameters is bad.")
    

    if getattr(cfg.model, "adaptation", False):
    #     # if not getattr(cfg.model, "untie_parameters"):
    #     #     for layer in model.decoder.layers:
    #     #         layer.add_adapter(1)
    #     #     for x,p in model.named_parameters():
    #     #         if not 'adapter' in x:
    #     #             p.requires_grad = False
         ffns = ['fc1', 'fc2']
         for x,p in model.named_parameters():
            if not any(ffn in x for ffn in ffns):
                 p.requires_grad = False
            else:
                 p.requires_grad = True
    
    

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
    checkpoint_path = "/usr/project/xtmp/rt195/DEMIX/PT_Models/dense_4_GPUs_transformer_lm_gpt3_small_IL/checkpoint_last.pt"
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

            # print(">", flush=True)
            stats = trainer.valid_step(sample)
            loss_array.append(stats['loss'].detach().cpu())
            
            examples_array.append(torch.clone(sample['net_input']['src_tokens'].reshape(-1,1024)))
            # if(kkk>5):
            #     break;   
    all_examples = torch.cat(examples_array).view(-1, 1024)
    all_losses = torch.cat(loss_array).view(-1, 1024)
    
    assert len(all_examples)==len(trainer.task.domain_datasets[subset][0])
    torch.save(all_losses, cfg.task.data + cfg.task.eval_domains + "/IRL_losses.pt" )
    torch.save(all_examples, cfg.task.data + cfg.task.eval_domains + "/IRL_inputs.pt" )
    try:
        for idx1 in range(len(all_examples)):
            assert (all_examples[idx1]==trainer.task.domain_datasets[subset][0][idx1]['source']).all()
    except:
        assert idx1 == len(all_examples)-1
        assert (all_examples[idx1][:len(trainer.task.domain_datasets[subset][0][idx1]['source'])] == trainer.task.domain_datasets[subset][0][idx1]['source']).all()
    import ipdb; ipdb.set_trace()
        # log validation stats
        # print(stats['loss'].shape)
        
        
    return 0


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

    main(cfg)
    # srun --label already creates ntasks number of parallel runs. ipdb will show all of them. Hence ntasks in salloc is very imp.
    # Path: dense

    # if args.profile:
    #     with torch.cuda.profiler.profile():
    #         with torch.autograd.profiler.emit_nvtx():
    #             # Path: dense
    #             distributed_utils.call_main(cfg, main)
    # else:
    #     distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
