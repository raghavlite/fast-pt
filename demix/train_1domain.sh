#!/bin/bash
# !IMP: only works with IRL_dense_multidomain branch
# Number of GPUs you'd like to train on
NUM_GPUS=$1


# seq_len
TOKENS_PER_SAMPLE=$2;
# batch size
BATCH_SIZE=$3;
# update_freq
UPDATE_FREQ=$4;
# num_steps
NUM_STEPS=$5;

MAX_UPDATE=$6;


# Number of nodes you'd like to train on (assuming 8 GPUs per node)
NUM_NODES=1
# Distributed port
PORT=$(($SLURM_JOBID%65535))
# Fairseq model name (e.g. transformer_lm; see https://github.com/kernelmachine/demix/blob/main/fairseq/models/transformer_lm.py for other options)
ARCH=$7
# Baseline type: choice between demix, dense, unbalanced_dense, and domain_token
EXPERIMENT=$8
# Path to data-bins
DATA_PATH=$9

# path to directory to where you'd like to output the model
SERIALIZATION_DIR=${10}


# list of domains you'd like to train on, that can be found in $DATA_PATH
domains=${11};
valid_subset=${12};

# standard : 8e-4
LR=${13};
# standard : 0.1
CLIP_NORM=${14};

# suffix to append to model output (e.g. "test", "final")
FILE_SUFFIX=${15}





# domains=1b,cs,med,realnews;
# valid_subset=valid_1b,valid_cs,valid_med,valid_realnews;
# domains=1b,1bo;
# validation datasets for each domain
# domains=1b,cs;
# valid_subset=valid_1b,valid_cs;
# valid_subset=valid_1b,valid_1bo;

# domains=1b,cs;
# # validation datasets for each domain
# valid_subset=valid_1b,valid_cs;
# name of wandb project to track model output (at wandb.ai)

# validation datasets for each domain
# domains=realnews;
# valid_subset=valid_realnews;

# domains=combined;
# valid_subset=valid_combined;


WANDB_PROJECT=fast-pt;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=10;
echo $EXPERIMENT
if [[ $ARCH == *"gpt3_small"* ]]; then
     BATCH_SIZE_VALIDATION=256
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     if [[ $FILE_SUFFIX == *"TK"* ]]; then    
          SAVE_INTERVAL_UPDATES=100;
          VALIDATION_INTERVAL=100;
          # SAVE_INTERVAL_UPDATES=100;
          # VALIDATION_INTERVAL=100;
          TASK_NAME=multidomain_language_modeling_TK;
     elif [[ $FILE_SUFFIX == *"IL"* ]]; then
          SAVE_INTERVAL_UPDATES=5000;
          VALIDATION_INTERVAL=1000;
          TASK_NAME=multidomain_language_modeling;
     elif [[ $FILE_SUFFIX == *"IRL"* ]]; then
          SAVE_INTERVAL_UPDATES=5000;
          VALIDATION_INTERVAL=1000;
          TASK_NAME=multidomain_language_modeling_EX;
     elif [[ $FILE_SUFFIX == *"baseline"* ]]; then    
          SAVE_INTERVAL_UPDATES=5000;
          VALIDATION_INTERVAL=1000;
          # SAVE_INTERVAL_UPDATES=100;
          # VALIDATION_INTERVAL=100;
          # SAVE_INTERVAL_UPDATES=30;
          # VALIDATION_INTERVAL=30;
          TASK_NAME=multidomain_language_modeling;
     elif [[ $FILE_SUFFIX == *"HL"* ]]; then    
          SAVE_INTERVAL_UPDATES=1000;
          VALIDATION_INTERVAL=1000;
          # SAVE_INTERVAL_UPDATES=100;
          # VALIDATION_INTERVAL=100;
          TASK_NAME=multidomain_language_modeling_EX;
     elif [[ $FILE_SUFFIX == *"ALM"* ]]; then    
          SAVE_INTERVAL_UPDATES=1000;
          VALIDATION_INTERVAL=1000;
          # SAVE_INTERVAL_UPDATES=100;
          # VALIDATION_INTERVAL=100;
          TASK_NAME=multidomain_language_modeling_EX;
     fi;
fi;



if [[ $EXPERIMENT == *"unbalanced"* ]]; then
     srun --label python fairseq_cli/train.py     $DATA_PATH \
               --task ${TASK_NAME} \
               --sample-break-mode none \
               --log-format simple  \
               --log-interval $LOG_INTERVAL    \
               --skip-invalid-size-inputs-valid-test     \
               --validate-interval-updates $VALIDATION_INTERVAL     \
               --save-interval-updates $SAVE_INTERVAL_UPDATES     \
               --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
               --arch $ARCH    \
               --criterion desynchronized_cross_entropy     \
               --lr-scheduler polynomial_decay     \
               --num-workers 4 \
               --max-sentences $BATCH_SIZE \
               --no-epoch-checkpoints \
               --max-sentences-valid $BATCH_SIZE \
               --lr $LR              \
               --tokens-per-sample $TOKENS_PER_SAMPLE          \
               --optimizer adam \
               --adam-betas '(0.9, 0.95)'  \
               --adam-eps 10e-8 \
               --weight-decay 0.1 \
               --clip-norm $CLIP_NORM      \
               --max-update ${MAX_UPDATE}    \
               --total-num-update $NUM_STEPS     \
               --warmup-updates $NUM_WARMUP_STEPS     \
               --update-freq $UPDATE_FREQ     \
               --save-dir ${SERIALIZATION_DIR}/unbalanced_dense_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
               --batch-size-valid ${BATCH_SIZE_VALIDATION}  \
               --wandb-project $WANDB_PROJECT           \
               --valid-subset $valid_subset \
               --train-domains $domains  \
               --eval-domains $domains \
               --required-batch-size-multiple 1 \
               --memory-efficient-fp16 \
               --distributed-world-size $NUM_GPUS \
               --distributed-port $PORT \
               --all-gather-list-size 128000 \
               --unbalanced ;
fi;
