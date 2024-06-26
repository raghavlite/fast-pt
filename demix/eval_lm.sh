#!/bin/bash
# Path to data-bins
data_bin=$1
# path to model, e.g. model_path/checkpoint_last.pt, or model_path/checkpoint_last-rank-4.pt if using demix
model=$2
# path to results output, e.g. ppl over blocks
results_path=$3
# split you'd like to evaluate on ("valid" or "test")
split=$4
# domain you'd like to evaluate on
target_domain=$5
# if using domain token, set this value to force a particular domain token prepended to each document, e.g "1b" or "med"
force_domain_token=$6

if [[ "$model" == *"domain_token"* ]]; then
        if [[ -z "$force_domain_token" ]]; then
                python fairseq_cli/eval_lm.py \
                        ${data_bin} \
                        --path ${model} \
                        --gen-subset ${split}_${target_domain} \
                        --task multidomain_language_modeling \
                        --sample-break-mode none \
                        --tokens-per-sample 1024     \
                        --batch-size 2  \
                        --original-domains 1b,anonymized_openwebtext,anonymized_realnews,anonymized_reviews,cs,legal,med,reddit \
                        --eval-domains ${target_domain} \
                        --results-path ${results_path} \
                        --partial-load \
                        --add-domain-token;
        else
                python fairseq_cli/eval_lm.py \
                        ${data_bin} \
                        --path ${model} \
                        --gen-subset ${split}_${target_domain} \
                        --task multidomain_language_modeling \
                        --sample-break-mode none \
                        --tokens-per-sample 1024     \
                        --batch-size 2  \
                        --eval-domains ${target_domain} \
                        --results-path ${results_path} \
                        --partial-load \
                        --add-domain-token \
                        --force-domain-token $force_domain_token;
        fi;

elif [[ "$model" == *"gshard"* || "$model" == *"switch"* ]]; then
  srun --label python fairseq_cli/eval_lm.py \
        ${data_bin} \
        --path ${model} \
        --gen-subset ${split}_${target_domain} \
        --task multidomain_language_modeling \
        --sample-break-mode none \
        --tokens-per-sample 1024     \
        --batch-size 2  \
        --eval-domains ${target_domain} \
        --results-path ${results_path} \
        --distributed-world-size 64 \
        --distributed-port 4234 \
	--is-moe;
else
        srun --label python fairseq_cli/eval_lm.py \
        ${data_bin} \
        --path ${model} \
        --gen-subset ${split}_${target_domain} \
        --task multidomain_language_modeling \
        --sample-break-mode none \
        --tokens-per-sample 128     \
        --batch-size 512  \
        --eval-domains ${target_domain} \
        --results-path ${results_path} \
	--partial-load 
fi
