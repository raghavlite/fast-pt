data_bin=$1
model=$2
domain=$3
dapt_model=$4
output=$5
estimate=$6
precomputed_prior=$7


if [[ $model == *"gpt3_small"* ]]; then
    bash scripts/ensemble_eval_lm.sh $data_bin  /checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-4.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-8.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-12.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-16.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-20.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-24.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-28.pt:${dapt_model} $domain $domain $output $estimate $precomputed_prior;
elif [[ $model == *"gpt3_medium"* ]]; then
    bash scripts/ensemble_eval_lm.sh $data_bin /checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-0.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-8.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-16.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-24.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-32.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-40.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-48.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-56.pt $domain $domain $output $estimate $precomputed_prior;
elif [[ $model == *"gpt3_large"* ]]; then
    bash scripts/ensemble_eval_lm.sh $data_bin /checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-16.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-32.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-48.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-64.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-80.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-96.pt:/checkpoint/suching/publication_results/48hr/${model}/checkpoint_last-rank-112.pt:${dapt_model} $domain $domain $output $estimate $precomputed_prior;
fi;
