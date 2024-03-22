

## Dataset

The multidomain dataset scripts are housed in another repository, located [here](https://github.com/kernelmachine/demix-data). Clone that repository and follow instructions to setup data to train on.

Follow that tutorial to generate data-bins on eight (small) example domains.

Make sure to set the `DATA_DIR` accordingly.


## Fairseq Installation

Follow installation from 


## Running Code
Needs 4 gpus

./demix/train_1domain.sh 4 128 16 64 25000 25000 transformer_lm_gpt3_small unbalanced /usr/xtmp/rt195/DEMIX_DATA/PTData/PTData3/TrainData/multi_IL_75k_64e-4/data_bin_IRL_75k/ ../PT_Models combined valid_combined 64e-4 0.1 0824_mIRL_sample


