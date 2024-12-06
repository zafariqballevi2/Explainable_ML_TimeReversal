#!/bin/bash
#
#SBATCH --job-name=MILC_CNN
#SBATCH --account=psy53c17
#SBATCH -o /data/users4/ziqbal5/abc/MILC/training_output/%j.out # STDOUT
#SBATCH --partition=qTRDGPUH
#SBATCH  --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --array=0-99

eval "$(conda shell.bash hook)"
conda activate z4_env
config=/data/users4/ziqbal5/abc/MILC/config.txt




Data=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
Encoder=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
Seed_value=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
ws=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
nw=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
wsize=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)
convsize=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)
ep=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)
tp=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $10}' $config)
samples=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $11}' $config)
l_ptr=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $12}' $config)
attr_alg=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $13}' $config)
fold_v=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $14}' $config)

b_file='output.txt'
echo " ${SLURM_JOBID}" >> $b_file

#python -m Main_exp --jobid $SLURM_JOBID --fold_v $fold_v --daata $Data --encoder $Encoder --seeds $Seed_value --ws $ws --nw $nw --wsize $wsize --convsize $convsize --epp $ep --tp $tp --samples $samples --l_ptr $l_ptr --attr_alg $attr_alg
#python -m Main_explosscurves --jobid $SLURM_JOBID --fold_v $fold_v --daata $Data --encoder $Encoder --seeds $Seed_value --ws $ws --nw $nw --wsize $wsize --convsize $convsize --epp $ep --tp $tp --samples $samples --l_ptr $l_ptr --attr_alg $attr_alg
python -m Main_downstreamlosscurves --jobid $SLURM_JOBID --fold_v $fold_v --daata $Data --encoder $Encoder --seeds $Seed_value --ws $ws --nw $nw --wsize $wsize --convsize $convsize --epp $ep --tp $tp --samples $samples --l_ptr $l_ptr --attr_alg $attr_alg
#python -m Main_Viz
#python -m test
#python -m Main_exp --fold_v 0 --daata HCP2 --encoder rnm --seeds 0 --ws 1200 --nw 1 --wsize 1200 --convsize 22 --epp 2 --tp 1200 --samples 830 --l_ptr F --attr_alg GS

#python -m Main_explosscurves --fold_v 0 --daata HCP2 --encoder rnn --seeds 1 --ws 1200 --nw 1 --wsize 1200 --convsize 22 --epp 2 --tp 1200 --samples 830 --l_ptr F --attr_alg GS
#python -m Main_downstreamlosscurves --fold_v 0 --daata FBIRN --encoder rnn --seeds 1 --ws 140 --nw 1 --wsize 140 --convsize 22 --epp 2 --tp 140 --samples 311 --l_ptr T --attr_alg GS
#python -m Main_downstreamlosscurves --fold_v 0 --daata BSNIP --encoder rnm --seeds 1 --ws 140 --nw 1 --wsize 140 --convsize 22 --epp 2 --tp 140 --samples 589 --l_ptr T --attr_alg IG

#%run Main_downstream.py --fold_v 0 --daata FBIRN --encoder rnm --seeds 0 --ws 140 --nw 1 --wsize 140 --convsize 22 --epp 2 --tp 140 --samples 311 --l_ptr F --attr_alg GS
#python -m Main_downstreamlosscurves --fold_v 0 --daata FBIRN --encoder cnn --seeds 1 --ws 20 --nw 7 --wsize 20 --convsize 2400 --epp 2 --tp 140 --samples 311 --l_ptr F --attr_alg GS