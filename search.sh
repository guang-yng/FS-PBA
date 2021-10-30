CUDA_VISIBLE_DEVICES=1,3,4,5 \
TASK=SST-2 \
TAG=exp \
BS=2 \
SEED=13 \
MODEL=roberta-large \
HARD=Y \
NOTRAIN=1 \
bash run_experiment.sh

python tools/gather_result.py --condition "{'tag': 'exp', 'task_name': 'sst-2', 'do_train': False}" > SST-2-Y-none.out