#!/bin/bash
# Required environment variables:
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B

TASK=SST-2
model=roberta-large
cuda=0
bs=1
gpun=1

[ ! -d "./result/$TASK" ] && mkdir ./result/$TASK

case $TASK in
    CoLA)
        task=cola
    ;;
    SST-2)
        task=sst-2
    ;;
    MNLI)
        task=mnli
    ;;
    STS-B)
        task=sts-b/pearson
    ;;
    MRPC)
        task=mrpc/f1
    ;;
    QQP)
        task=qqp/f1
    ;;
    QNLI)
        task=qnli
    ;;
    RTE)
        task=rte
    ;;
esac

TAG=exp-time
lr=1e-4
seed=13
hard=N

# Bias
for len in 496 16 32 64 128 256
do
    CUDA_VISIBLE_DEVICES=$cuda \
    TASK=$TASK \
    TAG=$TAG \
    BS=$bs \
    LR=$lr \
    PROMPT=prompt \
    SEED=$seed \
    MODEL=$model \
    HARD=$hard \
    GPUN=$gpun \
    bash run_experiment.sh "--training_params bias --max_seq_len 512 --input_length $len"
    if (($? != 0)); then exit 0; fi
    echo "Hi"
done

# Finetune
for len in 16 32 64 128 256 496
do
    CUDA_VISIBLE_DEVICES=$cuda \
    TASK=$TASK \
    TAG=$TAG \
    BS=$bs \
    LR=$lr \
    PROMPT=prompt \
    SEED=$seed \
    MODEL=$model \
    HARD=$hard \
    GPUN=$gpun \
    bash run_experiment.sh "--training_params all --max_seq_len 512 --input_length $len"
    if (($? != 0)); then exit 0; fi
done

# Adapter
for len in 16 32 64 128 256 496
do
    CUDA_VISIBLE_DEVICES=$cuda \
    TASK=$TASK \
    TAG=$TAG \
    BS=$bs \
    LR=$lr \
    PROMPT=prompt \
    SEED=$seed \
    MODEL=$model \
    HARD=$hard \
    GPUN=$gpun \
    bash run_experiment.sh "--training_params adapter --use_adapter --max_seq_len 512 --input_length $len"
    if (($? != 0)); then exit 0; fi
done


# Prompt
for len in 16 32 64 128 256 496
do
    CUDA_VISIBLE_DEVICES=$cuda \
    TASK=$TASK \
    TAG=$TAG \
    BS=$bs \
    LR=$lr \
    PROMPT=prompt \
    SEED=$seed \
    MODEL=$model \
    HARD=$hard \
    GPUN=$gpun \
    bash run_experiment.sh "--training_params prompt --max_seq_len 512 --input_length $len"
    if (($? != 0)); then exit 0; fi
done