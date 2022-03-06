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
# 
# for len in 512 16 32 64 128 256
# do
#     # None
#     CUDA_VISIBLE_DEVICES=$cuda \
#     TASK=$TASK \
#     TAG=$TAG \
#     BS=$bs \
#     SEED=13 \
#     MODEL=$model \
#     HARD=$hard \
#     NOTRAIN=1 \
#     GPUN=$gpun \
#     bash run_experiment.sh "--max_seq_len $len"
# done

# PROMPT
for len in 16 32 64 128 256 512
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
    bash run_experiment.sh "--training_params prompt --max_seq_len $len"
    if (($? != 0)); then exit 0; fi
    echo "Hi"
done

# Bias
for len in 16 32 64 128 256 512
do
    CUDA_VISIBLE_DEVICES=$cuda \
    TASK=$TASK \
    TAG=$TAG \
    BS=$bs \
    LR=$lr \
    SEED=$seed \
    MODEL=$model \
    HARD=$hard \
    GPUN=$gpun \
    bash run_experiment.sh "--training_params bias --max_seq_len $len"
    if (($? != 0)); then exit 0; fi
done

# Adapter
for len in 16 32 64 128 256 512
do
    CUDA_VISIBLE_DEVICES=$cuda \
    TASK=$TASK \
    TAG=$TAG \
    BS=$bs \
    LR=$lr \
    SEED=$seed \
    MODEL=$model \
    HARD=$hard \
    GPUN=$gpun \
    bash run_experiment.sh "--training_params adapter --use_adapter --max_seq_len $len"
    if (($? != 0)); then exit 0; fi
done

# Prompt + Adapter
for len in 16 32 64 128 256 512
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
    bash run_experiment.sh "--training_params prompt,adapter --use_adapter --max_seq_len $len"
    if (($? != 0)); then exit 0; fi
done

# Prompt + Bias
for len in 16 32 64 128 256 512
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
    bash run_experiment.sh "--training_params prompt,bias --max_seq_len $len"
    if (($? != 0)); then exit 0; fi
done

# Bias + Adapter
for len in 16 32 64 128 256 512
do
    CUDA_VISIBLE_DEVICES=$cuda \
    TASK=$TASK \
    TAG=$TAG \
    BS=$bs \
    LR=$lr \
    SEED=$seed \
    MODEL=$model \
    HARD=$hard \
    GPUN=$gpun \
    bash run_experiment.sh "--training_params bias,adapter --use_adapter --max_seq_len $len"
    if (($? != 0)); then exit 0; fi
done

# Prompt + Bias + Adapter
for len in 16 32 64 128 256 512
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
    bash run_experiment.sh "--training_params prompt,bias,adapter --use_adapter --max_seq_len $len"
    if (($? != 0)); then exit 0; fi
done


if [[ $task == mnli ]]; then
    task=mnli-mm
    for hard in Y N 
    do
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'do_train': False}" > ./result/$TASK/$TASK-mm-$hard-none.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt']}" > ./result/$TASK/$TASK-mm-$hard-prompt.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias']}" > ./result/$TASK/$TASK-mm-$hard-bias.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['adapter']}" > ./result/$TASK/$TASK-mm-$hard-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,adapter']}" > ./result/$TASK/$TASK-mm-$hard-prompt-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias']}" > ./result/$TASK/$TASK-mm-$hard-prompt-bias.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias,adapter']}" > ./result/$TASK/$TASK-mm-$hard-bias-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias,adapter']}" > ./result/$TASK/$TASK-mm-$hard-prompt-bias-adapter.out
    done
fi
