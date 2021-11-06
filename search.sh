# Required environment variables:
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B

TASK=MNLI
TAG=exp
model=roberta-large
cuda=6,7
bs=8
gpun=2

mkdir ./result/$TASK

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
esac

for hard in Y N
do

    # PROMPT
    for seed in 13 21 42 87 100
    do
        for lr in 3e-4
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
            bash run_experiment.sh "--training_params prompt"
            if (($? != 0)); then exit 0; fi
            echo "Hi"
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt'}" > ./result/$TASK/$TASK-$hard-prompt.out

    # Bias
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 3e-3
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
	    GPUN=$gpu \
            bash run_experiment.sh "--training_params bias"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'bias'}" > ./result/$TASK/$TASK-$hard-bias.out
done
