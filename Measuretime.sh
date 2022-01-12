#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -G 1
#SBATCH -c 1

TASK=SST-2
model=roberta-large
cuda=1
bs=4
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

for hard in Y N
do
    TAG=exp-$hard

    # PROMPT
    for seed in 13
    do
        for lr in 1e-4
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

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt']}" > ./result/$TASK/$TASK-$hard-prompt-time.out

    # Bias
    for seed in 13
    do
        for lr in 1e-4
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
            bash run_experiment.sh "--training_params bias"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias']}" > ./result/$TASK/$TASK-$hard-bias-time.out

    # Adapter
    for seed in 13
    do
        for lr in 1e-4
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
            bash run_experiment.sh "--training_params adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['adapter']}" > ./result/$TASK/$TASK-$hard-adapter-time.out

    # Prompt + Adapter
    for seed in 13
    do
        for lr in 1e-4
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
            bash run_experiment.sh "--training_params prompt,adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,adapter']}" > ./result/$TASK/$TASK-$hard-prompt-adapter-time.out

    # Prompt + Bias
    for seed in 13
    do
        for lr in 1e-4
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
            bash run_experiment.sh "--training_params prompt,bias"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias']}" > ./result/$TASK/$TASK-$hard-prompt-bias-time.out

    # Bias + Adapter
    for seed in 13
    do
        for lr in 1e-4
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
            bash run_experiment.sh "--training_params bias,adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias,adapter']}" > ./result/$TASK/$TASK-$hard-bias-adapter-time.out

    # Prompt + Bias + Adapter
    for seed in 13
    do
        for lr in 1e-4
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
            bash run_experiment.sh "--training_params prompt,bias,adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias,adapter']}" > ./result/$TASK/$TASK-$hard-prompt-bias-adapter-time.out
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
