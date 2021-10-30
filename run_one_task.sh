# Required environment variables:
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B

TAG=exp
model=roberta-large
cuda=1,2,3,4
bs=2

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
    # None
    CUDA_VISIBLE_DEVICES=$cuda \
    TASK=$TASK \
    TAG=$TAG \
    BS=$bs \
    SEED=13 \
    MODEL=$model \
    HARD=$hard \
    NOTRAIN=1 \
    bash run_experiment.sh

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'do_train': False}" > $TASK-$hard-none.out

    # PROMPT
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 3e-3 1e-3
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
            bash run_experiment.sh "--training_params prompt"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt'}" > $TASK-$hard-prompt.out

    # Bias
    for seed in 13 21 42 87 100
    do
        for lr in 1e-3 3e-4 1e-3
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            bash run_experiment.sh "--training_params bias"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'bias'}" > $TASK-$hard-bias.out

    # Adapter
    for seed in 13 21 42 87 100
    do
        for lr in 3e-4 1e-3 3e-5
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            bash run_experiment.sh "--training_params adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'adapter'}" > $TASK-$hard-adapter.out

    # Prompt + Adapter
    for seed in 13 21 42 87 100
    do
        for lr in 3e-4 1e-3 3e-5
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
            bash run_experiment.sh "--training_params prompt,adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt,adapter'}" > $TASK-$hard-prompt-adapter.out

    # Prompt + Bias
    for seed in 13 21 42 87 100
    do
        for lr in 3e-4 1e-3 3e-5
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
            bash run_experiment.sh "--training_params prompt,bias"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt,bias'}" > $TASK-$hard-prompt-bias.out

    # Bias + Adapter
    for seed in 13 21 42 87 100
    do
        for lr in 3e-4 1e-3 3e-5
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            bash run_experiment.sh "--training_params bias,adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'bias,adapter'}" > $TASK-$hard-bias-adapter.out

    # Prompt + Bias + Adapter
    for seed in 13 21 42 87 100
    do
        for lr in 3e-4 1e-3 3e-5
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
            bash run_experiment.sh "--training_params prompt,bias,adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt,bias,adapter'}" > $TASK-$hard-prompt-bias-adapter.out
done