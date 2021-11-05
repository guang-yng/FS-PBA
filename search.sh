TASK=MNLI
TAG=exp
model=roberta-large
cuda=2,3,5,7
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

hard=Y

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
        bash run_experiment.sh "--training_params prompt"
        if (($? != 0)); then exit 0; fi
        echo "Hi"
    done
done

python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt'}" > ./result/$TASK/$TASK-$hard-prompt.out
