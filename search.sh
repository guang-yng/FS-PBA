TASK=CoLA
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

hard=Y

for seed in 13 21 42 87 100
do
    for lr in 1e-2 3e-3 1e-4
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