TAG=exp-stage
model=roberta-large
cuda=1,5
bs=4
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

hard=Y
seed=13
lr=1e-3

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
TRAIN_PARM="prompt adatper bias" \
bash run_mulstage_experiment.sh "--use_adapter"