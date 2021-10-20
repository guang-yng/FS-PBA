for seed in 13
do
    for bs in 2
    do
        for lr in 3e-2 1e-2
        do
            for prompt in 2
            do
              CUDA_VISIBLE_DEVICES=1,2,3,4 \
              TAG=exp \
              TASK=SST-2 \
              PROMPT=$prompt \
              BS=$bs \
              LR=$lr \
              SEED=$seed \
              MODEL=roberta-large \
              bash run_prompt_experiment.sh
            done
        done
    done
done
