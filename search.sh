for seed in 13 21 42 87 100
do
    for bs in 2
    do
        for lr in 1e-3 3e-3 1e-3 3e-4
        do
            for prompt in 2
            do
                CUDA_VISIBLE_DEVICES=1,2,4,5 \
                TAG=exp \
                TASK=SST-2 \
                BS=$bs \
                LR=$lr \
                PROMPT=$prompt \
                SEED=$seed \
                MODEL=roberta-large \
                bash run_prompt_experiment.sh
                if (($? != 0)); then exit 0; fi
            done
        done
    done
done

python tools/gather_result.py --condition "{'tag': 'exp', 'task_name': 'sst-2'}"