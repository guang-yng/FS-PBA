for seed in 13 21 42 87 100
do
    for bs in 2
    do
        for lr in 3e-4
        do
            for prompt in 10
            do
                CUDA_VISIBLE_DEVICES=1,2,3,4 \
                TAG=exp-pre \
                TASK=SST-2 \
                BS=$bs \
                LR=$lr \
                PROMPT=$prompt \
                SEED=$seed \
                MODEL=roberta-large \
                TEMPLATE=*cls*prompt*sent_0*_It_was*mask*.*sep+* \
                bash run_prompt_experiment.sh "--training_params prompt"
                if (($? != 0)); then exit 0; fi
            done
        done
    done
done

python tools/gather_result.py --condition "{'tag': 'exp-pre', 'task_name': 'sst-2', 'training_params': 'prompt'}"
