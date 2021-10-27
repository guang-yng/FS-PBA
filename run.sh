TAG=exp
TASK=SST-2
K=16
SEED=42
MODEL=roberta-large
PROMPT=10
REAL_BS=2
GS=1
LR=3e-4
MAX_STEP=1000
EVAL_STEP=100
DATA_DIR=data/k-shot/$TASK/$K-$SEED
TRIAL_IDTF=hand
TEMPLATE=*cls**sent_0*_It_was*prompt*mask*.*sep+*
MAPPING="{'0':'terrible','1':'great'}"
CUDA_VISIBLE_DEVICES=0,1,2,5 python run.py \
  --task_name $TASK \
  --data_dir $DATA_DIR \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy steps \
  --model_name_or_path $MODEL \
  --use_prompt \
  --prompt_num $PROMPT \
  --num_k $K \
  --max_seq_length 256 \
  --per_device_train_batch_size $REAL_BS \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps $GS \
  --learning_rate $LR \
  --max_steps $MAX_STEP \
  --logging_steps $EVAL_STEP \
  --eval_steps $EVAL_STEP \
  --num_train_epochs 0 \
  --output_dir result/$TASK-$K-$PROMPT-$SEED-$MODEL-$TRIAL_IDTF \
  --seed $SEED \
  --tag $TAG \
  --template $TEMPLATE \
  --mapping $MAPPING \
  --training_params prompt

rm -r result/$TASK-$K-$PROMPT-$SEED-$MODEL-$TRIAL_IDTF \