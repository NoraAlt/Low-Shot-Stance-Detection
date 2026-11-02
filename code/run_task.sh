#Source for seq2seqTrainingArguments: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
set -e #sets the shell option to exit immediately if any command exits with a non-zero status, useful for error handling.
n_gpus=1
export CUDA_VISIBLE_DEVICES=0; #,1; #set to one GPU only to avoid parralization problems

task_name="SingleTask_stance"
data_dir="data/Prompted_data_EN"
train_dir="output/${task_name}/train"
eval_dir="output/${task_name}/eval"
model_dir="output/${task_name}/Trained_model" ### Change this based on the generated checkpoint ###
logging_train_dir="${train_dir}/runs"
logging_eval_dir="${eval_dir}/runs"

train_file="${data_dir}/train_Mawqif.json"
validation_file="${data_dir}/test_Mawqif.json"
test_file="${data_dir}/test_Mawqif.json"

metric_name_or_path="my_metric.py"
#metric_for_best_model="overall_f1" #it produce error
source_column="prompt" #"context"
target_column="response"
truncation_side="left" #determines which side of input sequence should be truncated if sequence length exceeds the maximum sequence length allowed by the tokenizer
max_source_length=1024
max_target_length=16
model_name_or_path="bigscience/mt0-large" #"/home/nora_alturayeif/STANCE/My_T/output/multi_noStance_Prompt/Trained_model"
per_device_train_batch_size=32
per_device_eval_batch_size=32
gradient_accumulation_steps=4
lr=3e-4  #1e-4 #1e-3 #"in my experiments 1e-4 and 3e-4 worked for almost all problems" #5e-5 #5e-4
num_train_epochs=10


#python -m torch.distributed.launch \
    #--nproc_per_node ${n_gpus} run_seq2seq_M.py \
#python run_seq2seq_M.py \
nohup python run_seq2seq_M.py \
    --task_name ${task_name} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --do_eval \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --output_dir ${model_dir} \
    --logging_dir ${logging_train_dir} \
    --overwrite_output_dir \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --optim adafactor \
    --gradient_checkpointing \
    --load_best_model_at_end \
    --early_stopping_patience 5 \
    --prediction_loss_only &> SingleTask_stance.log &
    #--report_to wandb
    #load_best_model_at_end --> means that the best model based on the validation loss will be saved at the end of the training process. Otherwise, he model will be saved at the end of the training process regardless of whether it has the best validation loss or not.
    ##--metric_for_best_model ${metric_for_best_model} --> it produce error
    #optim--> The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, adamw_anyprecision or adafactor.--> "I would strongly recommend using AdaFactor and not ADAM for T5 training and finetuning."
    #--cache_dir ${cache_dir} \ --> "Where to store the pretrained models downloaded from huggingface --> it produces error, better to store it in the default cash
