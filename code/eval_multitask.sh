set -e #sets the shell option to exit immediately if any command exits with a non-zero status, useful for error handling.
n_gpus=1
export CUDA_VISIBLE_DEVICES=0; #,1; #set to one GPU only to avoid parralization problems

### !!! ATTENTION !!! ### 
## change "file_dir" and "file_name" in the my_metric.py file based on the current experiment
## file_name = task_name, file_dir = eval_dir

task_name="zero_All" ## the experiment name
data_dir="data/Low-shot" 
train_dir="output/${task_name}/train"
eval_dir="output/${task_name}/eval"
model_dir="output/${task_name}/Trained_model" 
logging_train_dir="${train_dir}/runs"
logging_eval_dir="${eval_dir}/runs"

test_file="${data_dir}/test_Mawqif.json" ##

metric_name_or_path="my_metric.py"
#metric_for_best_model="overall_f1" #it produce error
source_column="prompt" #"text" 
target_column="response" #"stance"
truncation_side="left" #determines which side of input sequence should be truncated if sequence length exceeds the maximum sequence length allowed by the tokenizer
max_source_length=1024
max_target_length=16
#model_name_or_path="bigscience/mt0-large"
per_device_train_batch_size=32
per_device_eval_batch_size=32
gradient_accumulation_steps=4
lr=1e-4 #3e-4 #1e-3 #"in my experiments 1e-4 and 3e-4 worked for almost all problems" #5e-5 #5e-4
num_train_epochs=10


python run_seq2seq_M.py \
    --task_name ${task_name} \
    --test_file ${test_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${model_dir} \
    --do_predict \
    --predict_with_generate \
    --metric_name_or_path ${metric_name_or_path} \
    --output_dir ${eval_dir} \
    --logging_dir ${logging_eval_dir} \
    --overwrite_output_dir \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --optim adafactor \
    --gradient_checkpointing
    #--cache_dir ${cache_dir} \


#python merge_predict_res.py -d ${dataset_name} -s ${speaker} -c ${context_window_size} -p ${output_dir}/test_generated_predictions.json

#python ../../../nlu/evaluate_unified_datasets.py -p ${output_dir}/predictions.json

#from T0: it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32

