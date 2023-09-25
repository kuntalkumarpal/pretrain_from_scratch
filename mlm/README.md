# MLM Experiments

* Train MLM for adapting the model to nature of texts
* Score the performance of MLM in perplexity
* The source data is created along with tokenizer data creation
* To test impact of only human MLM : Create and use human_train and human_test from human.json
* To test impact of all data MLM : Use mlm_train and mlm_test

## MLM : Bulk
~~```
python run_mlm.py \
    --model_name_or_path /scratch/kkpal/NEW_VAREC/models/base_model \
    --model_type roberta \
    --tokenizer_name /scratch/kkpal/NEW_VAREC/models/base_model \
    --train_file /scratch/kkpal/NEW_VAREC/tokenizer/source_mlm_full/human_1K.json \
    --validation_split_percentage 5 \
    --max_seq_length 1024 \
    --mlm_probability 0.15 \
    --num_train_epochs 10 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 20 \
    --output_dir /scratch/kkpal/NEW_VAREC/models/mlm_full \
    --overwrite_output_dir
    ```~~

## MLM : line-by-line
~~```
python run_mlm.py \
    --model_name_or_path /scratch/kkpal/NEW_VAREC/models/base_model \
    --model_type roberta \
    --tokenizer_name /scratch/kkpal/NEW_VAREC/models/base_model \
    --train_file /scratch/kkpal/NEW_VAREC/tokenizer/source_mlm_full/human_1K.json \
    --validation_split_percentage 5 \
    --max_seq_length 1024 \
    --mlm_probability 0.15 \
    --num_train_epochs 10 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 20 \
    --output_dir /scratch/kkpal/NEW_VAREC/models/mlm_full \
    --line_by_line \
    --overwrite_output_dir
    ```~~


### MLM - Whole word masking (Trainonlydata)
```
python -m torch.distributed.launch --nproc_per_node=4 run_mlm_wwm.py \
    --model_name_or_path /scratch/kkpal/NEW_VAREC/models/base_model \
    --model_type roberta \
    --tokenizer_name /scratch/kkpal/NEW_VAREC/models/base_model \
    --train_file /scratch/kkpal/NEW_VAREC/tokenizer/source_mlm_trainonly/mlm_train.json \
    --validation_file /scratch/kkpal/NEW_VAREC/tokenizer/source_mlm_trainonly/mlm_test.json \
    --max_seq_length 800 \
    --mlm_probability 0.15 \
    --num_train_epochs 40 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 44 \
    --per_device_eval_batch_size 44 \
    --output_dir /scratch/kkpal/NEW_VAREC/models/mlm_wwm_trainonly \
    --save_steps 10000 \
    --logging_steps 5000 \
    --overwrite_output_dir
```

### MLM - Whole word masking (Trainonlydata - EVAL)
```
python run_mlm_wwm.py \
    --model_name_or_path /scratch/kkpal/NEW_VAREC/models/mlm_wwm_trainonly/checkpoint-480000/ \
    --model_type roberta \
    --tokenizer_name /scratch/kkpal/NEW_VAREC/models/mlm_wwm_trainonly/checkpoint-480000/ \
    --train_file /scratch/kkpal/NEW_VAREC/tokenizer/source_mlm_trainonly/mlm_train.json \
    --validation_file /scratch/kkpal/NEW_VAREC/tokenizer/source_mlm_trainonly/mlm_test.json \
    --max_seq_length 800 \
    --mlm_probability 0.15 \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --output_dir /scratch/kkpal/NEW_VAREC/models/mlm_wwm_trainonly/checkpoint-480000/ \
    --save_steps 5000 \
    --logging_steps 5000 
    
Perplexity : 1.41811313042803 - 420000
             1.4072407973252226 - 480000
```


