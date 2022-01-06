Environment:

python >= 3.6
pytorch >= 1.0.0
transformers >= 2.5.1
scikit-learn
tensorboardX

Files:

LPC.py: this file includes the LPC optimizer implementation, which is modified from AdamW optimizer implementation optimization.py by Huggingface Transformers.

loss_lc.py: this file includes the two types of Logits Calibration Loss, Cross Entropy with Logits Calibration (CELC) Loss and Mean Squared Error with Logits Calibration (MSELC) Loss, which is modified from loss.py from pytorch.nn.modules.

run_glue_with_LPC.py: this file is an example to run GLUE tasks with LPC optimizer, and is modified from the GLUE example run_glue.py by Huggingface Transformers.

download_glue_data.py: this file is the script to download GLUE dataset.


Run GLUE tasks:

With BERT-base pre-trained model:

export GLUE_DIR=/path/to/glue
export TASK_NAME=CoLA

python run_glue_with_LPC.py \
  --model_type bert \
  --model_name_or_path /path/to/model \
  --log_path ./log \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --warmup_steps 320 \
  --max_steps 9000 \
  --output_dir ./$TASK_NAME/ \
  --evaluate_during_training \
  --train_logging_steps 50 \
  --eval_logging_steps 180 \
  --optimizer LPC \
  --lpc_anneal_fun sigmoid \
  --lpc_anneal_t0 1000 \
  --lpc_anneal_k 0.1 \
  --lpc_pretrain_cof 5000.0 \
  --logging_Euclid_dist \
  --save_steps 100 \
  --reg_lambda 1 \
  --update_epoch 1 \
  --logits_calibraion_degree 1.0 \
  --eval_all_checkpoints

With ALBERT-xxlarge pre-trained model:

export GLUE_DIR=/path/to/glue
export TASK_NAME=CoLA

python run_glue_with_LPC.py \
  --model_type albert \
  --model_name_or_path /path/to/model \
  --log_path /path/to/log \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 1e-5 \
  --warmup_steps 320 \
  --max_steps 5336 \
  --output_dir /path/to/output/$TASK_NAME/ \
  --evaluate_during_training \
  --train_logging_steps 25 \
  --eval_logging_steps 100 \
  --albert_dropout 0.0 \
  --optimizer LPC \
  --lpc_anneal_fun sigmoid \
  --lpc_anneal_t0 1000 \
  --lpc_anneal_k 0.1 \
  --lpc_pretrain_cof 5000.0 \
  --logging_Euclid_dist \
  --save_steps 100 \
  --reg_lambda 1 \
  --update_epoch 1 \
  --logits_calibraion_degree 1.0 \
  --eval_all_checkpoints 








