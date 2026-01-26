model_name=M3LLM_Qwen
mix_embeds=True
mlp_hidden_layers=0
multi_scale=False
multi_var=True
mix_embeds_type=v11
gpu=0

# training one model with a context length
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 128 \
  --learning_rate 0.0009 \
  --mlp_hidden_layers $mlp_hidden_layers \
  --train_epochs 100 \
  --use_amp \
  --gpu $gpu \
  --cosine \
  --tmax 7 \
  --mix_embeds $mix_embeds \
  --multi_var $multi_var \
  --multi_scale $multi_scale \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --pdm_blocks_num 1 \
  --mix_embeds_type $mix_embeds_type \
  --patience 10 \
  --weight_decay 0.000005 \
  --drop_last

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 128 \
  --learning_rate 0.0009 \
  --mlp_hidden_layers $mlp_hidden_layers \
  --train_epochs 100 \
  --use_amp \
  --gpu $gpu \
  --cosine \
  --tmax 7 \
  --mix_embeds $mix_embeds \
  --multi_var $multi_var \
  --multi_scale $multi_scale \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --pdm_blocks_num 1 \
  --mix_embeds_type $mix_embeds_type \
  --patience 10 \
  --weight_decay 0.000005 \
  --drop_last \
  --test_dir long_term_forecast_ETTh1_672_96_${model_name}_ETTh1_sl672_ll576_tl96_lr0.0009_bt128_wd5e-06_hd256_hl${mlp_hidden_layers}_cosTrue_mix${mix_embeds}_multivar${multi_var}_multiscale${multi_scale}_type${mix_embeds_type}_test_0
done

