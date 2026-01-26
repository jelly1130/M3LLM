model_name=M3LLM_Llama3
mix_embeds=True
mlp_hidden_layers=2
mlp_hidden_dim=400
multi_scale=True
multi_var=False
mix_embeds_type=v12
gpu=0

# training one model with a context length
python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_672_96 \
  --model $model_name \
  --data ETTh2 \
  --des Exp \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --test_data_path ETTh1.csv \
  --batch_size 2048 \
  --learning_rate 0.0002 \
  --mlp_hidden_layers $mlp_hidden_layers \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mlp_activation tanh \
  --train_epochs 45 \
  --use_amp \
  --gpu $gpu \
  --cosine \
  --tmax 7 \
  --mix_embeds $mix_embeds \
  --multi_var $multi_var \
  --multi_scale $multi_scale \
  --down_sampling_layers 4 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --moving_avg 17 \
  --mix_embeds_type $mix_embeds_type \
  --patience 10 \
  --weight_decay 0.0001 \
  --dropout 0.15 \
  --drop_last

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh2_672_96 \
  --model $model_name \
  --data ETTh1 \
  --des Exp \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 2048 \
  --learning_rate 0.0002 \
  --mlp_hidden_layers $mlp_hidden_layers \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mlp_activation tanh \
  --train_epochs 45 \
  --use_amp \
  --gpu $gpu \
  --cosine \
  --tmax 7 \
  --mix_embeds $mix_embeds \
  --multi_var $multi_var \
  --multi_scale $multi_scale \
  --down_sampling_layers 4 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --moving_avg 17 \
  --mix_embeds_type $mix_embeds_type \
  --patience 10 \
  --weight_decay 0.0001 \
  --drop_last \
  --test_dir zero_shot_forecast_ETTh2_672_96_${model_name}_ETTh2_sl672_ll576_tl96_lr0.0002_bt2048_wd0.0001_hd${mlp_hidden_dim}_hl${mlp_hidden_layers}_cosTrue_mix${mix_embeds}_multivar${multi_var}_multiscale${multi_scale}_type${mix_embeds_type}_Exp_0
done
