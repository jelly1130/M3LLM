model_name=M3LLM_Qwen
mix_embeds=True
multi_var=False
mix_embeds_type=v12
multi_scale=True
mlp_hidden_dim=1024
mlp_hidden_layers=3
# lr=5e-05
lr=0.001

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

# training one model with a context length
torchrun --rdzv_endpoint=localhost:29300 --nnodes 1 --nproc-per-node 4 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.csv \
  --model_id PEMS_672_660 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 660 \
  --token_len 12 \
  --test_seq_len 672 \
  --test_label_len 660 \
  --test_pred_len 12 \
  --batch_size 512 \
  --learning_rate $lr \
  --train_epochs 1000 \
  --patience 3 \
  --cosine \
  --use_amp \
  --des 'Exp' \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mlp_hidden_layers $mlp_hidden_layers \
  --mlp_activation gelu \
  --use_multi_gpu \
  --mix_embeds $mix_embeds \
  --multi_var $multi_var \
  --mix_embeds_type $mix_embeds_type \
  --multi_scale $multi_scale \
  --multi_scale_num 3 \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method conv \
  --moving_avg 25 \
  --tmax 7 \

# testing the model on all forecast lengths
for test_pred_len in 12 24 48 96
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.csv \
  --model_id PEMS_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 660 \
  --token_len 12 \
  --test_seq_len 672 \
  --test_label_len 660 \
  --test_pred_len $test_pred_len \
  --batch_size 512 \
  --learning_rate $lr \
  --train_epochs 100 \
  --patience 3 \
  --cosine \
  --use_amp \
  --lradj type2 \
  --des 'Exp' \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mlp_hidden_layers $mlp_hidden_layers \
  --mlp_activation gelu \
  --mix_embeds $mix_embeds \
  --multi_var $multi_var \
  --mix_embeds_type $mix_embeds_type \
  --multi_scale $multi_scale \
  --multi_scale_num 3 \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method conv \
  --moving_avg 25 \
  --tmax 7 \
  --test_dir long_term_forecast_PEMS_672_660_${model_name}_custom_sl672_ll660_tl12_lr${lr}_bt512_wd0_hd${mlp_hidden_dim}_hl${mlp_hidden_layers}_cosTrue_mix${mix_embeds}_multivar${multi_var}_multiscale${multi_scale}_type${mix_embeds_type}_Exp_0
done
