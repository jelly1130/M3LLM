model_name=M3LLM_Qwen
mix_embeds=True
multi_var=True
mix_embeds_type=v11
multi_scale=True
mlp_hidden_dim=512
lr=0.001

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

# training one model with a context length
torchrun --rdzv_endpoint=localhost:29300 --nnodes 1 --nproc-per-node 2 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 128 \
  --learning_rate $lr \
  --train_epochs 100 \
  --patience 10 \
  --use_amp \
  --cosine \
  --lradj type2 \
  --des 'Exp' \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mlp_activation relu \
  --use_multi_gpu \
  --mix_embeds $mix_embeds \
  --multi_var $multi_var \
  --mix_embeds_type $mix_embeds_type \
  --multi_scale $multi_scale \
  --multi_scale_num 3 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method bicubic \
  --tmax 9 \

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 128 \
  --learning_rate $lr \
  --train_epochs 100 \
  --patience 10 \
  --use_amp \
  --cosine \
  --lradj type2 \
  --des 'Exp' \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mlp_activation relu \
  --mix_embeds $mix_embeds \
  --multi_var $multi_var \
  --mix_embeds_type $mix_embeds_type \
  --multi_scale $multi_scale \
  --multi_scale_num 3 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method bicubic \
  --tmax 9 \
  --test_dir long_term_forecast_weather_672_96_${model_name}_custom_sl672_ll576_tl96_lr${lr}_bt128_wd0_hd${mlp_hidden_dim}_hl2_cosTrue_mix${mix_embeds}_multivar${multi_var}_multiscale${multi_scale}_type${mix_embeds_type}_Exp_0
done
