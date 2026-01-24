export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

model_name=M3LLM_Qwen
mix_embeds=True
mlp_hidden_layers=-1

# training one model with a context length
torchrun --nnodes 1 --nproc-per-node 2 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --mlp_hidden_layers $mlp_hidden_layers \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --mlp_activation relu \
  --train_epochs 10 \
  --use_amp \
  --cosine \
  --tmax 10 \
  --mix_embeds ${mix_embeds} \
  --use_multi_gpu

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --mlp_hidden_layers $mlp_hidden_layers \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --mlp_activation relu \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds ${mix_embeds} \
  --test_dir long_term_forecast_traffic_672_96_${model_name}_custom_sl672_ll576_tl96_lr0.0001_bt256_wd1e-05_hd1024_hl${mlp_hidden_layers}_cosTrue_mix${mix_embeds}_test_0
done
