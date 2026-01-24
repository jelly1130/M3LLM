model_name=M3LLM_Qwen
mix_embeds=True
mlp_hidden_layers=1
mlp_hidden_dim=512
multi_var=True
mix_embeds_type=v11
multi_scale=True
dropout=0.1
gpu=2


data=(ER)
data_path=(
    ERing
)
export CUDA_LAUNCH_BLOCKING=1
for (( i=0; i<${#data[@]}; i++ ))
do
    echo $i,${data[$i]},${data_path[$i]}
    python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path ./all_datasets/Classification/UEA/${data_path[$i]} \
        --data_path ${data[$i]} \
        --model_id ${data[$i]} \
        --model $model_name \
        --data UEA \
        --token_len 16 \
        --batch_size 32 \
        --des 'Exp' \
        --itr 1 \
        --multi_var $multi_var \
        --mix_embeds $mix_embeds \
        --mix_embeds_type $mix_embeds_type \
        --learning_rate 0.0005 \
        --mlp_hidden_dim $mlp_hidden_dim \
        --mlp_hidden_layers $mlp_hidden_layers \
        --cosine \
        --train_epochs 100 \
        --patience 10 \
        --multi_scale $multi_scale \
        --down_sampling_layers 1 \
        --down_sampling_window 2 \
        --down_sampling_method avg \
        --dropout $dropout \
        --moving_avg 13 \
        --gpu $gpu \
        --wandb
done

# AWR
# python -u run.py \
#     --task_name classification \
#     --is_training 1 \
#     --root_path ./all_datasets/Classification/UEA/ArticularyWordRecognition \
#     --data_path AWR \
#     --model_id AWR \
#     --model $model_name \
#     --data UEA \
#     --token_len 16 \
#     --batch_size 32 \
#     --des 'Exp' \
#     --itr 1 \
#     --multi_var $multi_var \
#     --mix_embeds $mix_embeds \
#     --mix_embeds_type $mix_embeds_type \
#     --learning_rate 0.0006 \
#     --mlp_hidden_dim 512 \
#     --mlp_hidden_layers 1 \
#     --cosine \
#     --train_epochs 100 \
#     --patience 10 \
#     --multi_scale false \
#     --down_sampling_layers 1 \
#     --down_sampling_window 2 \
#     --down_sampling_method avg \
#     --dropout $dropout \
#     --moving_avg 13 \
#     --tmax 7 \
#     --gpu $gpu \
#     --wandb

# AF
# python -u run.py \
#     --task_name classification \
#     --is_training 1 \
#     --root_path ./all_datasets/Classification/UEA/AtrialFibrillation \
#     --data_path AF \
#     --model_id AF \
#     --model $model_name \
#     --data UEA \
#     --token_len 16 \
#     --batch_size 16 \
#     --des 'Exp' \
#     --itr 1 \
#     --multi_var $multi_var \
#     --mix_embeds $mix_embeds \
#     --mix_embeds_type $mix_embeds_type \
#     --learning_rate 0.0007 \
#     --mlp_hidden_dim 512 \
#     --mlp_hidden_layers 1 \
#     --cosine \
#     --train_epochs 100 \
#     --patience 10 \
#     --multi_scale true \
#     --down_sampling_layers 1 \
#     --down_sampling_window 2 \
#     --down_sampling_method avg \
#     --dropout $dropout \
#     --moving_avg 13 \
#     --tmax 6 \
#     --gpu $gpu \
#     --wandb

# CR
# python -u run.py \
#     --task_name classification \
#     --is_training 1 \
#     --root_path ./all_datasets/Classification/UEA/AtrialFibrillation \
#     --data_path AF \
#     --model_id AF \
#     --model $model_name \
#     --data UEA \
#     --token_len 16 \
#     --batch_size 32 \
#     --des 'Exp' \
#     --itr 1 \
#     --multi_var $multi_var \
#     --mix_embeds $mix_embeds \
#     --mix_embeds_type $mix_embeds_type \
#     --learning_rate 0.0005 \
#     --mlp_hidden_dim 512 \
#     --mlp_hidden_layers 1 \
#     --cosine \
#     --train_epochs 100 \
#     --patience 10 \
#     --multi_scale false \
#     --down_sampling_layers 1 \
#     --down_sampling_window 2 \
#     --down_sampling_method avg \
#     --dropout $dropout \
#     --moving_avg 13 \
#     --tmax 6 \
#     --gpu $gpu \
#     --wandb