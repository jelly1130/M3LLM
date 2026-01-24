import argparse
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
from exp.exp_in_context_forecasting import Exp_In_Context_Forecast
from exp.exp_classification import Exp_Classification

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='M3LLM')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, zero_shot_forecasting, in_context_forecasting]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='M3LLM_Llama',
                        help='model name, options: [M3LLM_Llama, M3LLM_Gpt2, M3LLM_Opt1b, M3LLM_Qwen]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--test_data_path', type=str, default='ETTh1.csv', help='test data file used in zero shot forecasting')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--drop_last', type=str2bool, nargs='?', const=True, help='drop last batch in data loader', default=False)
    parser.add_argument('--val_set_shuffle', action='store_false', default=True, help='shuffle validation set')
    parser.add_argument('--drop_short', action='store_true', default=False, help='drop too short sequences in dataset')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=576, help='label length')
    parser.add_argument('--token_len', type=int, default=96, help='token length')
    parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
    parser.add_argument('--test_label_len', type=int, default=576, help='test label len')
    parser.add_argument('--test_pred_len', type=int, default=96, help='test pred len')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--llm_ckp_dir', type=str, default='./llama2', help='llm checkpoints dir') # NOTE( ): useless
    parser.add_argument('--mlp_hidden_dim', type=int, default=256, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='mlp hidden layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', type=str2bool, nargs='?', const=True, help='use automatic mixed precision training', default=False)
    parser.add_argument('--cosine', type=str2bool, nargs='?', const=True, help='use cosine annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mix_embeds', type=str2bool, nargs='?', help='mix embeds', const=True, default=False)
    parser.add_argument('--mix_embeds_type', type=str, default='v1', help='mix embeds type')
    parser.add_argument('--n_vars', type=int, default=7, help='n_vars')
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    
    # multivar
    parser.add_argument('--multi_var', type=str2bool, nargs='?', help='multi var', const=True, default=False)
    
    # multiscale
    parser.add_argument('--multi_scale', type=str2bool, nargs='?', help='multi scale', const=True, default=False)
    parser.add_argument('--pdm_blocks_num', type=int, default=1, help='pdm blocks num')
    parser.add_argument('--multi_scale_type', type=str, default='v1', help='multi scale type')
    parser.add_argument('--moving_avg', type=int, default=25, help='moving average')
    parser.add_argument('--multi_scale_num', type=int, default=3, help='multi scale num')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    
    # wandb
    parser.add_argument('--wandb', type=str2bool, nargs='?', help='use wandb', const=True, default=False)
    
    args = parser.parse_args()
    
    wandb = None
    if args.wandb:
        import wandb
        wandb.init(
            project=args.data,
            config=vars(args),
        )
    
    # re-define
    model_chp_dir = {
        'M3LLM_Llama': './llama2',
        'M3LLM_Qwen': './qwen2',
    }
    if args.model not in model_chp_dir:
        print(f'{args.model} is not in model_chp_dir')
        exit(0)
    args.llm_ckp_dir = model_chp_dir[args.model]
    dataset_nvars = {
        'ETTh1.csv': 7,
        'weather.csv': 21,
        'AWR': 9,
        'AF': 2,
        'BL': 4,
        'CR': 6,
        'ER': 4,
        'FM': 28,
        'RS': 6,
        'SRS2': 7,
        'SWJ': 4,
        'UWG': 3
    }
    if args.data_path not in dataset_nvars:
        print(f'{args.data_path} is not in dataset_nvars')
        exit(0)
    args.n_vars = dataset_nvars[args.data_path]
    

    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0")) 
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)
    
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'zero_shot_forecast':
        Exp = Exp_Zero_Shot_Forecast
    elif args.task_name == 'in_context_forecast':
        Exp = Exp_In_Context_Forecast
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_multivar{}_multiscale{}_type{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.label_len,
                args.token_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.mlp_hidden_dim,
                args.mlp_hidden_layers,
                args.cosine,
                args.mix_embeds,
                args.multi_var,
                args.multi_scale,
                args.mix_embeds_type,
                args.des, ii)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting, wandb=wandb)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, wandb=wandb)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_multivar{}_multiscale{}_type{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.mlp_hidden_dim,
            args.mlp_hidden_layers,
            args.cosine,
            args.mix_embeds,
            args.multi_var,
            args.multi_scale,
            args.mix_embeds_type,
            args.des, ii)
        exp = Exp(args)  # set experiments
        exp.test(setting, test=1, wandb=wandb)
        torch.cuda.empty_cache()

    if args.wandb:
        wandb.finish()
