import argparse
import torch
from models.Preprocess_Qwen import Model

from data_provider.data_loader import Dataset_Preprocess, Dataset_Preprocess_Classification_Instruct
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M3LLM Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_ckp_dir', type=str, default='./qwen2', help='llm checkpoints dir')
    parser.add_argument('--dataset', type=str, default='weather', 
                        help='dataset to preprocess, options:[ETTh1, electricity, weather, traffic]')
    parser.add_argument('--multi_var', action='store_true', help='multi var', default=False)
    parser.add_argument('--flag', type=str, default='train', help='train/val/test')

    args = parser.parse_args()
    print(args.dataset)
    prefix = args.llm_ckp_dir.split('/')[-1]
    if args.multi_var:
        from data_provider.multivar_data_loader import Dataset_Preprocess, UEA_Dataset_Preprocess
        prefix = prefix + '_multi_var'
    model = Model(args)

    seq_len = 672
    label_len = 576
    pred_len = 96
    
    assert args.dataset in [
        'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'weather', 'traffic', 
        'classification_instruct', 'AWR', 'AF', 'BL', 'CR', 'ER', 'FM', 'RS', 'SRS2', 'SWJ', 'UWG'
    ]
    if args.dataset == 'ETTh1':
        data_set = Dataset_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTh1.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'ETTh2':
        data_set = Dataset_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTh2.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'ETTm1':
        data_set = Dataset_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTm1.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'ETTm2':
        data_set = Dataset_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTm2.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'electricity':
        data_set = Dataset_Preprocess(
            root_path='./dataset/electricity/',
            data_path='electricity.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'weather':
        data_set = Dataset_Preprocess(
            root_path='./dataset/weather/',
            data_path='weather.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'traffic':
        data_set = Dataset_Preprocess(
            root_path='./dataset/traffic/',
            data_path='traffic.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'AWR':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/ArticularyWordRecognition',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'AF':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/AtrialFibrillation',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'BL':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/Blink',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'CR':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/Cricket',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'ER':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/ERing',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'FM':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/FingerMovements',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'RS':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/RacketSports',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'SRS2':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/SelfRegulationSCP2',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'SWJ':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/StandWalkJump',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'UWG':
        data_set = UEA_Dataset_Preprocess(
            root_path='./all_datasets/Classification/UEA/UWaveGestureLibrary',
            size=[seq_len, label_len, pred_len],
            flag=args.flag
        )
    elif args.dataset == 'classification_instruct':
        data_set = Dataset_Preprocess_Classification_Instruct()
    if args.multi_var:
        data_loader = DataLoader(
            data_set,
            batch_size=1, # NOTE( ): change batch_size, must using 1, otherwise shape mismatch
            shuffle=False,
            collate_fn=lambda x: x # NOTE( ): change collate behavior of multi-var list https://github.com/pytorch/pytorch/issues/6893
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=128,
            shuffle=False,
        )

    from tqdm import tqdm
    save_dir_path = './dataset/'
    output_list = []
    for idx, data in tqdm(enumerate(data_loader)):
        output = model(data)
        output_list.append(output.detach().cpu())
    if args.multi_var:
        result = torch.stack(output_list) # bs x nvars x hidden_dim
    else:
        result = torch.cat(output_list, dim=0) # bs x hidden_dim
    print(result.shape)
    if isinstance(data_set, UEA_Dataset_Preprocess):
        torch.save(result, save_dir_path + f'/{args.dataset}.{prefix}.{args.flag}.pt')
    else:
        torch.save(result, save_dir_path + f'/{args.dataset}.{prefix}.pt')
