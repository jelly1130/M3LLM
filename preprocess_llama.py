import argparse
import torch
from models.Preprocess_Llama import Model

from data_provider.data_loader import Dataset_Preprocess
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M3LLM Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_ckp_dir', type=str, default='./llama2', help='llm checkpoints dir')
    parser.add_argument('--dataset', type=str, default='weather', 
                        help='dataset to preprocess, options:[ETTh1, electricity, weather, traffic]')
    parser.add_argument('--multi_var', action='store_true', help='multi var', default=False)

    args = parser.parse_args()
    print(args.dataset)
    prefix = args.llm_ckp_dir.split('/')[-1]
    if args.multi_var:
        from data_provider.multivar_data_loader import Dataset_Preprocess
        prefix = prefix + '_multi_var'
    model = Model(args)

    seq_len = 672
    label_len = 576
    pred_len = 96
    
    assert args.dataset in ['ETTh1', 'electricity', 'weather', 'traffic']
    if args.dataset == 'ETTh1':
        data_set = Dataset_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTh1.csv',
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
    if args.multi_var:
        data_loader = DataLoader(
            data_set,
            batch_size=1, # change batch_size, must using 1, otherwise shape mismatch
            shuffle=False,
            collate_fn=lambda x: x # change collate behavior of multi-var list 
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=128,
            shuffle=False,
        )

    from tqdm import tqdm
    print(len(data_set.data_stamp))
    print(data_set.tot_len)
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
    torch.save(result, save_dir_path + f'/{args.dataset}.{prefix}.pt')
