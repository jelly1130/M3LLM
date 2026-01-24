import torch
import torch.nn as nn
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
from layers.mlp import MLP
from layers.Autoformer_EncDec import series_decomp
import numpy as np
from models import attn_sum
import functools

class Bicubic(torch.nn.Module):
    def __init__(self, down_sampling_window):
        super().__init__()
        self.down_sampling_window = down_sampling_window

    def forward(self, x):
        x_4d = x.unsqueeze(2)
        y_4d = torch.nn.functional.interpolate(x_4d, mode='bicubic', scale_factor=[1, 1/self.down_sampling_window])
        y_3d = y_4d.squeeze()
        return y_3d

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        # 确保输入的形状符合 MultiheadAttention 的要求
        assert query.dim() == 3 and key.dim() == 3 and value.dim() == 3, "Input tensors should have shape (batch_size, seq_len, embed_dim)"
        
        # 计算交叉注意力
        attn_output, _ = self.multihead_attn(query, key, value)
        
        return attn_output

class MultiVarCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiVarCrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).

    def forward(self, input_tensor):
        batch_size, n_vars, token_num, token_len = input_tensor.shape

        all_attn_outputs = []
        for i in range(n_vars):
            query = input_tensor[:, i, :, :]  # Shape: [batch_size, token_num, token_len]
            key = value = input_tensor.reshape(batch_size, -1, token_len)  # Reshape to [batch_size, n_vars * token_num, token_len]
            # print(query.shape, key.shape, value.shape)
            attn_output, _ = self.multihead_attn(query, key, value)
            all_attn_outputs.append(attn_output)
        # for output in all_attn_outputs:
        #     print(output.shape)
        all_attn_outputs = torch.stack(all_attn_outputs, dim=1)  # Shape: [batch_size, n_vars, query_len, embed_dim]

        return all_attn_outputs
    
class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.down_sampling_window = configs.down_sampling_window
        # NOTE( ): temporary config
        d_model = configs.n_vars
        d_ff = configs.n_vars * 2
        dropout = 0.1
        moving_avg = configs.moving_avg
        channel_independence = 0
        decomp_method = 'moving_avg'

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence

        if decomp_method == 'moving_avg':
            self.decompsition = series_decomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError(f'unknown decompsition method: {decomp_method}')
        
        self.cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.token_len = configs.token_len
        self.pdm_blocks_num = configs.pdm_blocks_num
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(f'Build model on {self.device}')
        
        if self.task_name == 'classification':
            # self.classification_instruct = torch.load('dataset/classification.pt').to(self.device)
            self.classification_instruct = None
        
        self.llm = Qwen2ForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
        )
        self.hidden_dim_of_llm = self.llm.config.hidden_size
        print(f'{self.hidden_dim_of_llm=}')
        self.mix = configs.mix_embeds
        self.mix_embeds_type = configs.mix_embeds_type
        self.n_vars = configs.n_vars
        self.multi_scale = configs.multi_scale
        if self.multi_scale:
            self.multi_scale_type = configs.multi_scale_type
            self.multi_scale_num = configs.multi_scale_num
            self.down_sampling_layers = configs.down_sampling_layers
            self.down_sampling_window = configs.down_sampling_window
            self.down_sampling_method = configs.down_sampling_method
            self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                    for _ in range(1)]) # NOTE( ): using 1 pdm blocks
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
            if self.mix_embeds_type == 'v6':
                del self.add_scale # unused parameter
            if self.mix_embeds_type in ['v8', 'v9', 'v10', 'v11', 'v13']:
                del self.add_scale
                self.add_scale1 = nn.Parameter(torch.zeros([]))
                self.add_scale2 = nn.Parameter(torch.zeros([]))
                hidden_dim_for_cross = int(max(np.sqrt(self.n_vars) + 1, 2))
                if self.mix_embeds_type == 'v8':
                    self.hidden_vector_for_cross = torch.randn(self.n_vars, hidden_dim_for_cross).to(self.device)
                elif self.mix_embeds_type in ['v9', 'v11', 'v13']:
                    self.hidden_vector_for_cross_list = []
                    for i in range(self.n_vars):
                        if self.mix_embeds_type in ['v9', 'v11']:
                            self.hidden_vector_for_cross_list.append(torch.randn(self.n_vars, hidden_dim_for_cross).to(self.device))
                        elif self.mix_embeds_type == 'v13':
                            self.hidden_vector_for_cross_list.append(torch.randn(self.n_vars, self.hidden_dim_of_llm).to(self.device))
                    if self.mix_embeds_type in ['v11', 'v13']:
                        self.fusion = attn_sum.AttnSum(self.hidden_dim_of_llm)
                elif self.mix_embeds_type == 'v10':
                    self.hidden_vector_for_cross_list = []
                    self.hidden_linear_list = []
                    for i in range(self.n_vars):
                        self.hidden_linear_list.append(nn.Linear(self.n_vars, 1).to(self.device))
                        self.hidden_vector_for_cross_list.append(torch.randn(self.n_vars, hidden_dim_for_cross).to(self.device))
            if self.mix_embeds_type == 'v2':
                self.mix_linear = nn.Linear(self.hidden_dim_of_llm * self.n_vars, self.hidden_dim_of_llm) # hidden_dim_of_llm * n_vars -> hidden_dim_of_llm
            elif self.mix_embeds_type == 'v3':
                self.mix_mlp = MLP(self.hidden_dim_of_llm * self.n_vars, self.hidden_dim_of_llm, 
                                self.hidden_dim_of_llm, 2, 
                                configs.dropout, configs.mlp_activation)
            elif self.mix_embeds_type == 'v4':
                self.mix_attn = MultiVarCrossAttention(self.hidden_dim_of_llm, 1) # embed_dim must be divisible by num_heads
            elif self.mix_embeds_type in ['v5', 'v7']:
                self.mix_attn = CrossAttention(self.hidden_dim_of_llm * self.n_vars, self.n_vars) # embed_dim must be divisible by num_heads
        
        self.multi_var = False
        if configs.multi_var:
            self.multi_var = True
        
        for name, param in self.llm.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llm)
            self.decoder = nn.Linear(self.hidden_dim_of_llm, self.token_len)
            if self.multi_var and self.mix_embeds_type in ['v2', 'v3']:
                self.decoder = nn.Linear(self.hidden_dim_of_llm, self.token_len * self.n_vars) # NOTE( ):这里的decode难度就比原来要高了
            if self.task_name == 'classification':
                self.decoder = nn.Linear(self.hidden_dim_of_llm * self.n_vars, configs.num_class)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_llm, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_llm, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation)
            if self.multi_var and self.mix_embeds_type in ['v2', 'v3']:
                self.decoder = MLP(self.hidden_dim_of_llm, self.token_len * self.n_vars,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation) # NOTE( ):这里的decode难度就比原来要高了
            if self.task_name == 'classification':
                self.decoder = MLP(self.hidden_dim_of_llm * self.n_vars, configs.num_class,
                                   configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                                   configs.dropout, configs.mlp_activation)
    
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'bicubic':
            down_pool = Bicubic(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            enc_in = x_enc.shape[-1]
            down_pool = nn.Conv1d(in_channels=enc_in, out_channels=enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        __mydebug__ = False
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        if __mydebug__: print(f'0 {stdev.shape=}')
        if __mydebug__: print(f'1 {x_enc.shape=}') # torch.Size([256, 672, 1])
        bs, _, n_vars = x_enc.shape
        if self.multi_scale:
            x_list, _ = self.__multi_scale_process_inputs(x_enc, None)
            if __mydebug__: print(f'1.1 {[x.shape for x in x_list]=}')
            for pdm_block in self.pdm_blocks:
                x_list = pdm_block(x_list)
            x_enc = x_list[0]
        # x_enc: [bs x nvars x seq_len]    
        x_enc = x_enc.permute(0, 2, 1)
        if __mydebug__: print(f'2 {x_enc.shape=}') # torch.Size([256, 1, 672])
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        if __mydebug__: print(f'3 {x_enc.shape=}') # torch.Size([256, 672])
        
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        if __mydebug__: print(f'4 {fold_out.shape=}') # [bs * n_vars, token_num, token_len]
        
        token_num = fold_out.shape[1]
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_llm]
        times_embeds = self.encoder(fold_out)
        if __mydebug__: print(f'4.01 {times_embeds.shape=}') # [bs * n_vars, token_num, hidden_dim_of_llm]
        if self.mix:
            if self.mix_embeds_type in ['v1', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v13']:
                if self.mix_embeds_type in ['v1', 'v6', 'v8', 'v9', 'v10', 'v11', 'v13']:
                    # x_mark_enc: [bs x token_num x n_vars x hidden_dim_of_llm]
                    if self.multi_var:
                        x_mark_enc = x_mark_enc.permute(0, 2, 1, 3) # x_mark_enc: [bs x n_vars x token_num x hidden_dim_of_llm]
                        x_mark_enc = x_mark_enc.reshape(bs * n_vars, token_num, -1) # x_mark_enc: [bs * n_vars x token_num x hidden_dim_of_llm]
                    times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
                    x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
                    if __mydebug__: print(f'{x_mark_enc.shape=} {times_embeds.shape=} {self.multi_var=}')
                    if self.mix_embeds_type == 'v1':
                        times_embeds = times_embeds + self.add_scale * x_mark_enc
                    elif self.mix_embeds_type == 'v6':
                        times_embeds = times_embeds * x_mark_enc
                    elif self.mix_embeds_type in ['v8', 'v9', 'v10', 'v11', 'v13']:
                        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_llm]
                        ori_times_embeds = times_embeds
                        times_embeds = times_embeds.reshape(bs, n_vars, token_num, -1) # [bs x n_vars x token_num x hidden_dim_of_llm]
                        times_embeds = times_embeds.permute(0, 2, 3, 1) # [bs x token_num x hidden_dim_of_llm x n_vars]
                        if self.mix_embeds_type == 'v8':
                            times_embeds_multi_v = torch.matmul(times_embeds, self.hidden_vector_for_cross) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                            sum_square = torch.pow(times_embeds_multi_v, 2) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                            square_sum = torch.matmul(torch.pow(times_embeds, 2), torch.pow(self.hidden_vector_for_cross, 2)) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                            times_embeds = 0.5 * torch.sum(sum_square - square_sum, dim=3, keepdim=True) # [bs x token_num x hidden_dim_of_llm x 1]
                            times_embeds = times_embeds.permute(0, 1, 3, 2) # [bs x token_num x 1 x hidden_dim_of_llm]
                            times_embeds = times_embeds.repeat_interleave(n_vars, dim=2) # [bs x token_num x n_vars x hidden_dim_of_llm]
                        elif self.mix_embeds_type == ['v9', 'v11']: # MoE
                            times_embeds_output = []
                            for hidden_vector in self.hidden_vector_for_cross_list:
                                if __mydebug__: print(f'v9:0 {hidden_vector.shape=}, {times_embeds.shape=}')
                                times_embeds_multi_v = torch.matmul(times_embeds, hidden_vector) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                                sum_square = torch.pow(times_embeds_multi_v, 2) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                                square_sum = torch.matmul(torch.pow(times_embeds, 2), torch.pow(hidden_vector, 2)) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                                cross_output = 0.5 * torch.sum(sum_square - square_sum, dim=3, keepdim=True) # [bs x token_num x hidden_dim_of_llm x 1]
                                cross_output = cross_output.permute(0, 1, 3, 2) # [bs x token_num x 1 x hidden_dim_of_llm]
                                times_embeds_output.append(cross_output)
                            times_embeds = torch.cat(times_embeds_output, dim=2) # [bs x token_num x n_vars x hidden_dim_of_llm]
                        elif self.mix_embeds_type == 'v10': # MoE with 1st order factor
                            times_embeds_output1 = []
                            for linear_layer in self.hidden_linear_list:
                                times_embeds_output1.append(linear_layer(times_embeds.reshape(bs, n_vars, token_num, -1).permute(0, 2, 3, 1)))
                            times_embeds1 = torch.cat(times_embeds_output1, dim=3) # [bs x token_num x hidden_dim_of_llm x n_vars]
                            times_embeds1 = times_embeds1.permute(0, 1, 3, 2) # [bs x token_num x n_vars x hidden_dim_of_llm]
                            times_embeds_output2 = []
                            for hidden_vector in self.hidden_vector_for_cross_list:
                                times_embeds_multi_v = torch.matmul(times_embeds, hidden_vector) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                                sum_square = torch.pow(times_embeds_multi_v, 2) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                                square_sum = torch.matmul(torch.pow(times_embeds, 2), torch.pow(hidden_vector, 2)) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                                cross_output = 0.5 * torch.sum(sum_square - square_sum, dim=3, keepdim=True) # [bs x token_num x hidden_dim_of_llm x 1]
                                cross_output = cross_output.permute(0, 1, 3, 2) # [bs x token_num x 1 x hidden_dim_of_llm]
                                times_embeds_output2.append(cross_output)
                            times_embeds2 = torch.cat(times_embeds_output2, dim=2) # [bs x token_num x n_vars x hidden_dim_of_llm]
                            times_embeds = times_embeds1 + times_embeds2
                        elif self.mix_embeds_type == 'v13':
                            times_embeds_output = []
                            for hidden_vector in self.hidden_vector_for_cross_list:
                                if __mydebug__: print(f'v13:0 {hidden_vector.shape=}, {times_embeds.shape=}')
                                times_embeds_multi_v = times_embeds * hidden_vector # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                                sum_square = torch.pow(times_embeds_multi_v, 2) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                                square_sum = torch.pow(times_embeds, 2) * torch.pow(hidden_vector, 2) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                                cross_output = 0.5 * torch.sum(sum_square - square_sum, dim=3, keepdim=True) # [bs x token_num x hidden_dim_of_llm x 1]
                                cross_output = cross_output.permute(0, 1, 3, 2) # [bs x token_num x 1 x hidden_dim_of_llm]
                                times_embeds_output.append(cross_output)
                            times_embeds = torch.cat(times_embeds_output, dim=2) # [bs x token_num x n_vars x hidden_dim_of_llm]
                        times_embeds = times_embeds.permute(0, 2, 1, 3) # [bs x n_vars x token_num x hidden_dim_of_llm]
                        times_embeds = times_embeds.reshape(bs * n_vars, token_num, -1) # [bs * n_vars x token_num x hidden_dim_of_llm]
                        if self.mix_embeds_type in ['v11', 'v13']:
                            # times_embeds = self.fusion([ori_times_embeds, self.add_scale1 * times_embeds, self.add_scale2 * x_mark_enc])
                            # times_embeds = self.fusion(ori_times_embeds, self.add_scale1 * times_embeds + self.add_scale2 * x_mark_enc)
                            times_embeds = self.fusion(self.add_scale1 * times_embeds, self.add_scale2 * x_mark_enc)
                            times_embeds = self.fusion(ori_times_embeds, times_embeds)
                        elif self.mix_embeds_type == 'v12':
                            times_embeds = self.fusion(self.add_scale1 * times_embeds, self.add_scale2 * ori_times_embeds)
                        else:
                            times_embeds = ori_times_embeds + self.add_scale1 * times_embeds + self.add_scale2 * x_mark_enc
                elif self.mix_embeds_type in ['v5', 'v7']:
                    times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True) # [bs * n_vars x token_num x hidden_dim_of_llm]
                    x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=3, keepdim=True) # [bs x n_vars x token_num x hidden_dim_of_llm]
                    if __mydebug__: print(f'4.02 {x_mark_enc.shape=} {times_embeds.shape=}')
                    times_embeds = times_embeds.reshape(bs, n_vars, token_num, -1) # [bs x n_vars x token_num x hidden_dim_of_llm]
                    times_embeds = times_embeds.permute(0, 2, 1, 3) # [bs x token_num x n_vars x hidden_dim_of_llm]
                    times_embeds = times_embeds.reshape(bs, token_num, -1) # [bs x token_num x n_vars * hidden_dim_of_llm]
                    
                    x_mark_enc = x_mark_enc.permute(0, 2, 1, 3) # [bs x token_num x n_vars x hidden_dim_of_llm]
                    x_mark_enc = x_mark_enc.reshape(bs, token_num, -1) # [bs x token_num x n_vars * hidden_dim_of_llm]
                    if self.mix_embeds_type == 'v5':
                        times_embeds = self.mix_attn(times_embeds, x_mark_enc, x_mark_enc) # [bs x token_num x n_vars * hidden_dim_of_llm]
                    elif self.mix_embeds_type == 'v7':
                        times_embeds = self.mix_attn(x_mark_enc, times_embeds, times_embeds)
                    times_embeds = times_embeds.reshape(bs, token_num, n_vars, -1) # [bs x token_num x n_vars x hidden_dim_of_llm]
                    times_embeds.permute(0, 2, 1, 3) # [bs x n_vars x token_num x hidden_dim_of_llm]
                    times_embeds = times_embeds.reshape(bs * n_vars, token_num, -1) # [bs * n_vars x token_num x hidden_dim_of_llm]
            elif self.mix_embeds_type in ['v2', 'v3', 'v4']: # linear, mlp, attn
                if self.multi_var:
                    times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
                    x_mark_enc = x_mark_enc.permute(0, 2, 1, 3) # x_mark_enc: [bs x n_vars x token_num x hidden_dim_of_llm]
                    x_mark_enc = x_mark_enc.reshape(bs * n_vars, token_num, -1) # x_mark_enc: [bs * n_vars x token_num x hidden_dim_of_llm]
                    if __mydebug__: print(f'4.11 {x_mark_enc.shape=}')
                    x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
                    times_embeds = times_embeds + self.add_scale * x_mark_enc
                    times_embeds = times_embeds.reshape(bs, n_vars, token_num, -1) # [bs x n_vars x token_num x hidden_dim_of_llm]
                    times_embeds.permute(0, 2, 1, 3) # [bs x token_num x n_vars x hidden_dim_of_llm]
                    times_embeds = times_embeds.reshape(bs, token_num, -1) # [bs * n_vars x token_num x n_vars * hidden_dim_of_llm]
                    if self.mix_embeds_type == 'v2':
                        if __mydebug__: print(f'4.02 {times_embeds.shape=}') 
                        times_embeds = self.mix_linear(times_embeds) # [bs x token_num x hidden_dim_of_llm], input n_vars *hidden_dim, output hidden_dim
                        if __mydebug__: print(f'4.03 {times_embeds.shape=}')
                        
                    elif self.mix_embeds_type == 'v3':
                        times_embeds = self.mix_mlp(times_embeds)
                    elif self.mix_embeds_type == 'v4':
                        times_embeds = times_embeds.reshape(bs, token_num, n_vars, -1) # [bs x token_num x n_vars x hidden_dim_of_llm]
                        times_embeds = times_embeds.permute(0, 2, 1, 3) # [bs x n_vars x token_num x hidden_dim_of_llm]
                        times_embeds = self.mix_attn(times_embeds) # [bs x n_vars x token_num x hidden_dim_of_llm]
                        times_embeds = times_embeds.reshape(bs * n_vars, token_num, -1) # [bs * n_vars x token_num x hidden_dim_of_llm]
                else:
                    raise NotImplementedError('self.mix_embeds_type == "v2" and not self.multi_var')
        if __mydebug__: print(f'4.8 {times_embeds.shape=}')
        # outputs: [bs * n_vars x token_num x hidden_dim_of_llm]
        outputs = self.llm(inputs_embeds=times_embeds, output_hidden_states=True).hidden_states[-1] # last hidden state
        if __mydebug__: print(f'4.9 {outputs.shape=}')
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs) # input hidden_dim, output nvars * token_len
        if __mydebug__: print(f'4.10 {dec_out.shape=}') # 
        if self.multi_var and self.mix_embeds_type in ['v2', 'v3']:
            dec_out = dec_out.reshape(bs, token_num, n_vars, self.token_len) # [bs x token_num x n_vars x token_len]
            dec_out = dec_out.permute(0, 2, 1, 3) # [bs x n_vars x token_num x token_len]
        dec_out = dec_out.reshape(bs, n_vars, -1)
        # dec_out: [bs x token_num * token_len x n_vars]
        dec_out = dec_out.permute(0, 2, 1)
        if __mydebug__: print(f'5 {dec_out.shape=}') # torch.Size([256, 672, 7])
        if __mydebug__: print(f'6 {stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1).shape=}') # torch.Size([256, 672, 7])
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        if __mydebug__: print(f'6.1 {stdev[:, 0, :].shape=}')
        if __mydebug__: print(f'6.2 {stdev[:, 0, :].unsqueeze(1).shape=}')
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        if __mydebug__: print(f'7 {dec_out.shape=}')
        if __mydebug__: exit(0)
        
        return dec_out
    
    
    def classification(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        __mydebug__ = False
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        if __mydebug__: print(f'0 {stdev.shape=}')
        if __mydebug__: print(f'1 {x_enc.shape=}') # torch.Size([256, 672, 1])
        bs, _, n_vars = x_enc.shape
        if self.multi_scale:
            x_list, _ = self.__multi_scale_process_inputs(x_enc, None)
            if __mydebug__: print(f'1.1 {[x.shape for x in x_list]=}')
            for pdm_block in self.pdm_blocks:
                x_list = pdm_block(x_list)
            x_enc = x_list[0]
        # x_enc: [bs x nvars x seq_len]    
        x_enc = x_enc.permute(0, 2, 1)
        if __mydebug__: print(f'2 {x_enc.shape=}') # torch.Size([256, 1, 672])
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        if __mydebug__: print(f'3 {x_enc.shape=}') # torch.Size([256, 672])
        
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        if __mydebug__: print(f'4 {fold_out.shape=}') # [bs * n_vars, token_num, token_len]
        
        token_num = fold_out.shape[1]
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_llm]
        times_embeds = self.encoder(fold_out)
        if __mydebug__: print(f'4.01 {times_embeds.shape=}') # [bs * n_vars, token_num, hidden_dim_of_llm]
        # x_mark_enc: [bs x token_num x n_vars x hidden_dim_of_llm]
        x_mark_enc = x_mark_enc.unsqueeze(1).expand(bs, token_num, n_vars, self.hidden_dim_of_llm)
        if __mydebug__: print(f'4.02 {x_mark_enc.shape=}')
        if self.multi_var:
            x_mark_enc = x_mark_enc.permute(0, 2, 1, 3) # x_mark_enc: [bs x n_vars x token_num x hidden_dim_of_llm]
            x_mark_enc = x_mark_enc.reshape(bs * n_vars, token_num, -1) # x_mark_enc: [bs * n_vars x token_num x hidden_dim_of_llm]
        times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
        x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
        if __mydebug__: print(f'{x_mark_enc.shape=} {times_embeds.shape=} {self.multi_var=}')
        if self.mix_embeds_type in ['v9', 'v10', 'v11']:
            # times_embeds: [bs * n_vars x token_num x hidden_dim_of_llm]
            ori_times_embeds = times_embeds
            times_embeds = times_embeds.reshape(bs, n_vars, token_num, -1) # [bs x n_vars x token_num x hidden_dim_of_llm]
            times_embeds = times_embeds.permute(0, 2, 3, 1) # [bs x token_num x hidden_dim_of_llm x n_vars]
            if self.mix_embeds_type == ['v9', 'v11']: # MoE
                times_embeds_output = []
                for hidden_vector in self.hidden_vector_for_cross_list:
                    if __mydebug__: print(f'v9:0 {hidden_vector.shape=}, {times_embeds.shape=}')
                    times_embeds_multi_v = torch.matmul(times_embeds, hidden_vector) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                    sum_square = torch.pow(times_embeds_multi_v, 2) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                    square_sum = torch.matmul(torch.pow(times_embeds, 2), torch.pow(hidden_vector, 2)) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                    cross_output = 0.5 * torch.sum(sum_square - square_sum, dim=3, keepdim=True) # [bs x token_num x hidden_dim_of_llm x 1]
                    cross_output = cross_output.permute(0, 1, 3, 2) # [bs x token_num x 1 x hidden_dim_of_llm]
                    times_embeds_output.append(cross_output)
                times_embeds = torch.cat(times_embeds_output, dim=2) # [bs x token_num x n_vars x hidden_dim_of_llm]
            elif self.mix_embeds_type == 'v10': # MoE with 1st order factor
                times_embeds_output1 = []
                for linear_layer in self.hidden_linear_list:
                    times_embeds_output1.append(linear_layer(times_embeds.reshape(bs, n_vars, token_num, -1).permute(0, 2, 3, 1)))
                times_embeds1 = torch.cat(times_embeds_output1, dim=3) # [bs x token_num x hidden_dim_of_llm x n_vars]
                times_embeds1 = times_embeds1.permute(0, 1, 3, 2) # [bs x token_num x n_vars x hidden_dim_of_llm]
                times_embeds_output2 = []
                for hidden_vector in self.hidden_vector_for_cross_list:
                    times_embeds_multi_v = torch.matmul(times_embeds, hidden_vector) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                    sum_square = torch.pow(times_embeds_multi_v, 2) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                    square_sum = torch.matmul(torch.pow(times_embeds, 2), torch.pow(hidden_vector, 2)) # [bs x token_num x hidden_dim_of_llm x hidden_dim_for_cross]
                    cross_output = 0.5 * torch.sum(sum_square - square_sum, dim=3, keepdim=True) # [bs x token_num x hidden_dim_of_llm x 1]
                    cross_output = cross_output.permute(0, 1, 3, 2) # [bs x token_num x 1 x hidden_dim_of_llm]
                    times_embeds_output2.append(cross_output)
                times_embeds2 = torch.cat(times_embeds_output2, dim=2) # [bs x token_num x n_vars x hidden_dim_of_llm]
                times_embeds = times_embeds1 + times_embeds2
            times_embeds = times_embeds.permute(0, 2, 1, 3) # [bs x n_vars x token_num x hidden_dim_of_llm]
            times_embeds = times_embeds.reshape(bs * n_vars, token_num, -1) # [bs * n_vars x token_num x hidden_dim_of_llm]
            if self.mix_embeds_type == 'v11':
                # times_embeds = self.fusion([ori_times_embeds, self.add_scale1 * times_embeds, self.add_scale2 * x_mark_enc])
                # times_embeds = self.fusion(ori_times_embeds, self.add_scale1 * times_embeds + self.add_scale2 * x_mark_enc)
                times_embeds = self.fusion(self.add_scale1 * times_embeds, self.add_scale2 * x_mark_enc)
                times_embeds = self.fusion(ori_times_embeds, times_embeds)
            else:
                times_embeds = ori_times_embeds + self.add_scale1 * times_embeds + self.add_scale2 * x_mark_enc
        if __mydebug__: print(f'4.8 {times_embeds.shape=}')
        if self.classification_instruct is not None:
            instruct = self.classification_instruct.unsqueeze(0) # [1 x 1 x hidden_dim_of_llm]
            instruct_expand = instruct.expand(bs * n_vars, 1, self.hidden_dim_of_llm)
            times_embeds = torch.cat([times_embeds, instruct_expand], dim=1)
        # outputs: [bs * n_vars x token_num x hidden_dim_of_llm]
        outputs = self.llm(inputs_embeds=times_embeds, output_hidden_states=True).hidden_states[-1] # last hidden state
        if __mydebug__: print(f'4.9 {outputs.shape=}')
        outputs = outputs[:,-1,:] # NOTE( ): we only use the "last" last_hidden_state [bs * n_vars x hidden_dim_of_llm]
        outputs = outputs.reshape(bs, -1) # [bs x n_vars * hidden_dim_of_llm]
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs) # input hidden_dim, output nvars * token_len
        if __mydebug__: print(f'4.10 {dec_out.shape=}') # 
        # dec_out = dec_out.reshape(bs, n_vars, -1)
        # # dec_out: [bs x token_num * token_len x n_vars]
        # dec_out = dec_out.permute(0, 2, 1)
        # if __mydebug__: print(f'5 {dec_out.shape=}') # torch.Size([256, 672, 7])
        # if __mydebug__: print(f'6 {stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1).shape=}') # torch.Size([256, 672, 7])
        # dec_out = dec_out * \
        #     (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        # if __mydebug__: print(f'6.1 {stdev[:, 0, :].shape=}')
        # if __mydebug__: print(f'6.2 {stdev[:, 0, :].unsqueeze(1).shape=}')
        # dec_out = dec_out + \
        #     (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        if __mydebug__: print(f'7 {dec_out.shape=}')
        if __mydebug__: exit(0)
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc, x_dec, x_mark_dec)