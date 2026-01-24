'''
Author: Bolun Cai (bolun.cai@shopee.com)
Date: 2022-08-11 10:10:39
LastEditTime: 2023-05-08 08:15:17
Copyright (c) 2022 by Shopee, All Rights Reserved.
'''
import torch
from torch import nn, Tensor
from typing import Callable, List, Optional, Sequence, Tuple, Union

class BaseBlock(nn.Module):
    def __init__(self, 
        in_channel, 
        out_channel, 
        attention: int = 0,
        resdual: bool = False,
        bias: bool = True,
        dropout: float = 0.0
    ):
        super(BaseBlock, self).__init__()
        self.attention = None
        if attention > 0:
            self.attention = nn.Sequential(
                nn.Linear(out_channel, out_channel//attention, bias=False),
                nn.ReLU(True),
                nn.Linear(out_channel//attention, out_channel, bias=False),
                nn.Sigmoid()
            )
        self.resdual = resdual
        self.project = None
        if in_channel != out_channel and resdual == True:
            self.project = nn.Linear(in_channel, out_channel, bias=True)
        self.linear = nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=bias),
                nn.Dropout(dropout, True),
                nn.BatchNorm1d(out_channel)
            )
        self.relu = nn.ReLU(True)
        print(f'{in_channel=}, {out_channel=}, {attention=}, {resdual=}, {bias=}, {dropout=}')

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        if self.attention is not None:
            out = out * self.attention(out)
        if self.resdual:
            if self.project is not None:
                residual = self.project(x)
            else:
                residual = x
            out += residual
        out = self.relu(out)
        return out


class MLPFusion(nn.Module):
    def __init__(
        self,
        in_channels: Union[List[int], int],
        hidden_channels: List[int],
        attention_reduce: int = 4,
        resdual: bool = True,
        bias: bool = True,
        dropout: float = 0.0
    ):
        super(MLPFusion, self).__init__()

        layers = []
        if isinstance(in_channels, int):
            in_dim = in_channels
        elif isinstance(in_channels, list):
            in_dim = sum(in_channels)

        for hidden_dim in hidden_channels:
            layers.append(BaseBlock(in_dim, hidden_dim, attention_reduce, resdual, bias, dropout))
            in_dim = hidden_dim
        self.output_dim = hidden_channels[-1]
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: Union[List[Tensor], Tensor]) -> Tensor:
        if isinstance(embeddings, list):
            concatenated_in = torch.cat(embeddings, dim=-1)
        else:
            concatenated_in = embeddings
        ori_shape = concatenated_in.shape
        if len(ori_shape) == 3:
            concatenated_in = concatenated_in.reshape(-1, concatenated_in.shape[-1])
            self.mlp(concatenated_in)
            return self.mlp(concatenated_in).reshape(ori_shape[0], ori_shape[1], -1)
        else:
            return self.mlp(concatenated_in)


if __name__ == "__main__":
    from torch import nn
    x = [torch.randn(32, 8, 512), torch.randn(32, 8, 512)]
    model = nn.Sequential(
        MLPFusion([512, 512], [512, 512, 512, 512], 4, True),
        nn.Linear(512, 1000)
    )
    print(model)
    y = model(x)
    print(y.shape)