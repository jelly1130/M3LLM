import torch
from torch import nn, Tensor

class AttnSum(nn.Module):
    def __init__(self, 
        channel, 
        attention: int = 4, # 
        cross: bool = True, # 
        residual:bool = True, # 
        bias: bool = True, # 
        dropout: float = 0.0, # 
        gain = 0.1, # 
    ):
        super(AttnSum, self).__init__()
        self.cross = cross
        self.x_attention = nn.Sequential(
                nn.Linear(channel, channel//attention, bias=False),
                nn.ReLU(True),
                nn.Linear(channel//attention, channel, bias=False),
                nn.Sigmoid()
            )
        self.y_attention = nn.Sequential(
                nn.Linear(channel, channel//attention, bias=False),
                nn.ReLU(True),
                nn.Linear(channel//attention, channel, bias=False),
                nn.Sigmoid()
            )
        
        self.residual = residual
        if self.residual:
            self.linear = nn.Sequential(
                    nn.Linear(channel, channel, bias=bias),
                    nn.Dropout(dropout, True),
                )

        torch.nn.init.xavier_uniform_(self.x_attention[-2].weight, gain)
        torch.nn.init.xavier_uniform_(self.y_attention[-2].weight, gain)


    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.cross:
            y_attn = self.x_attention(x)
            x_attn = self.y_attention(y)
        else:
            x_attn = self.y_attention(x)
            y_attn = self.x_attention(y)

        if self.residual:
            proj_x = x + self.linear(x) * x_attn
            proj_y = y + self.linear(y) * y_attn
        else:
            proj_x = 2 * x * x_attn
            proj_y = 2 * y * y_attn
        return proj_x + proj_y


if __name__ == "__main__":
    from torch import nn
    x = torch.randn(32, 8, 512)
    y = torch.randn(32, 8, 512)
    model = AttnSum(512)
    out = model(x, y)
    print(out.shape)