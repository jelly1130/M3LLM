import torch
import os
import sys
import matplotlib.pyplot as plt

filename = sys.argv[1]
param = torch.load(filename + '/checkpoint.pth')
print(param.keys())

for k in param.keys():
    print(k, param[k].shape)
    plt.figure(figsize=(20, 10))
    plt.hist(param[k].flatten().cpu().numpy(), bins=100)
    plt.title(f"{k} of {filename}")
    plt.savefig(f"vis_fig/{k}_{filename.split('/')[1]}.png")
    plt.close()
    