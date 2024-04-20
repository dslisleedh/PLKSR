import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.backends import cudnn
import tqdm

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import argparse


class LKConv(nn.Module):
    def __init__(self, dim, ks):
        super(LKConv, self).__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, ks, padding=ks // 2, bias=True)

    def forward(self, x):
        x[:, :self.dim, :, :] = self.conv(x[:, :self.dim, :, :])
        return x


class SKSeqConv(nn.Module):
    def __init__(self, dim, ks, with_act):
        super(SKSeqConv, self).__init__()
        self.dim = dim
        n_convs = ks // 2

        if with_act:
            self.convs = nn.Sequential(*[nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=True), nn.GELU()) for _ in
                                         range(n_convs - 1)] + [nn.Conv2d(dim, dim, 3, padding=1, bias=True)])
        else:
            self.convs = nn.Sequential(*[nn.Conv2d(dim, dim, 3, padding=1, bias=True) for _ in range(n_convs)])

    def forward(self, x):
        x[:, :self.dim, :, :] = self.convs(x[:, :self.dim, :, :])
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=str, default='HD')
    parser.add_argument('--scale_factor', type=int, default=2)

    args = parser.parse_args()

    cudnn.benchmark = True

    scale_factor = args.scale_factor

    full_dim = 64
    dim = 16
    ks = 22

    test_size = args.test_size
    if test_size == 'HD':
        h, w = 720, 1280
    elif test_size == 'FHD':
        h, w = 1080, 1920
    elif test_size == 'QHD':
        h, w = 1440, 2560
    elif test_size == '4K':
        h, w = 2160, 3840
    elif test_size == '8K':
        h, w = 4320, 7680

    h, w = h // scale_factor, w // scale_factor
    x = torch.FloatTensor(1, full_dim, h, w).uniform_(0., 1.)

    res_dict = {}
    for ks_ in range(5, ks + 1, 2):
        for conv in ['lk', 'sk', 'sk_act']:
            module = LKConv(dim, ks_) if conv == 'lk' else SKSeqConv(dim, ks_, conv == 'sk_act')

            repetitions = 1000

            module = module.cuda()
            x = x.cuda()

            torch.cuda.synchronize()

            # warm up
            print('warm up ...\n')
            with torch.cuda.amp.autocast():
                with torch.inference_mode():
                    for _ in range(100):
                        _ = module(x)

            torch.cuda.reset_peak_memory_stats()

            # testing CUDA Event
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # initialize
            timings = np.zeros((repetitions, 1))

            print('testing ...\n')
            with torch.cuda.amp.autocast():
                with torch.inference_mode():
                    for rep in tqdm.tqdm(range(repetitions)):
                        starter.record()
                        _ = module(x)
                        ender.record()
                        torch.cuda.synchronize()  # wait for ending
                        curr_time = starter.elapsed_time(ender)  # from starter to ender (/ms)
                        timings[rep] = curr_time

            avg = timings.sum() / repetitions
            med = np.median(timings)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print('\navg={}\n'.format(avg))
            print('med={}\n'.format(med))
            print('fps={}\n'.format(1000 / avg))
            print('max_allocated_vram={}\n'.format(torch.cuda.max_memory_allocated() / 1024 / 1024))
            print('params={}\n'.format(params))

            if ks_ not in res_dict:
                res_dict[ks_] = {}
            res_dict[ks_][conv] = {'avg': avg, 'med': med,
                                   'max_allocated_vram': torch.cuda.max_memory_allocated() / 1024 / 1024,
                                   'params': params}

    print(res_dict)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    c = ['tab:blue', 'tab:orange', 'tab:green']
    # line as ks change
    ax1_2 = ax[1].twinx()
    for i, conv in enumerate(['sk_act', 'sk', 'lk']):
        ax[0].plot([res_dict[ks][conv]['med'] for ks in range(5, ks + 1, 2)], color=c[i], linestyle='--', marker='o',
                   markersize=10)
        if conv != 'sk_act':
            ax[1].plot(
                [res_dict[ks][conv]['max_allocated_vram'] for ks in range(5, ks + 1, 2)], color=c[i], linestyle='--',
                marker='o',
                label='A Large Kernel $\mathbf{MGO}$' if conv == 'lk' else 'Sequential 3x3 Kernels $\mathbf{MGO}$',
                markersize=10
            )
            ax1_2.plot(
                [res_dict[ks][conv]['params'] / 1000 for ks in range(5, ks + 1, 2)], color=c[i], linestyle='--',
                marker='x',
                label='A Large Kernel $\mathbf{Params}$' if conv == 'lk' else 'Sequential 3x3 Kernels $\mathbf{Params}$',
                markersize=10
            )

    ax[0].set_title('Latency', fontsize=18)
    ax[1].set_title('MGO and Parameters', fontsize=18)

    ax[0].set_xlabel('kernel size', fontsize=16)
    ax[1].set_xlabel('kernel size', fontsize=16)

    ax[0].set_ylabel('Latency (ms)', fontsize=16)
    ax[1].set_ylabel('MGO (mb)', fontsize=16)
    ax1_2.set_ylabel('Parameters (K)', fontsize=16, rotation=270, labelpad=15)

    ax[0].set_xticks(range(0, len(range(5, ks + 1, 2)), 1))
    ax[1].set_xticks(range(0, len(range(5, ks + 1, 2)), 1))

    ax[0].set_xticklabels(range(5, ks + 1, 2), fontsize=12)
    ax[1].set_xticklabels(range(5, ks + 1, 2), fontsize=12)

    ax[1].set_ylim(80, 90)
    ax1_2.set_ylim(0, 120)

    ax[0].legend(['Sequential 3x3 Kernels with GELU', 'Sequential 3x3 Kernels', 'A Large Kernel'], prop={'size': 16})
    lines, labels = ax[1].get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax[1].legend(
        lines + lines2, labels + labels2,
        loc="upper left",
        prop={'size': 16}
    )

    plt.tight_layout()
    plt.savefig('time_large_vs_seq.png', dpi=300)
