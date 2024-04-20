import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.backends import cudnn
import tqdm

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=str, default='HD')
    parser.add_argument('--scale_factor', type=int, default=2)

    args = parser.parse_args()

    cudnn.benchmark = True

    dim = 65
    start = 4
    stride = 4
    pdims = list(range(start, dim, stride))
    kernels = [3, 5, 7, 9, 11, 13, 15, 17]
    cm.plasma_r(np.linspace(0, 1, len(kernels)))
    res_dict = {}
    for pdim in pdims:
        for kernel_size in kernels:
            module = nn.Conv2d(pdim, pdim, kernel_size, padding=kernel_size // 2)
            print(module.weight.shape)

            test_size = args.test_size
            scale_factor = args.scale_factor

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
            x = torch.FloatTensor(1, pdim, h, w).uniform_(0., 1.)

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
            print('\navg={}\n'.format(avg))
            print('med={}\n'.format(med))
            print('fps={}\n'.format(1000 / avg))
            print('max_allocated_vram={}\n'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))

            res_dict[pdim] = res_dict.get(pdim, {})
            res_dict[pdim][f'ks_{kernel_size}'] = {'avg': avg, 'med': med,
                                                   'max_allocated_vram': torch.cuda.max_memory_allocated() / 1024 / 1024}

    import json

    print(json.dumps(res_dict, indent=4))
    # save json
    with open('time_test_pconv_{}_{}.json'.format(args.test_size, args.scale_factor), 'w') as f:
        json.dump(res_dict, f)

    fig, ax = plt.subplots(1, 2, figsize=(25, 5))
    kernels = [3, 5, 7, 9, 11, 13, 15, 17]
    colors = cm.plasma_r(np.linspace(0, 1, len(kernels)))

    for color, kernel_size in zip(colors, kernels):
        meds = [res_dict[pdim][f'ks_{kernel_size}']['med'] for pdim in pdims]
        max_allocated_vrams = [res_dict[pdim][f'ks_{kernel_size}']['max_allocated_vram'] for pdim in pdims]
        ax[0].plot(pdims, meds, color=color, linestyle='--', marker='o', markersize=5)
        ax[1].plot(pdims, max_allocated_vrams, color=color, linestyle='--', marker='o')

    ax[0].set_title('Latency vs Kernel Size vs Number of Input Channels', fontsize=18)
    ax[1].set_title('Memory Cost (MB)', fontsize=18)

    ax[0].set_xticks(pdims)
    ax[1].set_xticks(pdims)

    min_val_med = min([res_dict[pdim][f'ks_{kernel_size}']['med'] for pdim in pdims for kernel_size in kernels])
    max_val_med = max([res_dict[pdim][f'ks_{kernel_size}']['med'] for pdim in pdims for kernel_size in kernels])
    ax[0].set_ylim(min_val_med * 0.9, max_val_med * 1.1)
    min_val_max_allocated_vram = min(
        [res_dict[pdim][f'ks_{kernel_size}']['max_allocated_vram'] for pdim in pdims for kernel_size in kernels])
    max_val_max_allocated_vram = max(
        [res_dict[pdim][f'ks_{kernel_size}']['max_allocated_vram'] for pdim in pdims for kernel_size in kernels])
    ax[1].set_ylim(min_val_max_allocated_vram * 0.9, max_val_max_allocated_vram * 1.1)

    ax[0].set_xlabel('Number of Input Channels', fontsize=14)
    ax[0].set_ylabel('Latency (ms)', fontsize=14)

    plt.savefig('time_test_pconv_{}_{}.png'.format(args.test_size, args.scale_factor))

    # additional colorbar for the plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(2, 4), dpi=150)
    sm = cm.ScalarMappable(cmap=cm.plasma_r, norm=plt.Normalize(3, 17))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_ticks([3, 5, 7, 9, 11, 13, 15, 17])
    # colorbar yticks
    plt.savefig('time_test_pconv_colorbar.png')
