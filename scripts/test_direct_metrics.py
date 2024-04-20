import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.backends import cudnn

cudnn.benchmark = True

"""
Unfortunatelly, MGO is higly dependent on initialization that can't not be controlled.
(At least in my env[torch 2.1.0 and cuda 12.1 with RTX 4090 GPU])
So, we recommend run this multiple times.
"""


def test_direct_metrics(model, input_shape, n_repeat=100, use_float16=True):
    print(f'CUDNN Benchmark: {cudnn.benchmark}')
    if use_float16:
        context = torch.cuda.amp.autocast
        print('Using AMP(FP16) for testing ...')
    else:
        context = nullcontext
        print('Using FP32 for testing ...')

    x = torch.FloatTensor(*input_shape).uniform_(0., 1.)
    x = x.cuda()
    print(f'Input shape: {x.shape}')
    model = model.cuda()
    model.eval()

    with context():
        with torch.inference_mode():
            print('warmup ...')
            for _ in tqdm.tqdm(range(100)):
                model(x)  # Make sure CUDNN to find proper algorithms, especially for convolutions.
                torch.cuda.synchronize()

            print('testing ...')
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = np.zeros((n_repeat, 1))

            for rep in tqdm.tqdm(range(n_repeat)):
                starter.record()
                model(x)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

    avg = np.sum(timings) / n_repeat
    med = np.median(timings)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('------------ Results ------------')
    print(f'Average time: {avg:.5f} ms')
    print(f'Median time: {med:.5f} ms')
    print(f'Maximum GPU memory Occupancy: {torch.cuda.max_memory_allocated() / 1024 ** 2:.5f} MB')
    print(f'Params: {params / 1000}K')  # For convenience and sanity check.
    print('---------------------------------')
