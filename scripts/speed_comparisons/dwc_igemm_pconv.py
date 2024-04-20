import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cudnn

import numpy as np

import tqdm
from typing import Literal

from functools import partial

import torch
import torch.nn as nn
import torch.utils.cpp_extension as cpp_extension


class Partial(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.idx = int(dim * 0.25)  # 16 Channel
        print(self.idx)
        self.conv = nn.Conv2d(self.idx, self.idx, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        x1, x2 = x[:, :self.idx], x[:, self.idx:]
        return torch.cat([self.conv(x1), x2], dim=1)


class DWC(nn.Conv2d):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)


if __name__ == '__main__':
    cudnn.benchmark = True
    repetitions = 100
    kernel_size = 17
    conv_type = 'gemm_dwc'  # 'dwc', 'gemm_dwc', 'partial'

    print('Current Test: kernel_size:{}'.format(kernel_size))
    print('conv_type:{}'.format(conv_type))

    h = 720
    w = 1280
    batch_size = 64
    dim = 64
    upscaling_factor = 2
    x = torch.FloatTensor(
        batch_size, dim, h // upscaling_factor, w // upscaling_factor).uniform_(0., 1.)

    if conv_type == 'partial':
        module = Partial(dim, kernel_size)
    elif conv_type == 'dwc':
        module = DWC(dim, kernel_size)
    elif conv_type == 'gemm_dwc':
        # install GEMM_DWC in other env. it raises error in PLKSR's env.
        # https://github.com/DingXiaoH/RepLKNet-pytorch
        import _depthwise_conv2d_implicit_gemm_C as _extension


        class _DepthWiseConv2dImplicitGEMMFP32(torch.autograd.Function):
            @staticmethod
            @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
            def forward(ctx, x, w):
                ctx.save_for_backward(x, w)
                return _extension.forward_fp32(x, w)

            @staticmethod
            @torch.cuda.amp.custom_bwd
            def backward(ctx, grad):
                x, w = ctx.saved_tensors
                grad = grad.contiguous()
                x = x.contiguous()
                w = w.contiguous()
                dx = _extension.backward_data_fp32(grad, w)
                dw = _extension.backward_filter_fp32(grad, x, w)
                return dx, dw


        class _DepthWiseConv2dImplicitGEMMFP16(torch.autograd.Function):
            @staticmethod
            @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
            def forward(ctx, x, w):
                ctx.save_for_backward(x, w)
                return _extension.forward_fp16(x, w)

            @staticmethod
            @torch.cuda.amp.custom_bwd
            def backward(ctx, grad):
                x, w = ctx.saved_tensors
                grad = grad.contiguous()
                x = x.contiguous()
                w = w.contiguous()
                dx = _extension.backward_data_fp16(grad, w)
                dw = _extension.backward_filter_fp16(grad, x, w)
                return dx, dw


        class DepthWiseConv2dImplicitGEMM(nn.Conv2d):
            def __init__(self, channels, kernel, bias=False):
                super().__init__(channels, channels, kernel, groups=channels, bias=bias)
                # _load_extension()

            def forward(self, x):
                if torch.is_autocast_enabled() or x.dtype == torch.float16:
                    x = _DepthWiseConv2dImplicitGEMMFP16.apply(x, self.weight)
                elif x.dtype == torch.float32:
                    x = _DepthWiseConv2dImplicitGEMMFP32.apply(x, self.weight)
                else:
                    raise TypeError("Only support fp32 and fp16, get {}".format(x.dtype))
                if self.bias is not None:
                    x = x + self.bias.to(x).view(1, -1, 1, 1)
                return x


        module = DepthWiseConv2dImplicitGEMM(dim, kernel_size)

    # warm up
    module = module.cuda()
    x = x.cuda()
    print('warm up ...\n')
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for _ in tqdm.tqdm(range(100)):
                _ = module(x)
                torch.cuda.synchronize()

    # synchronize / wait for all the GPU process then back to cpu
    torch.cuda.synchronize()

    # testing CUDA Event
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # initialize
    timings = np.zeros((repetitions, 1))

    torch.cuda.reset_peak_memory_stats()

    print('testing ...\n')
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            for rep in tqdm.tqdm(range(repetitions)):
                starter.record()
                _ = module(x)
                ender.record()
                torch.cuda.synchronize()  # wait for ending
                curr_time = starter.elapsed_time(ender)  # from starter to ender (/ms)
                timings[rep] = curr_time

    print(f'average time: {timings.mean()}')
    print(f'median time: {np.median(timings)}')
    print(f'vram_alloc: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB')
    print(f'param: {sum(p.numel() for p in module.parameters() if p.requires_grad)}')
