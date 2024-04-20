# PLKSR: Partial Large Kernel CNNs for Efficient Super-Resolution
-------
This repository is an official implementation of the paper "Partial Large Kernel CNNs for Efficient Super-Resolution", Arxiv, 2024.

by Dongheon Lee, Seokju Yun, and Youngmin Ro

[[paper]](https://arxiv.org/abs/2404.11848)

## Installation
```bash
git clone https://github.com/dslisleedh/PLKSR.git
cd PLKSR
conda create -n plksr python=3.10
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
python setup.py develop
```
