# PLKSR: Partial Large Kernel CNNs for Efficient Super-Resolution
-------
This repository is an official implementation of the paper "Partial Large Kernel CNNs for Efficient Super-Resolution", Arxiv, 2024.

by Dongheon Lee, Seokju Yun, and Youngmin Ro

[[paper]](https://arxiv.org/abs/2404.11848) [[pretrained models]](https://drive.google.com/drive/u/1/folders/1lIkZ00y9cRQpLU9qmCIB2XtS-2ZoqKq8)

## Installation
```bash
git clone https://github.com/dslisleedh/PLKSR.git
cd PLKSR
conda create -n plksr python=3.10
conda activate plksr
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
python setup.py develop
```

## Train
### Single GPU
```bash
python plksr/train.py -opt=$CONFIG_PATH
```

## Test
```bash
python plksr/test.py -opt=$CONFIG_PATH
```
## Results

<details>
<summary>Quantitative Results</summary>

### Main model
![image](https://github.com/dslisleedh/PLKSR/blob/main/figs/Quantitative.png)
### Tiny model
![image](https://github.com/dslisleedh/PLKSR/blob/main/figs/Quantitative_tiny.png)
</details>

<details>
<summary>Visual Results</summary>

![image](https://github.com/dslisleedh/PLKSR/blob/main/figs/Qualitative_1.png)
![image](https://github.com/dslisleedh/PLKSR/blob/main/figs/Qualitative_2.png)
  
</details>

## Acknowledgement
This work is released under the MIT license. The codes are based on BasicSR. Thanks for their awesome works.

## Contact
If you have any questions, please contact dslisleedh@gmail.com
