## Towards Holistic Modeling for Video Frame Interpolation with Auto-regressive Diffusion Transformers<br><sub>Official PyTorch implementation</sub>

**Towards Holistic Modeling for Video Frame Interpolation with Auto-regressive Diffusion Transformers**<br>

Xinyu Peng*, Han Li*, Yuyang Huang, Ziyang Zheng, Yaoming Wang, Xin Chen, Wenrui Dai, Chenglin Li, Junni Zou, Hongkai Xiong<br>

\* Equal contribution  

<p align="center">
  <a href='https://xypeng9903.github.io/ldf-vfi-web/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  &nbsp;
  <a href=''><img src="https://img.shields.io/static/v1?label=Arxiv&message=LDF-VFI(soon)&color=red&logo=arxiv"></a>
  &nbsp;
  <a href='https://huggingface.co/onecat-ai/LDF-VFI'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>
</p>

**Abstract:**
*Existing video frame interpolation (VFI) methods often adopt a frame-centric approach, processing videos as independent short segments (e.g., triplets), which leads to temporal inconsistencies and motion artifacts. To overcome this, we propose a holistic, video-centric paradigm named **L**ocal **D**iffusion **F**orcing for **V**ideo **F**rame **I**nterpolation (LDF-VFI). Our framework is built upon an auto-regressive diffusion transformer that models the entire video sequence to ensure long-range temporal coherence. To mitigate error accumulation inherent in auto-regressive generation, we introduce a novel skip-concatenate sampling strategy that effectively maintains temporal stability. Furthermore, LDF-VFI incorporates sparse, local attention and tiled VAE encoding, a combination that not only enables efficient processing of long sequences but also allows generalization to arbitrary spatial resolutions (e.g., 4K) at inference without retraining. An enhanced conditional VAE decoder, which leverages multi-scale features from the input video, further improves reconstruction fidelity. Empirically, LDF-VFI achieves state-of-the-art performance on challenging long-sequence benchmarks, demonstrating superior per-frame quality and temporal consistency, especially in scenes with large motion.*

## Installation

### Environment

- Python 3.11 and PyTorch 2.5.1
- Python libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Pretrained model

- Download the pretrained [LDF-VFI model](https://huggingface.co/onecat-ai/LDF-VFI)


## Quick Start

Run the following command to perform 8x frame interpolation on `assets/demo.mp4` (requires ~20GB GPU memory):

```bash
bash quick_start/generate.sh
```

\* Please modify the following variables in the scripts according to your evaluation environment.

- `model_path`: modify to `/path/to/LDF-VFI/transformer`
- `vae_path`: modify to `/path/to/LDF-VFI/Wan2.1_VAE_cond_v2.pth`

If you want to perform frame interpolation on your own video, modify:

- `data_path`: path to a video file or a folder of videos
- `temporal_sf`: the temporal scaling factor, currently supports 1 to 16.

**Distributed inference**

The script also enables accelerated inference or supports higher resolution on multiple GPUs. If multiple GPUs are available, please modify `distributed_args` according to your distributed environment. The videos in the `data_path` (assuming it is a folder of videos) will be automatically distributed on multiple GPUs. In addition, increasing the sequence parallel size (`--sp_size` in `performance_args`, support 1, 2, 4) enables accelerated inference and higher resolution under limited memory per GPU.


## Evaluation

### Prepare data

We evaluate LDF-VFI on the following two datasets:

- [SNU-FILM](https://myungsub.github.io/CAIN/). Expected data structure:

```
SNU_FILM
└── test
    ├── GOPRO_test
    │   ├── GOPR0384_11_00
    │   ├── GOPR0384_11_05
    |   ...
    |
    └── YouTube_test
        ├── 0000
        ├── 0001
        ...
```

- [X4K1000FPS](https://www.dropbox.com/scl/fo/88aarlg0v72dm8kvvwppe/AHxNqDye4_VMfqACzZNy5rU?rlkey=a2hgw60sv5prq3uaep2metxcn&e=1&dl=0). You should run `mp4_decoding.py` from X4K1000FPS to obtain the desired files. Expected data structure:

```
X4K1000FPS
└── test
    ├── Type1
    │   ├── TEST01_003_f0433
    │   ├── TEST02_045_f0465
    |   ...
    |
    ├── Type2
    │   ├── TEST06_001_f0273
    │   ├── TEST07_076_f1889
    |   ...
    |
    └── Type3
        ├── TEST11_078_f4977
        ├── TEST12_087_f2721
        ...
```


### Running evaluation

To evaluate on SNU-FILM dataset, run

```bash
bash quick_start/eval_snu_film.sh
```

To evaluate on X4K1000FPS dataset, run

```bash
bash quick_start/eval_x4k.sh
```

\* Please modify the following variables in the scripts according to your evaluation environment.

- `distributed_args`: modify according to your distributed environment.
- `model_path`: modify to `/path/to/LDF-VFI/transformer`
- `vae_path`: modify to `/path/to/LDF-VFI/Wan2.1_VAE_cond_v2.pth`
- `data_path`: modify to `/path/to/{SNU-FILM|X4K1000FPS}`

## Training

### Prepare data

- Download the [LAVIB dataset](https://github.com/alexandrosstergiou/LAVIB)
- Extract LAVIB and precompute the video paths with
```
python data_tools.py --src=/path/to/LAVIB --dst=data/lavib-hf
```

### Prepare pretrained models

- Download [Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- Download [Wan2.1_VAE.pth](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth?download=true)

### Running training

To start training, run

```bash
bash quick_start/train_dit.sh
```

\* Please modify the following variables in the scripts according to your training environment.

- `distributed_args`: modify according to your distributed environment.
- `model_path`: modify to `/path/to/Wan2.1-T2V-1.3B-Diffusers/transformer`
- `vae_path`: modify to `/path/to/Wan2.1_VAE.pth`

(Optional) If you run out of GPU memory, you can modify the `performance_args` by either reducing the batch size per GPU (`--batch_gpu`) to 1, 2, or 4, or increasing the sequence parallel size (`--sp_size`) to 2 or 4.

