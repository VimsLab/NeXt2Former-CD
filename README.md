# NeXt2Former-CD
This repository contains the code for ["NeXt2Former-CD: Efficient Remote Sensing Change Detection with Modern Vision Architectures"](https://arxiv.org/abs/2602.18717).

## Abstract
State Space Models (SSMs) have recently gained traction in remote sensing change detection (CD) for their favorable scaling properties. In this paper, we explore the potential of modern convolutional and attention-based architectures as a competitive alternative. We propose NeXt2Former-CD, an end-to-end framework that integrates a Siamese ConvNeXt encoder initialized with DINOv3 weights, a deformable attention-based temporal fusion module, and a Mask2Former decoder. This design is intended to better tolerate residual co-registration noise and small object-level spatial shifts, as well as semantic ambiguity in bi-temporal imagery. Experiments on LEVIR-CD, WHU-CD, and CDD datasets show that our method achieves the best results among the evaluated methods, improving over recent Mamba-based baselines in both F1 score and IoU. Furthermore, despite a larger parameter count, our model maintains inference latency comparable to SSM-based approaches, suggesting it is practical for high-resolution change detection tasks.

## Environment Setup
The following setup instructions were tested on Ubuntu 24.04.

After cloning this repository, initialize dependencies:
```
git submodule update --init --recursive
```

Create and configure the environment as follows:
```
conda create -n next2former-cd python=3.12
conda activate next2former-cd
```

Install PyTorch, CUDA toolkit, and Python dependencies:
```
pip install torch torchvision

conda install nvidia::cuda-toolkit=12.8

pip install deepspeed fvcore easydict tensorboardX timm opencv-python opencv-python-headless tqdm scipy scikit-image scikit-learn python-dotenv netCDF4 torchmetrics ipykernel
```

Set CUDA-related environment variables inside the conda environment:
```
conda env config vars set CUDA_HOME="$CONDA_PREFIX"
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib64"
conda deactivate
conda activate next2former-cd
```

Install `selective_scan`:
```
cd models/encoders/selective_scan
pip install . --no-build-isolation
cd ../../..
```

Install `detectron2`:
```
git submodule update --init --recursive detectron2
TORCH_CUDA_ARCH_LIST="12.0" pip install -e detectron2 --no-build-isolation
```

Install `Mask2Former`:
```
git submodule update --init --recursive Mask2Former

cd Mask2Former/mask2former/modeling/pixel_decoder/ops

# might need to fix deprecated calls inms_deform_attn_cuda.cu and ms_deform_attn.h

TORCH_CUDA_ARCH_LIST="12.0" sh make.sh
cd ../../../../..
```

For `TORCH_CUDA_ARCH_LIST`, use `"12.0"` for RTX 5090 and `"8.6"` for A6000.

## General file descriptions
- configs/*.py - config files which control multiple parameters related to data training, logging etc.
- dataloader/changeDataset.py - dataset class defined here.
- models/* - model files available here
- train.py - driver file for training. Instructions below
- eval.py - driver file for evaluation. Instructions below

## Link to model checkpoints
You can find our pretrained NeXt2Former-CD model checkpoints [here](https://drive.google.com/file/d/1I-vk8gsa74XMyETwxgDy7rnQVdboNIc3/view?usp=sharing)

## Datasets

1. We test our models on three public Change Detection datasets:
    - [LEVIR-CD](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=0)
    - [WHU-CD](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)
    - [CDD](https://www.dropbox.com/s/ls9fq5u61k8wxwk/CDD.zip?dl=0)

    The preprocessed links above are from [DDPM-CD](https://github.com/wgcban/ddpm-cd).

    We also provide the processed dataset splits used in this project here:
    [NeXt2Former-CD datasets](https://drive.google.com/file/d/1JbP29TwhnNhvfQXil0ZvUB65UCSgTNWP/view?usp=sharing)

    Please refer to the original dataset websites for additional details on each dataset.

2. If you are using your own datasets, please organize the dataset folder in the following structure:
    ```
    <root_folder>
    |-- A
        |-- <name1>.png
        |-- <name2>.png
        ...
    |-- B
        |-- <name1>.png
        |-- <name2>.png
        ...
    |-- gt
        |-- <name1>.png
        |-- <name2>.png
        ...
    |-- list
        |-- train.txt
        |-- val.txt
        |-- test.txt
    ```

    `train.txt/val.txt/test.txt` contains the names of items in training/validation/testing set, e.g.:

    ```
    <name1>
    <name2>
    ...
    ```

    Please make sure to change the root folder in the config files available in the `configs` folder. Also, if the files are in a format other than `png`, please specify the extension in the config.

    For custom datasets, you would need to create a config file similar to the existing files in the `configs` folder.

## Training
1. Please download the pretrained [VMamba]([https://github.com/MzeroMiko/VMamba](https://drive.google.com/drive/folders/1nSgzU-j0MVIbx4mG5xPKYnj1T_2CFBJu?usp=sharing)) weights:

    - [VMamba_Tiny](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth).
    - [VMamba_Small]([https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmsmall_dp03_ckpt_epoch_238.pth](https://drive.google.com/drive/folders/1nSgzU-j0MVIbx4mG5xPKYnj1T_2CFBJu?usp=sharing)).
    - [VMamba_Base]([https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmbase_dp06_ckpt_epoch_241.pth](https://drive.google.com/drive/folders/1nSgzU-j0MVIbx4mG5xPKYnj1T_2CFBJu?usp=sharing)).

    Place them under `pretrained/vmamba/`.

2. Please download the pretrained DINOv3 checkpoints from:

    - https://github.com/facebookresearch/dinov3

    For the default NeXt2Former-CD model, we use the ConvNeXt-Large backbone pretrained on the LVD-1689M dataset:
    `pretrained/DINOv3/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth`

    Place the downloaded checkpoint under `pretrained/DINOv3/`.

    Our code also supports other pretrained DINOv3 checkpoints, as well as VMamba checkpoints, by updating the corresponding config file.


3. Config setting.

    Edit the dataset-specific NeXt2Former-CD config file in the `configs` folder, for example:
    `config_levir_next2former-cd.py`, `config_whu_next2former-cd.py`, or `config_cdd_next2former-cd.py`.

    The current NeXt2Former-CD configs use `C.backbone = 'dinov3'`.
    Make sure `C.dinov3_pretrained` points to your downloaded checkpoint under `pretrained/DINOv3/`.

4. Run multi-GPU distributed training:

    ```shell
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0,1,2,3" \
        torchrun \
            --nproc_per_node=4 \
            --master_port 29502 \
            train.py -p 29502 \
            -d 0,1,2,3 -n "config_levir_next2former-cd"
    ```

    Here, `-n` specifies the config name, for example
    `config_levir_next2former-cd`, `config_whu_next2former-cd`, or `config_cdd_next2former-cd`.

5. You can also use single-GPU training:

    ```shell
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0" \
        torchrun \
            --nproc_per_node=1 \
            --master_port 29502 \
            train.py -p 29502 \
            -d 0 -n "config_levir_next2former-cd"
    ```
6. Results will be saved in `log_paper_final_8bs` folder.


## Evaluation
1. Run the evaluation by:

    ```shell
    python eval.py -d "0" \
      --config_path configs/config_cdd_next2former-cd.py \
      --checkpoint_path log_paper_final_checkpoints/NeXt2Former-CD_CDD.pth \
      --split test \
      --save_visualizations T \
      --save_path ./paper_eval_visualizations
    ```

    Here, `--config_path` selects the dataset-specific config file, `--checkpoint_path` points to the checkpoint to evaluate, and `--split` can be set to `train`, `val`, or `test`.

2. If you want to use multi GPUs please specify multiple Device IDs:

    ```shell
    python eval.py -d "0,1,2,3" \
      --config_path configs/config_cdd_next2former-cd.py \
      --checkpoint_path log_paper_final_checkpoints/NeXt2Former-CD_CDD.pth \
      --split test \
      --save_visualizations T \
      --save_path ./paper_eval_visualizations
    ```

3. If `--save_visualizations T` is enabled, outputs will be written under the directory given by `--save_path`.

## Acknowledgements
Our code is adapted from [M-CD](https://github.com/JayParanjape/M-CD/), which in turn builds on [Sigma](https://github.com/zifuwan/Sigma). We thank the authors of both projects for their valuable contributions and for open-sourcing their implementations. The dataset links above are sourced from [DDPM-CD](https://github.com/wgcban/ddpm-cd), and we thank the authors for making the processed splits easily accessible.

This material is based upon work supported by the National Science Foundation under NSF EIR Grant No. 2401835, entitled "Mapping of Natural Disasters by Deep Subspace Learning in Multi-band and Multi-spectral Satellite Images."

## Citation
```
To be added
```
