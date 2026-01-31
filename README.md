# <img src="./assets/stereoadapter_logo.png" alt="logo" width="50"/> StereoAdapter: Adapting Stereo Depth Estimation to Underwater Scenes

This is the official repository for the paper:
> **StereoAdapter: Adapting Stereo Depth Estimation to Underwater Scenes**
>
> Zhengri Wu\*, [Yiran Wang](https://github.com/u7079256)\*, Yu Wen\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Biao Wu, and [Hao Tang](https://ha0tang.github.io/)<sup>‡</sup>  
>
> *Equal contribution. <sup>†</sup>Project lead. <sup>‡</sup>Corresponding author.
>
> ***ICRA 2026***
>
> ### [Paper](https://arxiv.org/abs/2509.16415) | [Website](https://aigeeksgroup.github.io/StereoAdapter/) | [Dataset](https://huggingface.co/datasets/AIGeeksGroup/UW-StereoDepth-40K) | [Model](https://huggingface.co/AIGeeksGroup/StereoAdapter) | [HF Paper](https://huggingface.co/papers/2509.16415)

> [!NOTE]
> 💪 This visualizations brief introduce and show the real world deployment of StereoAdapter.

https://github.com/user-attachments/assets/7c3c656c-ea00-4b7c-8f24-f4294b84628f

## ✏️ Citation

If you find our code or paper helpful, please consider starring ⭐ us and citing:

```
@article{wu2025stereoadapter,
  title={StereoAdapter: Adapting Stereo Depth Estimation to Underwater Scenes},
  author={Wu, Zhengri and Wang, Yiran and Wen, Yu and Zhang, Zeyu and Wu, Biao and Tang, Hao},
  journal={arXiv preprint arXiv:2509.16415},
  year={2025}
}
```

## TODO List

- ✅ Release UW-StereoDepth-40K. (see [UW-StereoDepth-40K](https://huggingface.co/datasets/AIGeeksGroup/UW-StereoDepth-40K))
- ⬜️ Upload our paper to arXiv and build project pages.
- ⬜️ Upload the pretrained model.
- ✅ Upload the code.

## 🏃 Intro StereoAdapter

## 🔧Run Your StereoAdapter

### 1. Install & Requirements

```bash
conda env create -f environment.yaml
conda activate stereoadapter
```

### 2. Download Pertrained Depth Anything V2 Model

```bash
mkdir -p Depth-Anything-V2/checkpoints
cd Depth-Anything-V2/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
cd ../..
```

### 3. Train StereoAdapter Model

```bash
python train_dist_2.py\
  --name MODEL_NAME\
  --exp_opts options/TiO-Depth/train/gru-dav2_codyra-tartanair.yaml\
  --batch_size BATH_SIZE\ 
  --metric_source rawdepth sdepth\ 
  --save_freq SAVE_FREQUENCY\ 
  --visual_freq VISUAL_FREQUENCY\ 
  --is_codyra True\ 
  --step_epochs 20 30
```

### 4. Evaluate StereoAdapter

```bash
python evaluate.py\
 --exp_opts options/TiO-Depth/train/gru-dav2_codyra-tartanair.yaml\
 --model_path MODEL_PATH
```
