import argparse
import os
import sys

from PIL import Image, ImageFile
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf

from models.get_models import get_model_with_opts
from saver import load_model_for_evaluate
from utils.platform_loader import read_yaml_options
from visualizer import Visualizer

sys.path.append(os.getcwd())
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='SMDE Prediction Parser')

parser.add_argument('--image_path',
                    dest='image_path',
                    required=True,
                    help='the path to the images')
parser.add_argument('--exp_opts',
                    dest='exp_opts',
                    required=True,
                    help="the yaml file for model's options")
parser.add_argument('--model_path',
                    dest='trained_model',
                    required=True,
                    help='the path of trained model')
parser.add_argument('--input_size',
                    dest='input_size',
                    type=int,
                    nargs='+',
                    default=None,
                    help='the size of input images')
parser.add_argument('--out_dir',
                    dest='out_dir',
                    type=str,
                    default=None,
                    help='the folder name for the outputs')

parser.add_argument('--cpu',
                    dest='cpu',
                    action='store_true',
                    default=False,
                    help='predicting with cpu')
parser.add_argument('-gpp',
                    '--godard_post_process',
                    action='store_true',
                    default=False,
                    help='Post-processing as done in Godards paper')
parser.add_argument('-mspp',
                    '--multi_scale_post_process',
                    action='store_true',
                    default=False,
                    help='Post-processing as done in FAL-Net')

opts = parser.parse_args()


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in
    Monodepthv1."""
    _, _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    l_mask = torch.from_numpy(l_mask.copy()).unsqueeze(0).to(l_disp)
    r_mask = torch.from_numpy(r_mask.copy()).unsqueeze(0).to(l_disp)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def multi_scale_post_process(l_disp, r_down_disp):
    norm = l_disp / (np.percentile(l_disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 1] = 1

    return (1 - norm) * l_disp + norm * r_down_disp

import cv2
import numpy as np
import os

def predict_one_image(network, inputs, visualizer, save_path, file):
    outputs = network.inference_forward(inputs, is_train=False)
    if opts.godard_post_process:
        inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
        flip_outputs = network.inference_forward(inputs, is_train=False)
        fflip_depth = torch.flip(flip_outputs[('depth', 's')], dims=[3])
        pp_depth = batch_post_process_disparity(
            1 / outputs[('depth', 's')], 1 / fflip_depth)
        pp_depth = 1 / pp_depth
        inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
        outputs[('depth', 's')] = pp_depth
    elif opts.multi_scale_post_process:
        inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
        up_fac = 2/3
        H, W = inputs['color_s'].shape[2:]
        inputs['color_s'] = F.interpolate(inputs['color_s'],
                                          scale_factor=up_fac,
                                          mode='bilinear',
                                          align_corners=True)
        flip_outputs = network.inference_forward(inputs, is_train=False)
        flip_depth = flip_outputs[('depth', 's')]
        flip_depth = up_fac * F.interpolate(flip_depth,
                                            size=(H, W),
                                            mode='nearest')
        fflip_depth = torch.flip(flip_depth, dims=[3])
        pp_depth = batch_post_process_disparity(
            1 / outputs[('depth', 's')], 1 / fflip_depth)
        pp_depth = 1 / pp_depth

        inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
        outputs[('depth', 's')] = pp_depth
    else:
        pp_depth = outputs[('depth', 's')]

    visual_map = {}
    visual_map['pp_disp'] = 1 / pp_depth
    visual_map['pp_depth'] = pp_depth
    visualizer.update_visual_dict(inputs, outputs, visual_map)
    visualizer.do_visualizion(os.path.splitext(file)[0] + '_visual')
    
    # 将深度图转换为 numpy 数组，并打印深度范围
    pp_depth_np = pp_depth.squeeze(0).squeeze(0).cpu().numpy()
    min_depth = np.amin(pp_depth_np)
    max_depth = np.amax(pp_depth_np)
    print(f"深度范围depth range: {min_depth:.4f} 到 {max_depth:.4f}")
    
    # 保存原始深度图 (npy 格式)
    npy_save_path = os.path.join(save_path, os.path.splitext(file)[0] + '_pred.npy')
    np.save(npy_save_path, pp_depth_np)
    
    # --- 以下部分为额外保存归一化的彩色深度图 ---
    # 归一化深度图到 [0, 1]
    depth_norm = (pp_depth_np - min_depth) / (max_depth - min_depth + 1e-8)
    # 转换为 0-255 的 8-bit 图像
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    # 应用彩色映射，这里使用 COLORMAP_JET，你可以根据需要选择其他映射
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    # 保存归一化的彩色深度图，建议保存为 PNG 格式
    color_save_path = os.path.join(save_path, os.path.splitext(file)[0] + '_color.png')
    cv2.imwrite(color_save_path, depth_color)

def predict():
    # Initialize the random seed and device
    if opts.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Initialize the options
    opts_dic = read_yaml_options(opts.exp_opts)

    # Initialize the network
    print('->Load the pretrained model')
    # print('->model name: {}'.format(opts_dic['model']['type']))
    print('->pretrained path: {}'.format(opts.trained_model))
    network = get_model_with_opts(opts_dic, device)
    network = load_model_for_evaluate(opts.trained_model, network)
    network.eval()

    if opts.out_dir is not None:
        os.makedirs(opts.out_dir, exist_ok=True)
        save_path = opts.out_dir
    else:
        if os.path.isfile(opts.image_path):
            save_path = os.path.dirname(opts.image_path)
        else:
            save_path = opts.image_path
    
    visualizer = Visualizer(save_path, {'type':{'pp_depth': 'depth'},
                                        'shape': [['pp_depth']]})

    to_tensor = tf.ToTensor()
    normalize = tf.Normalize(mean=opts_dic['pred_norm'],
                             std=[1, 1, 1])
    if opts.input_size is not None:
        image_size = opts.input_size
    else:
        image_size = opts_dic['pred_size']
    print('->resize image(s) into: {}'.format(image_size))
    try:
        # 新版本 Pillow
        resize = tf.Resize(image_size, interpolation=Image.Resampling.LANCZOS)
    except AttributeError:
        # 旧版本 Pillow
        resize = tf.Resize(image_size, interpolation=Image.Resampling.LANCZOS)

    # Predict
    if opts.godard_post_process or opts.multi_scale_post_process:
        print('->Use the post processing')
    print('->Start prediction')
    with torch.no_grad():
        if os.path.isfile(opts.image_path):
            img = Image.open(opts.image_path)
            img = img.convert('RGB')
            img = normalize(to_tensor(resize(img))).unsqueeze(0)
            inputs = {}
            inputs['color_s'] = img.to(device)
            file = os.path.basename(opts.image_path)
            predict_one_image(network, inputs, visualizer, save_path, file)
        else:
            for r, ds, fs in os.walk(opts.image_path):
                for f in fs:
                    if f.endswith('.png') or f.endswith('.jpg'):
                        print(f)
                        img = Image.open(os.path.join(r, f))
                        img = img.convert('RGB')
                        img = normalize(to_tensor(resize(img))).unsqueeze(0)
                        inputs = {}
                        inputs['color_s'] = img.to(device)
                        predict_one_image(network, inputs, visualizer,
                                          save_path, f)


    print('Finish Prediction!')
    print('Prediction files are saved in {}'.format(save_path))


def predict_stereo():
    # Initialize the random seed and device
    if opts.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Initialize the options
    opts_dic = read_yaml_options(opts.exp_opts)

    # Initialize the network
    print('->Load the pretrained model')
    # print('->model name: {}'.format(opts_dic['model']['type']))
    print('->pretrained path: {}'.format(opts.trained_model))
    network = get_model_with_opts(opts_dic, device)
    network = load_model_for_evaluate(opts.trained_model, network)
    network.eval()

    if opts.out_dir is not None:
        os.makedirs(opts.out_dir, exist_ok=True)
        save_path = opts.out_dir
    else:
        if os.path.isfile(opts.image_path):
            save_path = os.path.dirname(opts.image_path)
        else:
            save_path = opts.image_path
    
    visualizer = Visualizer(save_path, {'type':{'pp_depth': 'depth'},
                                        'shape': [['pp_depth']]})

    to_tensor = tf.ToTensor()
    normalize = tf.Normalize(mean=opts_dic['pred_norm'],
                             std=[1, 1, 1])
    if opts.input_size is not None:
        image_size = opts.input_size
    else:
        image_size = opts_dic['pred_size']
    print('->resize image(s) into: {}'.format(image_size))
    try:
        # 新版本 Pillow
        resize = tf.Resize(image_size, interpolation=Image.Resampling.LANCZOS)
    except AttributeError:
        # 旧版本 Pillow
        resize = tf.Resize(image_size, interpolation=Image.Resampling.LANCZOS)

    # Predict
    if opts.godard_post_process or opts.multi_scale_post_process:
        print('->Use the post processing')
    print('->Start prediction')
    with torch.no_grad():
        def readImg(imgpath, filename=None):

            if filename:
                # 从文件名去掉可能的后缀，方便后面加 _L/_R
                base_name = os.path.splitext(filename)[0]
                print(base_name)
                base_name = base_name.replace('_L', '')
                # 假设文件以 _L 和 _R 结尾
                imgL = Image.open(os.path.join(imgpath, f"{base_name}_L.png"))
                imgR = Image.open(os.path.join(imgpath, f"{base_name}_R.png"))
                print(imgL.size)
                print(imgR.size)
            else:
                # 对于单个文件路径，直接替换 _L 为 _R
                base_path = imgpath.replace('_L.png', '')
                imgL = Image.open(f"{base_path}_L.png")
                imgR = Image.open(f"{base_path}_R.png")
                print(imgL.size)
                print(imgR.szie)
            
            # 图像预处理
            imgL = imgL.convert('RGB')
            imgR = imgR.convert('RGB')
            imgL = normalize(to_tensor(resize(imgL))).unsqueeze(0)
            imgR = normalize(to_tensor(resize(imgR))).unsqueeze(0)
            
            return imgL, imgR

        # 主要代码修改
        if os.path.isfile(opts.image_path):
            imgL, imgR = readImg(opts.image_path)
            inputs = {}
            inputs['color_s'] = imgL.to(device)
            inputs['color_o'] = imgR.to(device)
            file = os.path.basename(opts.image_path)
            predict_one_image(network, inputs, visualizer, save_path, file)
        else:
            for r, ds, fs in os.walk(opts.image_path):
                for f in fs:
                    # 只处理左图
                    if (f.endswith('_L.png') or f.endswith('_L.jpg')):
                        print("reading imgs---,",f)
                        imgL, imgR = readImg(r, f)
                        inputs = {}
                        inputs['color_s'] = imgL.to(device)
                        inputs['color_o'] = imgR.to(device)
                        predict_one_image(network, inputs, visualizer, save_path, f)

    print('Finish Prediction!')
    print('Prediction files are saved in {}'.format(save_path))


if __name__ == '__main__':
    predict_stereo()
