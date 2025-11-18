import os
import random

root_dir = '/home/ywan0794/TiO-Depth_pytorch/flsea-stereo'
subfolders = os.listdir(root_dir)
random.seed(0)  
def rel_path(path):
    # 转为相对路径
    return os.path.relpath(path, root_dir)

samples = []

for subfolder in subfolders:
    img_lft_dir = os.path.join(root_dir, subfolder, subfolder, 'imgs', 'LFT')
    img_rgt_dir = os.path.join(root_dir, subfolder, subfolder, 'imgs', 'RGT')
    depth_lft_dir = os.path.join(root_dir, subfolder, subfolder, 'depth', 'LFT')
    depth_rgt_dir = os.path.join(root_dir, subfolder, subfolder, 'depth', 'RGT')
    if not (os.path.isdir(img_lft_dir) and os.path.isdir(img_rgt_dir)
            and os.path.isdir(depth_lft_dir) and os.path.isdir(depth_rgt_dir)):
        continue

    img_files = sorted(os.listdir(img_lft_dir))
    for img_file in img_files:
        if not img_file.endswith('.tif'):
            continue

        img_lft_path = os.path.join(img_lft_dir, img_file)
        img_file_rgt = img_file.replace('LFT', 'RGT')
        img_rgt_path = os.path.join(img_rgt_dir, img_file_rgt)
        depth_lft_file = img_file.replace('.tif', '_abs_depth.tif')
        depth_rgt_file = img_file_rgt.replace('.tif', '_abs_depth.tif')
        depth_lft_path = os.path.join(depth_lft_dir, depth_lft_file)
        depth_rgt_path = os.path.join(depth_rgt_dir, depth_rgt_file)

        if all(os.path.exists(x) for x in [img_lft_path, img_rgt_path, depth_lft_path, depth_rgt_path]):
            # 拼接相对路径+左右标志
            line = f"{rel_path(img_lft_path)} l {rel_path(img_rgt_path)} r {rel_path(depth_lft_path)} l {rel_path(depth_rgt_path)} r"
            samples.append(line)

random.shuffle(samples)

num_total = len(samples)
num_train = int(0.2 * num_total)
num_val = int(0 * num_total)
num_test = int(0.025 * num_total)

def write_txt(filename, sample_list):
    print(f"Writing {len(sample_list)} samples to {filename}")
    with open(filename, 'w') as f:
        for line in sample_list:
            f.write(line + '\n')

write_txt('data_splits/flsea/train.txt', samples[:num_train])
write_txt('data_splits/flsea/val.txt', samples[num_train:num_train+num_val])
write_txt('data_splits/flsea/test.txt', samples[num_train+num_val:num_train+num_val+num_test])
write_txt('data_splits/flsea/full.txt', samples[:])

print("Done! Please check train.txt, val.txt, test.txt in:", os.getcwd())
