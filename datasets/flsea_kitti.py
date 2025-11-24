import os
import random

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
import cv2
from torch.utils import data

from datasets.utils.data_reader import get_input_img
from datasets.utils.my_transforms import (
    BatchRandomCrop,
    NoneTransform,
    do_fal_color_aug,
)
from path_my import Path
from utils import platform_manager

scale_x = 832 / 1218
scale_y = 256 / 614

fx = 1867.3 * 1
fy = 1867.3 * 1
cx = 1141.8 * 1
cy = 134.49 * 1
baseline = 400.3473 / 1000  # 转换为米（如果是毫米）

# fx = 721.54
# fy = 721.54
# cx = 609.56
# cy = 172.85
# baseline = 0.54

K_of_FLSEA = [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]]


@platform_manager.DATASETS.add_module
class FlseaDataset(data.Dataset):
    def __init__(
        self,
        dataset_mode,
        split_file,
        full_size=None,
        patch_size=None,
        random_resize=True,
        normalize_params=[0.411, 0.432, 0.45],
        flip_mode=None,
        color_aug=True,
        output_frame=["o"],  # -1, 1
        multi_out_scale=None,
        load_KTmatrix=False,
        load_depth=True,
        load_depthhints=False,
        load_semantic=False,
        is_fixK=False,
        stereo_test=False,
        jpg_test=False,
        improved_test=False,
    ):
        super().__init__()
        self.init_opts = locals()

        self.dataset_dir = Path.get_path_of("flsea")
        self.dataset_mode = dataset_mode
        self.split_file = split_file
        self.full_size = full_size
        self.patch_size = patch_size
        self.random_resize = random_resize
        self.flip_mode = flip_mode
        self.color_aug = color_aug
        self.output_frame = output_frame
        self.multi_out_scale = multi_out_scale
        self.load_KTmatrix = load_KTmatrix
        self.load_depth = load_depth
        self.load_depthhints = load_depthhints
        self.load_semantic = load_semantic
        self.is_fixK = is_fixK
        self.improved_test = improved_test
        self.stereo_test = stereo_test
        self.jpg_test = jpg_test

        self.file_list = self._get_file_list(split_file)

        # Initialize transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])
        if dataset_mode == "train":
            # random resize and crop
            if self.patch_size is not None:
                self.crop = BatchRandomCrop(patch_size)
                self.canny = None
            else:
                self.crop = NoneTransform()
                self.canny = None
        else:
            if self.full_size is not None:
                self.color_resize = tf.Resize(
                    full_size, interpolation=Image.Resampling.LANCZOS
                )
            else:
                self.color_resize = NoneTransform()

        # Change the root path if use jpg images
        if self.jpg_test:
            raise NotImplementedError

        if not self.is_fixK:
            raise NotImplementedError

        if self.load_semantic:
            raise NotImplementedError

        if self.load_depthhints:
            raise NotImplementedError

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):
        global baseline
        
        file_info = self.file_list[f_idx]
        base_path_list, img_side = self._get_formative_file_base(file_info)

        # Read data
        inputs = {}
        target_path = base_path_list[0]
        color_side_path = target_path.format(
            "image_" + ("02" if img_side == "l" else "03")
        )
        inputs["color_s_raw"] = get_input_img(color_side_path)
        img_W, img_H = inputs["color_s_raw"].size

        intrinsic = np.array(K_of_FLSEA)
        k = intrinsic[0, 0] * baseline
        inputs["disp_k"] = torch.tensor(k, dtype=torch.float)
        if self.full_size is not None:
            inputs["disp_k"] *= self.full_size[1] / img_W

        inputs["direct"] = torch.tensor(1, dtype=torch.float)
        if img_side == "r":
            inputs["direct"] = -inputs["direct"]

        if self.load_depth:
            depth_path = target_path.format(
                "depth_" + ("02" if img_side == "l" else "03")
            )
            inputs["depth"] = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if self.stereo_test:
            color_path = target_path.format(
                "image_" + ("03" if img_side == "l" else "02")
            )
            inputs["color_o_raw"] = get_input_img(color_path)

        if self.load_KTmatrix:
            intric = np.zeros((4, 4))
            intric[:3, :3] = intrinsic[:, :3]
            intric[3, 3] = 1
            if self.full_size is not None:
                intric[0, :] *= self.full_size[1] / img_W
                intric[1, :] *= self.full_size[0] / img_H
            if self.is_fixK:
                baseline = -baseline if img_side == "l" else baseline
                extric = torch.tensor(
                    [[1, 0, 0, baseline], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )
            else:
                factor = -1 if img_side == "l" else 1
                extric = torch.tensor(
                    [[1, 0, 0, factor * 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )

            inputs["K"] = torch.from_numpy(intric).to(torch.float)
            inputs["inv_K"] = torch.from_numpy(np.linalg.pinv(intric)).to(torch.float)
            inputs["T"] = extric

        if self.dataset_mode == "train":
            for idx_frame, base_path in enumerate(base_path_list[1:]):
                if self.output_frame[idx_frame] != "o":
                    color_path = base_path.format(
                        "image_" + ("02" if img_side == "l" else "03")
                    )
                else:
                    color_path = base_path_list[0].format(
                        "image_" + ("03" if img_side == "l" else "02")
                    )
                inputs["color_{}_raw".format(self.output_frame[idx_frame])] = (
                    get_input_img(color_path)
                )

            if self.flip_mode is not None:
                if self.flip_mode == "both":  # random flip mode
                    switch_img = random.uniform(0, 1) > 0.5
                    switch_k = random.uniform(0, 1) > 0.5
                    if switch_img and switch_k:
                        is_flip = False
                        inputs["color_o_raw"], inputs["color_s_raw"] = (
                            inputs["color_s_raw"],
                            inputs["color_o_raw"],
                        )
                        inputs["direct"] = -inputs["direct"]
                        if self.load_KTmatrix:
                            inputs["T"][0, 3] = -inputs["T"][0, 3]
                    elif switch_img and not switch_k:
                        is_flip = True
                        inputs["color_o_raw"], inputs["color_s_raw"] = (
                            inputs["color_s_raw"],
                            inputs["color_o_raw"],
                        )
                    elif switch_img and not switch_k:
                        is_flip = True
                        inputs["direct"] = -inputs["direct"]
                        if self.load_KTmatrix:
                            inputs["T"][0, 3] = -inputs["T"][0, 3]
                    else:
                        is_flip = False
                else:
                    is_flip = random.uniform(0, 1) > 0.5
                    if is_flip:
                        flip_img = self.flip_mode == "img"
                        if flip_img:
                            inputs["color_o_raw"], inputs["color_s_raw"] = (
                                inputs["color_s_raw"],
                                inputs["color_o_raw"],
                            )
                        else:
                            inputs["direct"] = -inputs["direct"]
                            if self.load_KTmatrix:
                                inputs["T"][0, 3] = -inputs["T"][0, 3]

            # resize
            if self.full_size is not None:
                img_size = self.full_size
            else:
                _size = inputs["color_s_raw"].size  # (w, h)
                img_size = (_size[1], _size[0])
            scale_factor = (
                random.uniform(0.75, 1.5)
                if self.patch_size is not None and self.random_resize
                else 1
            )
            if scale_factor != 1 or self.full_size is not None:
                random_size = tuple(int(s * scale_factor) for s in img_size)
                self.color_resize = tf.Resize(
                    random_size, interpolation=Image.Resampling.LANCZOS
                )
                if self.multi_out_scale is not None:
                    self.multi_resize = {}
                    if self.patch_size is not None:
                        base_size = self.patch_size
                    else:
                        base_size = img_size
                    for scale in self.multi_out_scale:
                        s = 2**scale
                        self.multi_resize[scale] = tf.Resize(
                            [x // s for x in base_size],
                            interpolation=Image.Resampling.LANCZOS,
                        )

                self.depth_resize = tf.Resize(random_size, interpolation=Image.NEAREST)
            else:
                self.color_resize = NoneTransform()
                self.depth_resize = NoneTransform()

            # crop
            crop_params = self.crop.get_params(img_size, self.patch_size, scale_factor)
            # color jit
            if self.color_aug and random.uniform(0, 1) < 0.5:
                if self.color_aug == "v2":
                    do_color_aug = tf.ColorJitter.get_params(
                        (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1)
                    )
                else:
                    gamma_param = random.uniform(0.8, 1.2)
                    bright_param = random.uniform(0.5, 2)
                    cbright_param = [random.uniform(0.8, 1.2) for _ in range(3)]
            else:
                if self.color_aug == "v2":
                    do_color_aug = lambda x: x
                else:
                    gamma_param = 1
                    bright_param = 1
                    cbright_param = [1, 1, 1]

            for key in list(inputs):
                if "color" in key:
                    raw_img = inputs[key]
                    if is_flip:
                        raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
                    raw_img = self.crop(self.color_resize(raw_img), crop_params)
                    img = self.to_tensor(raw_img)
                    if self.color_aug == "v2":
                        aug_img = do_color_aug(img)
                    else:
                        aug_img = do_fal_color_aug(
                            img, gamma_param, bright_param, cbright_param
                        )
                    inputs[key.replace("_raw", "")] = self.normalize(img)
                    inputs[key.replace("_raw", "_aug")] = self.normalize(aug_img)
                    if self.multi_out_scale is not None:
                        for scale in self.multi_out_scale:
                            scale_img = self.multi_resize[scale](raw_img)
                            scale_img = self.to_tensor(scale_img)
                            if self.color_aug == "v2":
                                scale_aug_img = do_color_aug(scale_img)
                            else:
                                scale_aug_img = do_fal_color_aug(
                                    scale_img, gamma_param, bright_param, cbright_param
                                )
                            inputs[key.replace("_raw", "_{}".format(scale))] = (
                                self.normalize(scale_img)
                            )
                            inputs[key.replace("_raw", "_{}_aug".format(scale))] = (
                                self.normalize(scale_aug_img)
                            )

                elif "depth" in key:
                    # depth will be changed when resize
                    raw_depth = inputs[key] / scale_factor
                    if is_flip:
                        raw_depth = np.fliplr(raw_depth)
                    depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)
                    depth = self.crop(self.depth_resize(depth), crop_params)
                    inputs[key] = depth

                elif "hints" in key:
                    raw_hints = inputs[key] / scale_factor
                    # raw_hints = inputs[key] * random_size[1]
                    if is_flip:
                        raw_hints = torch.flip(raw_hints, dims=[2])
                    hints = self.crop(self.depth_resize(raw_hints), crop_params)
                    inputs[key] = hints.to(torch.float)
                elif "seg" in key:
                    raw_seg = inputs[key]
                    if is_flip:
                        raw_seg = torch.flip(raw_seg, dims=[2])
                    seg = self.crop(self.depth_resize(raw_seg), crop_params)
                    inputs[key] = seg

        # resize for other modes
        else:
            for key in list(inputs):
                if "color" in key:
                    raw_img = inputs[key]
                    img = self.color_resize(raw_img)
                    inputs[key.replace("_raw", "")] = self.normalize(
                        self.to_tensor(img)
                    )

                elif "depth" in key:
                    # do not resize ground truth in test
                    raw_depth = inputs[key]
                    depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)
                    inputs[key] = depth
                elif "hints" in key:
                    # do not resize hints truth in test
                    raw_hints = inputs[key]
                    inputs[key] = raw_hints.to(torch.float)

        # delete the raw data
        inputs.pop("color_s_raw")
        if self.stereo_test:
            inputs.pop("color_o_raw")
        if self.dataset_mode == "train":
            for id_frame in self.output_frame:
                inputs.pop("color_{}_raw".format(id_frame))

        inputs["file_info"] = [file_info]
        return inputs

    def _get_file_list(self, split_file):
        with open(split_file, "r") as f:
            files = f.readlines()
            filenames = []
            for f in files:
                file_name = f.replace("\n", "")
                filenames.append(file_name)
        return filenames

    def _get_formative_file_base(self, info):
        info_part = info.split(" ")
        data_dir = info_part[0]
        data_idx = info_part[1]
        img_side = info_part[2]

        base_path_list = []
        base_path = os.path.join(
            self.dataset_dir,
            data_dir,
            "{}",
            "data",
            "{:06d}".format(int(data_idx)) + ".png",
        )
        base_path_list.append(base_path)
        for frame_id in self.output_frame:
            if frame_id != "o":
                base_path = os.path.join(
                    self.dataset_dir,
                    data_dir,
                    "{}",
                    "data",
                    "{:06d}".format(int(data_idx) + frame_id) + ".png",
                )
                base_path_list.append(base_path)
            else:
                base_path = os.path.join(
                    self.dataset_dir,
                    data_dir,
                    "{}",
                    "data",
                    "{:06d}".format(int(data_idx)) + ".png",
                )
                base_path_list.append(base_path)

        return base_path_list, img_side

    @property
    def dataset_info(self):
        infos = []
        infos.append("    -{} Datasets".format(self.dataset_mode))
        infos.append("      get {} of data".format(len(self)))
        for key, val in self.init_opts.items():
            if key not in ["self", "__class__"]:
                infos.append("        {}: {}".format(key, val))
        return infos
