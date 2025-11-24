import os
import random
from collections import namedtuple

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
import cv2
from torch.utils import data

from datasets.utils.data_reader import get_input_depth, get_input_img, get_intrinsic
from datasets.utils.my_transforms import (
    BatchRandomCrop,
    NoneTransform,
    do_fal_color_aug,
)
from path_my import Path
from utils import platform_manager


fx = 1867.3
fy = 1867.3
cx = 1141.8
cy = 134.49
baseline = 400.3473 / 1000  # 转换为米（如果是毫米）

K_of_FLSEA = [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]]


Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.
        "trainId",  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",  # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",  # Whether this label distinguishes between single instances or not
        "ignoreInEval",  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)


@platform_manager.DATASETS.add_module
class FlseaDataset(data.Dataset):
    """flsea dataset for depth estimation from images."""

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

        if self.load_semantic:
            assert os.path.exists(os.path.join(self.dataset_dir, "segmentation"))

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
                if hasattr(Image, "Resampling"):
                    RESAMPLE = Image.Resampling.LANCZOS
                else:
                    RESAMPLE = Image.ANTIALIAS
                self.color_resize = tf.Resize(full_size, interpolation=RESAMPLE)
            else:
                self.color_resize = NoneTransform()

        # Change the root path if use jpg images
        if self.jpg_test:
            if dataset_mode == "test":
                self.dataset_dir = Path.get_path_of("eigen_kitti_test_jpg")
            else:
                raise NotImplementedError

        # Read pre-generated ground-truth depths for testing:
        self.gt_depths = None
        # if len(self.file_list) == 697 or len(self.file_list) == 652:
        #     with open("debug_log.txt", "a") as f:
        #         f.write("Loading pre depth: ", "\n")

        #     if self.improved_test:
        #         gt_path = os.path.join(self.dataset_dir, 'gt_depths_improved.npz')
        #         self.gt_depths = np.load(gt_path,
        #                                  fix_imports=True,
        #                                  encoding='latin1',
        #                                  allow_pickle=True)['data']
        #     else:
        #         gt_path = os.path.join(self.dataset_dir, 'gt_depths_raw.npz')
        #         self.gt_depths = np.load(gt_path,
        #                                  fix_imports=True,
        #                                  encoding='latin1',
        #                                  allow_pickle=True)['data']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):
        """Return a data dictionary."""
        file_info = self.file_list[f_idx]
        img_lft_path, img_rgt_path, depth_lft_path, depth_rgt_path, data_side = (
            self._get_formative_file_base(file_info)
        )

        # Read data
        inputs = {}
        base_path_list = []
        base_path_list.append(img_lft_path if data_side == "l" else img_rgt_path)
        inputs["color_s_raw"] = get_input_img(
            img_lft_path if data_side == "l" else img_rgt_path
        )
        img_W, img_H = inputs["color_s_raw"].size

        if self.is_fixK:
            intrinsic = np.array(K_of_FLSEA)
        else:
            # intrinsic = get_intrinsic(date_path)
            intrinsic = np.array(K_of_FLSEA)
        original_width = 1218
        fx_n = fx
        if self.full_size is not None:
            scale = self.full_size[1] / original_width
            fx_n *= scale

        disp_k = fx_n * baseline
        inputs["disp_k"] = torch.tensor(disp_k, dtype=torch.float)

        inputs["direct"] = torch.tensor(
            1, dtype=torch.float
        )  # used to determine the direction of baseline
        if data_side == "r":
            inputs["direct"] = -inputs["direct"]

        # take ground-truth depths
        # if self.gt_depths is not None:
        #     inputs['depth'] = self.gt_depths[f_idx]
        # else:
        #     if self.load_depth:
        #         inputs['depth'] = cv2.imread(depth_lft_path if data_side == 'l' else depth_rgt_path, cv2.IMREAD_UNCHANGED)

        inputs["depth"] = cv2.imread(
            depth_lft_path if data_side == "l" else depth_rgt_path, cv2.IMREAD_UNCHANGED
        )
        # if inputs['depth'] is not None:
        #     with open("/home/ywan0794/TiO-Depth_pytorch-main/debug_log.txt", "a") as f:
        #         f.write("Loading depyh image: "  + "\n")
        #         f.write(depth_lft_path if data_side == 'l' else depth_rgt_path + "\n")
        # else:
        #     with open("/home/ywan0794/TiO-Depth_pytorch-main/debug_log.txt", "a") as f:
        #         f.write("Failed to load depyh image: " + "\n")
        #         f.write(depth_lft_path if data_side == 'l' else depth_rgt_path + "\n")

        if self.stereo_test:
            inputs["color_o_raw"] = get_input_img(
                img_lft_path if data_side == "r" else img_rgt_path
            )

        if self.load_KTmatrix:
            intric = np.zeros((4, 4))
            intric[:3, :3] = intrinsic[:, :3]
            intric[3, 3] = 1
            if self.full_size is not None:
                intric[0, :] *= self.full_size[1] / img_W
                intric[1, :] *= self.full_size[0] / img_H
            if self.is_fixK:
                if data_side == "l":
                    baseline_n = -400.3473 / 1000
                else:
                    baseline_n = 400.3473 / 1000
                extric = torch.tensor(
                    [[1, 0, 0, baseline_n], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )
            else:
                if data_side == "l":
                    factor = -1
                else:
                    factor = 1
                extric = torch.tensor(
                    [
                        [1, 0, 0, factor * (400.3473 / 1000)],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )

            inputs["K"] = torch.from_numpy(intric).to(torch.float)
            inputs["inv_K"] = torch.from_numpy(np.linalg.pinv(intric)).to(torch.float)
            inputs["T"] = extric

        if self.dataset_mode == "train":
            # for idx_frame, base_path in enumerate(base_path_list[1:]):
            #     if self.output_frame[idx_frame] != 'o':
            #         color_path = base_path.format(
            #             *self.DATA_NAME_DICT['color_{}'.format(data_side)])
            #     else:
            #         color_path = base_path_list[0].format(*self.DATA_NAME_DICT[
            #             'color_{}'.format('r' if data_side == 'l' else 'l')])
            #     if self.jpg_test:
            #         color_path = color_path.replace('.png', '.jpg')
            #     inputs['color_{}_raw'.format(self.output_frame[idx_frame])]\
            #         = get_input_img(color_path)

            # if self.load_depthhints:
            #     hints_l_path = base_path.format(
            #         *self.DATA_NAME_DICT['hints_{}'.format(data_side)])
            #     hints_l_path = hints_l_path.replace(
            #         self.dataset_dir,
            #         self.dataset_dir + '/depth_hints').replace('data/', '')
            #     hints_depth_l = torch.from_numpy(np.load(hints_l_path))
            #     # 0.058 = 0.58 (K[0, 0]) * 0.1 (baseline)
            #     inputs['hints_s'] = 5.4 * hints_depth_l
            #     # inputs['hints_s'] = 0.058 / (hints_depth_l +
            #     #                              1e-8) * (hints_depth_l > 0)
            #     if 'o' in self.output_frame:
            #         oside = 'r' if data_side == 'l' else 'l'
            #         hints_r_path = base_path.format(
            #             *self.DATA_NAME_DICT['hints_{}'.format(oside)])
            #         hints_r_path = hints_r_path.replace(
            #             self.dataset_dir,
            #             self.dataset_dir + '/depth_hints').replace('data/', '')
            #         hints_depth_r = torch.from_numpy(np.load(hints_r_path))
            #         inputs['hints_o'] = 5.4 * hints_depth_r
            #         # inputs['hints_o'] = 0.058 / (hints_depth_r +
            #         #                              1e-8) * (hints_depth_r > 0)
            inputs["color_o_raw"] = get_input_img(
                img_lft_path if data_side == "r" else img_rgt_path
            )

            if self.dataset_mode == "train" and self.flip_mode is not None:
                if self.flip_mode == "both":  # random flip mode
                    switch_img = random.uniform(0, 1) > 0.5
                    switch_k = random.uniform(0, 1) > 0.5
                    if switch_img and switch_k:
                        is_flip = False
                        inputs["color_o_raw"], inputs["color_s_raw"] = (
                            inputs["color_s_raw"],
                            inputs["color_o_raw"],
                        )
                        if self.load_depthhints:
                            inputs["hints_s"], inputs["hints_o"] = (
                                inputs["hints_o"],
                                inputs["hints_s"],
                            )
                        if self.load_semantic:
                            inputs["seg_s"], inputs["seg_o"] = (
                                inputs["seg_o"],
                                inputs["seg_s"],
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
                        if self.load_depthhints:
                            inputs["hints_s"], inputs["hints_o"] = (
                                inputs["hints_o"],
                                inputs["hints_s"],
                            )
                        if self.load_semantic:
                            inputs["seg_s"], inputs["seg_o"] = (
                                inputs["seg_o"],
                                inputs["seg_s"],
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
                            if self.load_depthhints:
                                inputs["hints_s"], inputs["hints_o"] = (
                                    inputs["hints_o"],
                                    inputs["hints_s"],
                                )
                            if self.load_semantic:
                                inputs["seg_s"], inputs["seg_o"] = (
                                    inputs["seg_o"],
                                    inputs["seg_s"],
                                )
                        else:
                            inputs["direct"] = -inputs["direct"]
                            if self.load_KTmatrix:
                                inputs["T"][0, 3] = -inputs["T"][0, 3]

        # Process data
        # resize crop & color jit & flip for train
        if self.dataset_mode == "train":
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
                if hasattr(Image, "Resampling"):
                    RESAMPLE = Image.Resampling.LANCZOS
                else:
                    RESAMPLE = Image.ANTIALIAS
                self.color_resize = tf.Resize(random_size, interpolation=RESAMPLE)
                if self.multi_out_scale is not None:
                    self.multi_resize = {}
                    if self.patch_size is not None:
                        base_size = self.patch_size
                    else:
                        base_size = img_size
                    for scale in self.multi_out_scale:
                        if hasattr(Image, "Resampling"):
                            RESAMPLE = Image.Resampling.LANCZOS
                        else:
                            RESAMPLE = Image.ANTIALIAS
                        s = 2**scale
                        self.multi_resize[scale] = tf.Resize(
                            [x // s for x in base_size], interpolation=RESAMPLE
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
        inputs.pop("color_s_raw", None)
        if self.stereo_test:
            inputs.pop("color_o_raw", None)
        if self.dataset_mode == "train":
            for id_frame in self.output_frame:
                inputs.pop("color_{}_raw".format(id_frame), None)

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
        img_lft_dir = info_part[0]
        data_side = info_part[1]
        img_rgt_dir = info_part[2]
        depth_lft_dir = info_part[4]
        depth_rbt_dir = info_part[6]

        img_lft_path = os.path.join(self.dataset_dir, img_lft_dir)
        img_rgt_path = os.path.join(self.dataset_dir, img_rgt_dir)
        depth_lft_path = os.path.join(self.dataset_dir, depth_lft_dir)
        depth_rgt_path = os.path.join(self.dataset_dir, depth_rbt_dir)

        return img_lft_path, img_rgt_path, depth_lft_path, depth_rgt_path, data_side

    @property
    def dataset_info(self):
        infos = []
        infos.append("    -{} Datasets".format(self.dataset_mode))
        infos.append("      get {} of data".format(len(self)))
        for key, val in self.init_opts.items():
            if key not in ["self", "__class__"]:
                infos.append("        {}: {}".format(key, val))
        return infos
