class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == "kitti":
            return '/zhouzm/Datasets/kitti'
        elif name == "kitti_stereo2015":
            return '/zhouzm/Datasets/kitti_2015'
        elif name == 'swin':
            return '/home/ywan0794/TiO-Depth_pytorch-main/swin_tiny_patch4_window7_224_22k.pth'
        elif name == 'flsea':
            return '/home/ywan0794/TiO-Depth_pytorch-main/flsea-stereo'
        elif name == 'tartanair':
            return '/home/ywan0794/tartanair_occ'
        else:
            raise NotImplementedError