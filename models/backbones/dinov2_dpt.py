import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import warnings
from pathlib import Path

# 添加Depth-Anything-V2到路径
depth_anything_path = 'Depth-Anything-V2'
if depth_anything_path not in sys.path:
    sys.path.append(depth_anything_path)

from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.dpt import _make_scratch,_make_fusion_block


def check_image_size_for_patching(h, w, patch_size=14):
    """检查图像尺寸是否可以被patch_size整除"""
    if h % patch_size != 0 or w % patch_size != 0:
        warnings.warn(
            f"图像尺寸 ({h}x{w}) 不能被patch_size={patch_size}整除。"
            f"这可能导致特征提取不准确。建议使用能被{patch_size}整除的尺寸。"
            f"例如：{(h//patch_size)*patch_size}x{(w//patch_size)*patch_size} 或 "
            f"{((h//patch_size)+1)*patch_size}x{((w//patch_size)+1)*patch_size}"
        )
        return False
    return True

def load_weights(model, weights_path):
    """
    最简单的方式加载预训练权重。直接使用PyTorch的load_state_dict并忽略不匹配的键。
    
    Args:
        model: 目标模型
        weights_path: 预训练权重文件路径
        
    Returns:
        loaded_count: 成功加载的参数数量
    """
    if not os.path.exists(weights_path):
        print(f"预训练权重文件不存在: {weights_path}")
        return 0
    
    print(f"正在加载预训练权重: {weights_path}")
    
    # 加载权重
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # 处理不同格式的权重
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 直接加载，strict=False允许跳过不匹配的参数
    incompatible = model.load_state_dict(state_dict, strict=False)
    
    # 打印加载结果
    if incompatible.missing_keys:
        print(f"未找到的参数: {len(incompatible.missing_keys)} 个")
        if len(incompatible.missing_keys) < 10:
            print("具体包括:", incompatible.missing_keys)
        else:
            print("前10个:", incompatible.missing_keys[:10])
    
    if incompatible.unexpected_keys:
        print(f"模型中未使用的权重: {len(incompatible.unexpected_keys)} 个")
    
    # 计算实际加载的参数数量
    total_params = len(model.state_dict().keys())
    loaded_params = total_params - len(incompatible.missing_keys)
    print(f"成功加载 {loaded_params}/{total_params} 个参数")
    
    return loaded_params

class DINOv2_DPT_Backbone(nn.Module):
    """
    DINOv2 + DPT backbone for feature extraction
    Designed to be a drop-in replacement for Swin Transformer in TiO-Depth
    """
    def __init__(
        self,
        encoder_type='vitl',
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False
    ):
        super(DINOv2_DPT_Backbone, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder_type
        self.pretrained = DINOv2(model_name=encoder_type)
        
        #冻结DINOv2主干网络参数，在训练中不更新
        for param in self.pretrained.parameters():
            print("The enocder weight is frozen")
            param.requires_grad = False
        #print("Encoder weight is not frozen")
        # 初始化DPT头部 - 改名为depth_head以匹配预训练权重
        self.depth_head = DepthHeadFeaturePyramid(
            self.pretrained.embed_dim,
            features, 
            use_bn, 
            out_channels=out_channels,
            use_clstoken=use_clstoken
        )
        for param in self.depth_head.parameters():
            print("The dpt weight is frozen")
            param.requires_grad = False
        # print("DPT weight is not frozen")
    
    def forward(self, x):
        """提取特征金字塔，返回4层特征，与Swin Transformer输出兼容"""
        # 计算patch grid大小
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        # 直接打印某一层的权重
        if not hasattr(self, 'weight_check_counter'):
            self.weight_check_counter = 0
        
        #选择间隔多少次打印一次，避免输出过多
        # if self.weight_check_counter % 100 == 0:
        #     # 选择具体的层来监控
        #     target_layer = self.pretrained.blocks[0].attn.qkv.weight
        #     # 只打印权重矩阵的一小部分，例如前5个值
        #     print(f"DINOv2 Weight Check [{self.weight_check_counter}]:")
        #     print(f"Weight values: {target_layer[:5, :5]}")  # 打印5×5的子矩阵
        
        self.weight_check_counter += 1

        # 从DINOv2中提取中间层特征
        features = self.pretrained.get_intermediate_layers(
            x, 
            self.intermediate_layer_idx[self.encoder], 
            return_class_token=True
        )
    
        # 通过DPT头部获取特征金字塔
        feature_pyramid = self.depth_head(features, patch_h, patch_w)
        
        return feature_pyramid


class DepthHeadFeaturePyramid(nn.Module):
    """
    特征金字塔提取模块，基于DPT架构
    提取特征金字塔而不是最终深度图，使用和DPTHead类似的结构但更加清晰
    重命名为DepthHeadFeaturePyramid以匹配预训练权重
    """
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DepthHeadFeaturePyramid, self).__init__()
        
        self.in_channels = in_channels
        self.features = features
        self.use_bn = use_bn
        self.out_channels = out_channels
        self.use_clstoken = use_clstoken
        
        # 定义将特征投影到特定通道的卷积层
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        # 定义调整特征空间分辨率的层
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        # 如果使用class token，创建readout投影
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        # 创建用于统一通道数的层 (始终创建)
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        """
        提取特征金字塔并返回与Swin Transformer相似的四层特征
        
        Args:
            out_features: 从DINOv2提取的中间层特征
            patch_h: patch grid的高度
            patch_w: patch grid的宽度
            
        Returns:
            list: 包含4个特征图，从浅层到深层排列
        """
        # 打印ViT特征的形状信息，用于调试
        # print(f"\n[DPTHeadFeaturePyramid] Feature extraction for patch grid: {patch_h}x{patch_w}")
        for i, feat in enumerate(out_features):
            token_shape = feat[0].shape if isinstance(feat, tuple) else feat.shape
            
            #print(f"*****Layer {i} DINOv2 feature shape: {token_shape}")
        
        out = []
        for i, x in enumerate(out_features):
            #print(f"\nProcessing feature layer {i}:")
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                #print(f"  After using cls_token: {x.shape}")
            else:
                x = x[0]  # 不包括class token
                #print(f"  Without cls_token: {x.shape} [batch, seq_len, hidden_dim]")
            
            # 将序列特征重塑为2D特征图
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            #print(f"  Reshaped to 2D feature map: {x.shape} [batch, channels, height, width]")
            
            # 投影到指定的通道数
            x = self.projects[i](x)
            #print(self.projects[i].state_dict())
            #print(x)
            #a == 1
            #print(f"  After projection: {x.shape}")
            
            # 调整空间分辨率
            x = self.resize_layers[i](x)
            #print(f"  After spatial resolution adjustment: {x.shape}")
            
            out.append(x)
        
        # 现在我们有了初始特征金字塔：layer_1, layer_2, layer_3, layer_4
        layer_1, layer_2, layer_3, layer_4 = out
        #print("out----")
        #for ele in out:
        #    print(ele.shape)
        #print("out----")
        #a == 1
        # 使用scratch层统一特征维度
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # print("\n[DPTHeadFeaturePyramid] Final feature pyramid dimensions (unified channels):")
        # print(f"  Layer 1 (1/4 scale): {layer_1_rn.shape}")
        # print(f"  Layer 2 (1/8 scale): {layer_2_rn.shape}")
        # print(f"  Layer 3 (1/16 scale): {layer_3_rn.shape}")
        # print(f"  Layer 4 (1/32 scale): {layer_4_rn.shape}")
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        #feature_pyramid = [path_1, path_2, path_3, path_4]
        #feature_pyramid = [layer_1, layer_2, layer_3, layer_4]
        
        feature_pyramid = [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]
        #print('use rn outputs')
        #print('using reseemable output--the layer before fusion, same channel dimension')
        #print("feature_pyramid")
        #weights = self.scratch.refinenet1.out_conv.weight.data.cpu().tolist()
        #print(weights[0][:20])  
        #a == 1
        return feature_pyramid


def get_dinov2_dpt_backbone(backbone_name='dinov2b', pretrained=True):
    """
    获取DINOv2+DPT骨干网络，返回模型和通道数列表
    
    Args:
        backbone_name: 'dinov2s', 'dinov2b', 'dinov2l' 或 'dinov2g'
        pretrained: 是否使用预训练权重（目前未使用）
        
    Returns:
        model: DINOv2_DPT_Backbone实例
        enc_ch_num: 编码器特征通道数列表
    """
    encoder_type_map = {
        'dinov2s': 'vits',
        'dinov2b': 'vitb',
        'dinov2l': 'vitl',
        'dinov2g': 'vitg'
    }

    # 使用与Depth-Anything-V2 run.py相同的模型配置
    model_configs = {
        'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'features': 128, 'out_channels': [128, 128, 128, 128]},
        'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder_type = encoder_type_map.get(backbone_name, 'vitb')

    # 从模型配置中获取特征数和输出通道
    config = model_configs[encoder_type]
    features = config['features']
    out_channels = config['out_channels']
    print('========================')

    print(f"初始化 DINOv2+DPT ({encoder_type}):")
    print(f"  features: {features}")
    print(f"  out_channels: {out_channels}")

    # 创建模型
    model = DINOv2_DPT_Backbone(
        encoder_type=encoder_type,
        features=features,
        out_channels=[96,192,384,768],
        use_clstoken=False
    )

    pth_path = (
        Path(os.getcwd())
        / "Depth-Anything-V2"
        / "checkpoints"
        / "depth_anything_v2_vitb.pth"
    )

    model.load_state_dict(torch.load(pth_path, map_location='cpu'))

    # 输出的通道数列表
    enc_ch_num = [features] * 4
    if encoder_type == 'vitb':
        # 这里使用您测试中实际观察到的通道数
        enc_ch_num = [out_channels[0], out_channels[1], out_channels[2], out_channels[3]]
    return model, enc_ch_num 
