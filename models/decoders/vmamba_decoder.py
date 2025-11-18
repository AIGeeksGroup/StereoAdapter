import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from einops import rearrange
import sys
import os

# Add VMamba path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VMamba'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from VMamba.vmamba import VSSBlock, SS2D
from utils.gru_utils import bilinear_sampler
from .gru_decoder import BasicMotionEncoder, UpdateHead
try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    pass


class ConvVMamba(nn.Module):
    """
    简化版 VMamba 替换 ConvGRU - 基于实际 VSSM 配置
    """
    def __init__(self, hidden_dim, input_dim, d_state=1, ssm_ratio=2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        total_dim = hidden_dim

        # 输入投影层
        self.input_proj = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 1)

        # VMamba 核心块 - 使用实际 VSSM 的参数配置
        self.vss_block = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=0.1,  # 推理时不使用 drop_path
            norm_layer=nn.LayerNorm,
            channel_first=True,
            ssm_d_state=1,
            ssm_ratio=1.0,  # 标准扩展比例
            ssm_dt_rank="auto",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_act_layer=nn.SiLU,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2_noz",  # 使用支持oflex的版本
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,

        )

        # 输出门控 - 简单的残差门
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

        # 检查并显示实际使用的加速后端
        self._check_backend()

    def _check_backend(self):
        """检查并显示VMamba实际使用的加速后端"""
        import sys

        # 检查可用的加速模块
        try:
            import selective_scan_cuda_oflex
            oflex_available = True
        except ImportError:
            oflex_available = False

        try:
            import selective_scan_cuda
            mamba_available = True
        except ImportError:
            mamba_available = False

        # 获取配置的后端
        backend_config = "unknown"
        if hasattr(self.vss_block, 'op'):
            ss2d = self.vss_block.op
            if hasattr(ss2d, 'forward_core') and hasattr(ss2d.forward_core, 'keywords'):
                backend_config = ss2d.forward_core.keywords.get('selective_scan_backend', 'not set')

        # 确定实际使用的后端
        actual_backend = None
        if backend_config == 'oflex' and oflex_available:
            actual_backend = "oflex (CUDA优化)"
        elif backend_config == 'mamba' and mamba_available:
            actual_backend = "mamba (CUDA)"
        elif backend_config == 'torch' or (not oflex_available and not mamba_available):
            actual_backend = "torch (纯PyTorch,较慢)"
        else:
            # 根据优先级推断
            if oflex_available:
                actual_backend = "oflex (CUDA优化)"
            elif mamba_available:
                actual_backend = "mamba (CUDA)"
            else:
                actual_backend = "torch (纯PyTorch,较慢)"

        # 只在第一次创建时打印
        if not hasattr(self.__class__, '_backend_printed'):
            print(f"\n{'='*60}")
            print(f"VMamba Decoder 加速后端信息")
            print(f"{'='*60}")
            print(f"配置的后端: {backend_config}")
            print(f"oflex可用: {oflex_available}")
            print(f"mamba可用: {mamba_available}")
            print(f"🎯 实际使用: {actual_backend}")
            print(f"{'='*60}\n")
            self.__class__._backend_printed = True

    def forward(self, h, cz, cr, cq, *x_list):
        """
        简化的前向传播 - 不再保持三门结构
        h: [B, hidden_dim, H, W] 隐藏状态
        cz, cr, cq: 忽略这些参数（仅为接口兼容）
        x_list: 输入特征列表
        """
        # 处理输入
        x = torch.cat(x_list, dim=1) if x_list else torch.zeros_like(h[:, :0])

        # 合并隐藏状态和输入
        combined = torch.cat([h, x], dim=1)  # [B, hidden_dim+input_dim, H, W]
        feat = self.input_proj(combined)  # [B, hidden_dim, H, W]

        # 通过 VMamba 处理
        out = self.vss_block(feat)

        # 残差连接 + 门控
        gate = self.gate(out)
        h_new = gate * out + (1 - gate) * h

        return h_new


class BasicVMambaUpdateBlock(nn.Module):
    """
    使用 VMamba 替换 GRU 的多尺度更新块
    """
    def __init__(self, args, hidden_dims=[], up_factor=None):
        super().__init__()
        self.args = args
        encoder_output_dim = args.encoder_output_dim
        self.n_gru_layers = args.n_gru_layers
        self.n_downsample = args.n_downsample

        # 使用 ConvVMamba 替换 ConvGRU
        # 为每个尺度创建 VMamba 模块 - 使用优化后的参数
        self.vmamba08 = ConvVMamba(
            hidden_dims[2],
            encoder_output_dim + hidden_dims[1] * (self.n_gru_layers > 1),
            d_state=1,  # 使用轻量级配置
            ssm_ratio=1.0
        )

        self.vmamba16 = ConvVMamba(
            hidden_dims[1],
            hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2],
            d_state=1,
            ssm_ratio=1.0
        )

        self.vmamba32 = ConvVMamba(
            hidden_dims[0],
            hidden_dims[1],
            d_state=1,
            ssm_ratio=1.0
        )

        # 导入原始的运动编码器和流预测头

        self.encoder = BasicMotionEncoder(args)
        self.flow_head = UpdateHead(hidden_dims[2], hidden_dim=256, output_dim=2)

        # 上采样 mask - 与 GRU 版本保持一致
        factor = 2**self.n_downsample if up_factor is None else up_factor
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0)
        )

    def forward(self, net, inp, corr=None, corr_mono=None, flow=None,
                flow_conf=None, iter08=True, iter16=True, iter32=True,
                iter64=True, update=True):
        """
        与 BasicMultiUpdateBlock 相同的前向传播接口
        """
        def pool2x(x):
            return F.avg_pool2d(x, 3, stride=2, padding=1)

        def interp(x, dest):
            interp_args = {'mode': 'bilinear', 'align_corners': True}
            return F.interpolate(x, dest.shape[2:], **interp_args)

        # 多尺度更新（从粗到细）
        if iter32:
            # VMamba 不需要 inp（那是给门控用的），只传入其他特征
            net[2] = self.vmamba32(net[2], None, None, None, pool2x(net[1]))

        if iter16:
            if self.n_gru_layers > 2:
                net[1] = self.vmamba16(net[1], None, None, None, pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.vmamba16(net[1], None, None, None, pool2x(net[0]))

        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.n_gru_layers > 1:
                net[0] = self.vmamba08(net[0], None, None, None, motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.vmamba08(net[0], None, None, None, motion_features)

        if not update:
            return net

        # 预测流场更新
        delta_flow = self.flow_head(net[0])
        mask = .25 * self.mask(net[0])

        return net, mask, delta_flow


class VMambaDecoder(nn.Module):
    """
    使用 VMamba 替换 GRU 的解码器
    保持与 GRUDecoder 相同的接口
    """
    def __init__(self,
                 corr_implementation = "reg",
                 corr_radius = 4,
                 corr_levels = 4,
                 n_gru_layers = 3,
                 context_dims = [128, 128, 128],
    ):
        super().__init__()
        self.corr_implementation = corr_implementation
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.n_gru_layers = n_gru_layers
        self.context_dims = context_dims

        # Context 网络（生成初始隐藏状态）
        self.cnet_net = nn.ModuleList()
        self.cnet_inp = nn.ModuleList()
        # VMamba 不需要 z/q/r 门控，删除 context_zqr_convs

        for dim in self.context_dims:
            # 创建 cnet_net 模块（生成初始隐藏状态）
            cnet_net_module = nn.Sequential(
                nn.Conv2d(1, dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1)
            )
            self.cnet_net.append(cnet_net_module)

            # 创建 cnet_inp 模块（生成输入特征）
            cnet_inp_module = nn.Sequential(
                nn.Conv2d(1, dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1)
            )
            self.cnet_inp.append(cnet_inp_module)

        # 使用 VMamba 版本的更新块
        raft_args = SimpleNamespace(
            corr_radius = corr_radius,
            corr_levels = corr_levels,
            n_gru_layers = n_gru_layers,
            encoder_output_dim = context_dims[-1],
            n_downsample = 2
        )

        self.update_block = BasicVMambaUpdateBlock(
            raft_args, hidden_dims=self.context_dims, up_factor=7
        )

    def forward(
        self,
        d2d,
        outputs,
        img2, img3,
        fmap2, fmap3,
        directs,
        is_train,
        cnet_list,
        iters=12,
        use_scaled_mde_init=True,
    ):
        """
        使用 VMamba 进行立体匹配的迭代优化
        """
        device = img2.device
        B, _, H, W = img2.shape
        (Hc, Wc), (fh, fw) = self._pick_stride_and_factor(fmap2, img2)
        assert fh == fw, "非等比下采样暂不支持"
        up_factor = fh



        # 选择 1D corr 实现
        corr_block_cls = CorrBlockFast1D if self.corr_implementation == "reg_cuda" else CorrBlock1D

        # 构建 stereo 相关体积
        stereo_corr_volume = corr_block_cls.corr(fmap2.float(), fmap3.float()).squeeze(3).unsqueeze(1)
        stereo_corr_fn = corr_block_cls(
            stereo_corr_volume.squeeze(1).unsqueeze(3),
            radius=self.corr_radius,
            num_levels=self.corr_levels
        )

        # 用 Mono 分支的 mde 作为初始化
        assert 'mono_disp_0_s' in outputs, "需要 Mono 分支先产生 mono_disp_0_s"
        mde2 = outputs['mono_disp_0_s'] # [B,1,H,W] （视差表征）
        mde2_low = F.interpolate(mde2, size=(Hc, Wc), mode="bilinear", align_corners=True) / float(up_factor)

        # 处理 context 列表（初始隐藏状态）
        net_list = [torch.tanh(x) for x in cnet_list]
        inp_list = [torch.relu(x) for x in cnet_list]


        # 初始化 RAFT 坐标网格
        coords0, coords1 = self._initialize_flow(B, Hc, Wc, device=device)
        if use_scaled_mde_init:
            init_flow = torch.zeros(B, 2, Hc, Wc, device=device)
            init_flow[:, 0:1] = mde2_low # 仅水平位移
            coords1 = coords1 + init_flow

        # RAFT 迭代更新（使用 VMamba）
        disp_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()  # 截断梯度
            stereo_corr = stereo_corr_fn(coords1)  # 采样相关体积
            flow_lr = coords1 - coords0  # 当前流场

            # 使用 VMamba 更新块进行迭代优化
            net_list, mask_up, delta_flow = self.update_block(
                net_list, inp_list, stereo_corr, None, flow_lr,
                iter32=(self.n_gru_layers >= 3),
                iter16=(self.n_gru_layers >= 2),
                iter08=True
            )

            # 仅保留水平位移（立体匹配只有水平视差）
            delta_flow[:, 1] = 0.0
            coords1 = coords1 + delta_flow

            # 提取水平视差
            disp_lr = (coords1 - coords0)[:, :1]  # [B, 1, Hc, Wc]

            # 上采样到输入分辨率
            flow_up = self._convex_upflow(disp_lr, mask_up, up_factor=up_factor, out_hw=(H, W))
            disp_predictions.append(flow_up)

        # 保存最终结果
        disp = disp_predictions[-1]
        outputs['stereo_disp_0_s'] = disp
        outputs['stereo_depth_0_s'] = d2d / disp

        if is_train:
            outputs['stereo_flow_preds'] = disp_predictions

        return outputs

    # ========== 辅助函数 ==========
    @staticmethod
    def _coords_grid(b, h, w, device):
        """生成坐标网格"""
        y, x = torch.meshgrid(
            torch.linspace(0, h - 1, h, device=device),
            torch.linspace(0, w - 1, w, device=device),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        return grid

    def _initialize_flow(self, b, h, w, device):
        """初始化流场"""
        coords0 = self._coords_grid(b, h, w, device)
        coords1 = coords0.clone()
        return coords0, coords1

    @staticmethod
    @torch.no_grad()
    def _pick_stride_and_factor(fmap, img):
        """推断特征分辨率与放大倍率"""
        B, _, H, W = img.shape
        _, _, Hc, Wc = fmap.shape
        assert H % Hc == 0 and W % Wc == 0, "图像与特征需整倍率对齐"
        return (Hc, Wc), (H // Hc, W // Wc)

    @staticmethod
    def _convex_upflow(flow, mask, up_factor=None, out_hw=None):
        """使用 convex 组合进行上采样"""
        B, C, Hc, Wc = flow.shape
        if up_factor is None:
            assert out_hw is not None, "需要 out_hw 推断上采样倍率"
            H, W = out_hw
            fh, fw = H // Hc, W // Wc
            assert fh == fw, "非等比下采样暂不支持"
            up_factor = fh

        # 重塑 mask: [B, 1, 9, f, f, Hc, Wc]
        mask = mask.view(B, 1, 9, up_factor, up_factor, Hc, Wc)
        mask = torch.softmax(mask, dim=2)

        # 展开 3x3 邻域
        up_flow = F.unfold(flow, [3, 3], padding=1)  # [B, 9, Hc*Wc]
        up_flow = up_flow.view(B, 1, 9, Hc, Wc)

        # 加权求和
        up_flow = torch.sum(mask * up_flow.unsqueeze(3).unsqueeze(3), dim=2)

        # 重排维度
        up_flow = up_flow.permute(0, 1, 4, 5, 2, 3).reshape(B, 1, Hc * up_factor, Wc * up_factor)

        # 缩放流场值
        return up_flow * up_factor


class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume, coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlockFast1D:
    def __init__(self, fullcorr, num_levels=4, radius=4, pad=[0,0]):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.pad = pad

        # all pairs correlation
        self.fullcorr = fullcorr

        batch, h1, w1, dim, w2 = self.fullcorr.shape
        corr = self.fullcorr.reshape(batch*h1*w1, dim, 1, w2)
        for i in range(self.num_levels):
            self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2//2**i))
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])

    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        coords = coords[:, :1] # B 2 H W -> B 1 H W
        coords = coords + self.pad[0] # Real coords are shifted by pad[0]
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), coords/2**i, self.radius)
            corr = corr.view(bz, -1, ht, wd)
            corr = corr[:, :, :, self.pad[0]:wd-self.pad[1]] # B 2r+1 H W'
            out_pyramid.append()
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap2, fmap3):
        B, D, H, W1 = fmap2.shape
        _, _, _, W2 = fmap3.shape
        fmap2_dtype = fmap2.dtype

        fmap2 = fmap2.view(B, D, H, W1)
        fmap3 = fmap3.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap2, fmap3)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return (corr / torch.sqrt(torch.tensor(D))).to(fmap2_dtype)

#Cannot create correlation volume dynamically
#class PytorchAlternateCorrBlock1D
#class AlternateCorrBlock

class CorrBlock1D:
    def __init__(self, fullcorr, num_levels=4, radius=4, pad = [0,0]):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.pad = pad

        # all pairs correlation
        self.fullcorr = fullcorr

        batch, h1, w1, dim, w2 = self.fullcorr.shape # B, H, W2, 1, W3
        corr = self.fullcorr.reshape(batch*h1*w1, dim, 1, w2) # BHW 1 W3

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1) # B 2 H W -> B 1 H W -> B H W 1
        coords = coords + self.pad[0] # Real coords are shifted by pad[0]
        batch, h1, w1, _ = coords.shape
        coords_dtype = coords.dtype

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coords.device) # 1 1 2r+1 1
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i # BHW 1 2r+1 1
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1) # BHW 1 2r+1 2
            corr = bilinear_sampler(corr, coords_lvl) # BHW 1 2r+1
            corr = corr.view(batch, h1, w1, -1) # B H W 2r+1
            corr = corr[:, :, self.pad[0]:w1-self.pad[1], :] # B H W' 2r+1
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1) # B H W' (2r+1)*num_levels
        return out.permute(0, 3, 1, 2).contiguous().to(coords_dtype)# B (2r+1)*num_levels H W'

    @staticmethod
    def corr(fmap2, fmap3):
        B, D, H, W2 = fmap2.shape
        _, _, _, W3 = fmap3.shape
        fmap2_dtype = fmap2.dtype

        fmap2 = fmap2.view(B, D, H, W2)
        fmap3 = fmap3.view(B, D, H, W3)

        # a i j k: batch, feature, height, width
        # a i j h: batch, feature, height, disparity
        # a j k h: batch, height, width, disparity

        corr = torch.einsum('aijk,aijh->ajkh', fmap2, fmap3)
        corr = corr.reshape(B, H, W2, 1, W3).contiguous()
        return (corr / torch.sqrt(torch.tensor(D))).to(fmap2_dtype)