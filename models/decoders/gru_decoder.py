import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from utils.gru_utils import bilinear_sampler

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

class GRUDecoder(nn.Module):
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
        
        #self.cnet = MultiBasicEncoder()
        self.cnet_net = nn.ModuleList()
        self.cnet_inp = nn.ModuleList()
        self.context_zqr_convs = nn.ModuleList()

        for dim in self.context_dims:
            # Create cnet_net module
            cnet_net_module = nn.Sequential(
                nn.Conv2d(1, dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1)
            )
            self.cnet_net.append(cnet_net_module)
            
            # Create cnet_inp module  
            cnet_inp_module = nn.Sequential(
                nn.Conv2d(1, dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1)
            )
            self.cnet_inp.append(cnet_inp_module)
            
            # Create context_zqr_convs module
            context_zqr_conv_module = nn.Conv2d(dim, dim * 3, 3, padding=1)
            self.context_zqr_convs.append(context_zqr_conv_module)

        # RAFT 风格的多层 GRU 更新块（与 StereoAnywhere 一致的调用接口）
        raft_args = SimpleNamespace(corr_radius = corr_radius,
                                    corr_levels = corr_levels,
                                    n_gru_layers = n_gru_layers,
                                    encoder_output_dim = context_dims[-1],
                                    n_downsample = 2
                                    )
        self.update_block = BasicMultiUpdateBlock(
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
        device = img2.device
        B, _, H, W = img2.shape
        (Hc, Wc), (fh, fw) = self._pick_stride_and_factor(fmap2, img2)
        assert fh == fw, "非等比下采样暂不支持"
        up_factor = fh

        corr_block_cls = CorrBlockFast1D if self.corr_implementation == "reg_cuda" else CorrBlock1D

        # CorrBlock1D.corr: (fmap2, fmap3) -> [B, 1, Hc, Wc, Wc]
        stereo_corr_volume = corr_block_cls.corr(fmap2.float(), fmap3.float()).squeeze(3).unsqueeze(1)
        
        stereo_corr_fn = corr_block_cls(
            stereo_corr_volume.squeeze(1).unsqueeze(3),
            radius=self.corr_radius,
            num_levels=self.corr_levels
        )

        assert 'mono_disp_0_s' in outputs, "需要 Mono 分支先产生 mono_disp_0_s 供 stereo 使用"
        mde2 = outputs['mono_disp_0_s']  # [B,1,H,W] （视差表征）
        mde2_low = F.interpolate(mde2, size=(Hc, Wc), mode="bilinear", align_corners=True) / float(up_factor)
        
        net_list = [torch.tanh(x) for x in cnet_list]
        inp_list = [torch.relu(x) for x in cnet_list] 

        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        coords0, coords1 = self._initialize_flow(B, Hc, Wc, device=device)
        if use_scaled_mde_init:
            init_flow = torch.zeros(B, 2, Hc, Wc, device=device)
            init_flow[:, 0:1] = mde2_low
            coords1 = coords1 + init_flow

        disp_predictions = []
        for itr in range(iters):

            coords1 = coords1.detach()
            stereo_corr = stereo_corr_fn(coords1)
            flow_lr = coords1 - coords0

            net_list, mask_up, delta_flow = self.update_block(
                net_list, inp_list, stereo_corr, None, flow_lr,
                iter32=(self.n_gru_layers >= 3),
                iter16=(self.n_gru_layers >= 2),
                iter08=True
            )

            delta_flow[:, 1] = 0.0
            coords1 = coords1 + delta_flow

            disp_lr = (coords1 - coords0)[:, :1]

            flow_up = self._convex_upflow(disp_lr, mask_up, up_factor=up_factor, out_hw=(H, W))

            disp_predictions.append(flow_up)
            
        disp = disp_predictions[-1]
        outputs['stereo_disp_0_s'] = disp
        outputs['stereo_depth_0_s'] = d2d / disp
        if is_train:
            outputs['stereo_flow_preds'] = disp_predictions

        return outputs
    
    @staticmethod
    def _coords_grid(b, h, w, device):
        y, x = torch.meshgrid(
            torch.linspace(0, h - 1, h, device=device),
            torch.linspace(0, w - 1, w, device=device),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)  # [B, 2, H, W]
        return grid
    
    def _initialize_flow(self, b, h, w, device):
        coords0 = self._coords_grid(b, h, w, device)
        coords1 = coords0.clone()
        return coords0, coords1
    
    @staticmethod
    @torch.no_grad()
    def _pick_stride_and_factor(fmap, img):
        B, _, H, W = img.shape
        _, _, Hc, Wc = fmap.shape
        assert H % Hc == 0 and W % Wc == 0, "Feature alignment error"
        return (Hc, Wc), (H // Hc, W // Wc)
    
    @staticmethod
    def _convex_upflow(flow, mask, up_factor=None, out_hw=None):
        B, C, Hc, Wc = flow.shape
        if up_factor is None:
            assert out_hw is not None, "The upsampling factor needs to be inferred from out_hw"
            H, W = out_hw
            fh, fw = H // Hc, W // Wc
            assert fh == fw, "Non-uniform downsampling is not currently supported (due to inconsistent H/W ratios)"
            up_factor = fh

        # [B, 1, 9, f, f, Hc, Wc]
        mask = mask.view(B, 1, 9, up_factor, up_factor, Hc, Wc)
        mask = torch.softmax(mask, dim=2)

        # [B, 1*9, Hc*Wc] -> [B, 1, 9, Hc, Wc]
        up_flow = F.unfold(flow, [3, 3], padding=1)  # [B, 9, Hc*Wc]
        up_flow = up_flow.view(B, 1, 9, Hc, Wc)

        # [B, 1, f, f, Hc, Wc]
        up_flow = torch.sum(mask * up_flow.unsqueeze(3).unsqueeze(3), dim=2)
        # [B, 1, H, W]
        up_flow = up_flow.permute(0, 1, 4, 5, 2, 3).reshape(B, 1, Hc * up_factor, Wc * up_factor)

        return up_flow * up_factor
    
    def _build_stereo_context_from_mde(self, mde_low):
        net_list, inp_list = [], []
        for i in range(self.n_gru_layers):
            net_i = torch.tanh(self.cnet_net[i](mde_low))
            inp_i = F.relu(self.cnet_inp[i](mde_low), inplace=True)
            zqr = self.context_zqr_convs[i](inp_i)
            z, q, r = torch.chunk(zqr, 3, dim=1)
            net_list.append(net_i)
            inp_list.append([z, q, r])
            
        return net_list, inp_list

class UpdateHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(UpdateHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SigmoidUpdateHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(SigmoidUpdateHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return F.sigmoid(self.conv2(self.relu(self.conv1(x))))
    
class ScaleShiftUpdateHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(ScaleShiftUpdateHead, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.scaler = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),            
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        backbone = self.conv2(self.relu(self.conv1(x)))
        return self.scaler(backbone).reshape(B, self.output_dim, 1, 1)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)

        h = (1-z) * h + z * q
        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.corr_levels * (2*args.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicConfidenceAwareMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicConfidenceAwareMotionEncoder, self).__init__()
        self.args = args
        self.corr_levels = args.corr_levels
        self.corr_radius = args.corr_radius
        self.encoder_output_dim = args.encoder_output_dim

        cor_planes = self.corr_levels * (2*self.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convcf1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convcf2 = nn.Conv2d(64, 64, 3, padding=1)
        self._conv_with_conf = nn.Conv2d(64+64+64+64, 128-3, 3, padding=1)

    def forward(self, flow, flow_conf, corr, corr_mono):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1(corr_mono))
        cor_mono = F.relu(self.convc2(cor_mono))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        flo_conf = F.relu(self.convcf1(flow_conf))
        flo_conf = F.relu(self.convcf2(flo_conf))

        cor_flo = torch.cat([cor, cor_mono, flo, flo_conf], dim=1)
        out = F.relu(self._conv_with_conf(cor_flo))
        return torch.cat([out, flow, flow_conf], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[], up_factor=None):
        super().__init__()
        self.args = args

        encoder_output_dim = args.encoder_output_dim
        self.n_gru_layers = args.n_gru_layers
        self.n_downsample = args.n_downsample

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (self.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        
        self.encoder = BasicMotionEncoder(args)
        self.flow_head = UpdateHead(hidden_dims[2], hidden_dim=256, output_dim=2)

        factor = 2**self.n_downsample if up_factor is None else up_factor
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, corr=None, corr_mono=None, flow=None, flow_conf=None, iter08=True, iter16=True, iter32=True, iter64=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, corr)

            if self.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)
        
        
        if not update:
            return net

        delta_flow = self.flow_head(net[0])

        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])

        return net, mask, delta_flow

class BasicMultiUpdateScalerBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicConfidenceAwareMotionEncoder(args)
        encoder_output_dim = 128
        self.n_gru_layers = 3
        self.n_downsample = 2

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (self.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])

        self.lscale_head = SigmoidUpdateHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        self.conf_head = SigmoidUpdateHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        self.gscale_gshift_head = ScaleShiftUpdateHead(hidden_dims[2], hidden_dim=256, output_dim=2)

    def forward(self, net, inp, corr=None, flow=None, flow_conf=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, flow_conf, corr)
            if self.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_gscale_gshift = self.gscale_gshift_head(net[0])
        delta_gscale, delta_gshift = delta_gscale_gshift[:,0:1], delta_gscale_gshift[:,1:2]
        delta_confidence = self.conf_head(net[0])
        delta_lscale = self.lscale_head(net[0])

        return net, delta_lscale, delta_gscale, delta_gshift, delta_confidence

class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
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

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[[128]*3, [128]*3], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        y = self.layer4(x)
        outputs16 = [f(y) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        z = self.layer5(y)
        outputs32 = [f(z) for f in self.outputs32]

        return (outputs08, outputs16, outputs32, v) if dual_inp else (outputs08, outputs16, outputs32)
