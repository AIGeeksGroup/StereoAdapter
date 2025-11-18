import math
import torch.nn as nn

from .dinov2_dpt import DINOv2_DPT_Backbone, get_dinov2_dpt_backbone

class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        #self.w_identity = torch.eye(qkv.in_features)
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += (self.alpha // self.r) * new_q
        qkv[:, :, -self.dim :] += (self.alpha // self.r) * new_v
        return qkv

class DAv2_MeLo_Backbone(nn.Module):
    def __init__(self, dav2_model: DINOv2_DPT_Backbone, r: int, alpha: int, lora_layer=None):
        super(DAv2_MeLo_Backbone, self).__init__()

        assert r > 0
        assert alpha > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(dav2_model.pretrained.blocks)))

        # dim = dav2_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in dav2_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(dav2_model.pretrained.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha
            )
            print("MeLo layer added")

        self.reset_parameters()
        self.dav2_melo = dav2_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def freeze(self):
        """Freeze all LoRA layers by setting requires_grad = False"""
        for layer in self.w_As + self.w_Bs:
            for param in layer.parameters():
                param.requires_grad = False
        print("All LoRA layers frozen")

    def forward(self, x):
        return self.dav2_melo(x)


def get_dav2_melo_backbone(backbone_name='dinov2b', pretrained=True, r=4, alpha=4):
    
    dav2_model, enc_ch_num = get_dinov2_dpt_backbone(backbone_name=backbone_name.replace("melo", ""), pretrained=pretrained)
    model = DAv2_MeLo_Backbone(dav2_model=dav2_model, r=r, alpha=alpha)
    
    return model, enc_ch_num
