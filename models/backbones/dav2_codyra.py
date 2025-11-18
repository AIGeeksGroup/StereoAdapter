import math
import torch.nn as nn
import torch

from .dinov2_dpt import DINOv2_DPT_Backbone, get_dinov2_dpt_backbone

class _CoDyRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Parameter,
        linear_b_q: nn.Parameter,
        linear_a_k: nn.Parameter,
        linear_b_k: nn.Parameter,
        linear_a_v: nn.Parameter,
        linear_b_v: nn.Parameter,
        i_w_q: nn.Parameter,
        i_w_k: nn.Parameter,
        i_w_v: nn.Parameter,
        max_kappa: float = 0.005,
        lambda_reg: float = 1e-4,
        num_epochs: int = 20,
        dense_ratio: float = 0.5,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.i_w_q = i_w_q
        self.i_w_k = i_w_k
        self.i_w_v = i_w_v
        self.dim = qkv.in_features
        self.lambda_reg = lambda_reg
        self.max_kappa = max_kappa
        self.dense_ratio = dense_ratio
        self.current_sparse_epoch = 0
        self.dense_epoch = int(num_epochs * dense_ratio)
        self.sparse_epochs = num_epochs - self.dense_epoch - 1
        self.is_frozen = False

    def reset(self, num_epochs):
        self.current_sparse_epoch = 0
        self.dense_epoch = int(num_epochs * self.dense_ratio)
        self.sparse_epochs = num_epochs - self.dense_epoch - 1
        self.is_frozen = False

    def next_epoch(self):
        if self.dense_epoch > 0:
            self.dense_epoch -= 1
            print(f"Dense epoch remains: {self.dense_epoch}")
        else:
            self.current_sparse_epoch += 1
            print(f"Sparse epoch {self.current_sparse_epoch} / {self.sparse_epochs}")

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    def _codyra_result(self, linear_a: nn.Parameter, linear_b: nn.Parameter, i_w: nn.Parameter, x):
        x = x @ linear_a.T
        x *= i_w
        x = x @ linear_b.T
        return x

    def compute_sparsity_loss(self):
        if self.is_frozen:
            return 0

        return self.lambda_reg * (
            torch.norm(self.i_w_q, p=1)
            + torch.norm(self.i_w_k, p=1)
            + torch.norm(self.i_w_v, p=1)
        )

    def _update_i_w(self, kappa: float, i_w: nn.Parameter):
        if i_w.grad is None:
            return

        with torch.no_grad():
            signs = torch.sign(i_w)
            w_updated = torch.where(
                torch.abs(i_w) > kappa, 
                i_w + signs * kappa, 
                torch.zeros_like(i_w)
            )
            i_w.copy_(w_updated)

    def update_iws(self,):
        if self.dense_epoch > 0:
            return

        kappa = self.max_kappa

        if self.sparse_epochs > 0:
            kappa *= min(self.current_sparse_epoch / self.sparse_epochs, 1)

        self._update_i_w(kappa, self.i_w_q)
        self._update_i_w(kappa, self.i_w_k)
        self._update_i_w(kappa, self.i_w_v)

    def get_active_ranks(self):
        active_ranks = []
        active_ranks.append(sum(self.i_w_q != 0))
        active_ranks.append(sum(self.i_w_k != 0))
        active_ranks.append(sum(self.i_w_v != 0))
        return active_ranks

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        if self.is_frozen:
            return qkv

        new_q = self._codyra_result(self.linear_a_q, self.linear_b_q, self.i_w_q, x)
        new_k = self._codyra_result(self.linear_a_k, self.linear_b_k, self.i_w_k, x)
        new_v = self._codyra_result(self.linear_a_v, self.linear_b_v, self.i_w_v, x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, self.dim : -self.dim] += new_k
        qkv[:, :, -self.dim :] += new_v
        return qkv

    def _get_delta_w(self, linear_a: nn.Parameter, linear_b: nn.Parameter, i_w: nn.Parameter):
        with torch.no_grad():
            mask = i_w != 0
            active_w_a = linear_a[mask]
            active_w_b = linear_b[:, mask]
            active_i_w = i_w[mask]

            weighted_b = active_w_b * active_i_w.unsqueeze(0)
            delta_w = weighted_b @ active_w_a
            return delta_w

    def merge_weights(self):
        with torch.no_grad():
            delta_w_q = self._get_delta_w(self.linear_a_q, self.linear_b_q, self.i_w_q)
            delta_w_k = self._get_delta_w(self.linear_a_k, self.linear_b_k, self.i_w_k)
            delta_w_v = self._get_delta_w(self.linear_a_v, self.linear_b_v, self.i_w_v)

            original_weight = self.qkv.weight
            original_weight[: self.dim, :] += delta_w_q
            original_weight[self.dim : -self.dim, :] += delta_w_k  # K section
            original_weight[-self.dim :, :] += delta_w_v

            self.is_frozen = True

class _CoDyRA_linear(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Module,
        linear_a: nn.Parameter,
        linear_b: nn.Parameter,
        i_w: nn.Parameter,
        max_kappa: float = 0.005,
        lambda_reg: float = 1e-4,
        num_epochs: int = 20,
        dense_ratio: float = 0.5,
    ):
        super().__init__()
        self.linear_layer = linear_layer
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.i_w = i_w

        self.lambda_reg = lambda_reg
        self.max_kappa = max_kappa
        self.current_sparse_epoch = 0
        self.dense_ratio = dense_ratio
        self.dense_epoch = int(num_epochs * dense_ratio)
        self.sparse_epochs = num_epochs - self.dense_epoch - 1
        self.is_frozen = False

    def reset(self, num_epochs):
        self.current_sparse_epoch = 0
        self.dense_epoch = int(num_epochs * self.dense_ratio)
        self.sparse_epochs = num_epochs - self.dense_epoch - 1
        self.is_frozen = False

    def next_epoch(self):
        if self.dense_epoch > 0:
            self.dense_epoch -= 1
            print(f"Dense epoch remains: {self.dense_epoch}")
        else:
            self.current_sparse_epoch += 1
            print(f"Sparse epoch {self.current_sparse_epoch} / {self.sparse_epochs}")

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    def compute_sparsity_loss(self):
        if self.is_frozen:
            return 0

        return self.lambda_reg * torch.norm(self.i_w, p=1)

    def update_iw(self):
        if self.dense_epoch > 0:
            return

        if self.i_w.grad is None:
            return

        kappa = self.max_kappa
        if self.sparse_epochs > 0:
            kappa *= min(self.current_sparse_epoch / self.sparse_epochs, 1)

        with torch.no_grad():
            signs = torch.sign(self.i_w)
            w_updated = torch.where(
                torch.abs(self.i_w) > kappa, 
                self.i_w + signs * kappa, 
                torch.zeros_like(self.i_w)
            )
            self.i_w.copy_(w_updated)

    def get_active_ranks(self):
        return sum(self.i_w != 0)

    def forward(self, x):
        linear_output = self.linear_layer(x)
        if self.is_frozen:
            return linear_output

        codyra_output = x @ self.linear_a.T
        codyra_output *= self.i_w
        codyra_output = codyra_output @ self.linear_b.T
        return linear_output + codyra_output

    def merge_weights(self):
        with torch.no_grad():
            mask = self.i_w != 0
            active_w_a = self.linear_a[mask]
            active_w_b = self.linear_b[:, mask]
            active_i_w = self.i_w[mask]

            weighted_b = active_w_b * active_i_w.unsqueeze(0)
            delta_w = weighted_b @ active_w_a
            self.linear_layer.weight += delta_w

            self.is_frozen = True


class DAv2_CoDyRA_Backbone(nn.Module):
    def __init__(
        self,
        dav2_model: DINOv2_DPT_Backbone,
        r: int,
        lora_layer=None,
        max_kappa: float = 0.005,
        lambda_reg: float = 1e-4,
        num_epochs: int = 20,
        dense_ratio: float = 0.5,
    ):
        super(DAv2_CoDyRA_Backbone, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(dav2_model.pretrained.blocks)))

        self.max_kappa = max_kappa
        # dim = dav2_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = nn.ParameterList()
        self.w_Bs = nn.ParameterList()
        self.i_ws = nn.ParameterList()

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
            w_a_linear_q = nn.Parameter(torch.zeros(r, self.dim))
            w_b_linear_q = nn.Parameter(torch.zeros(self.dim, r))
            w_a_linear_k = nn.Parameter(torch.zeros(r, self.dim))
            w_b_linear_k = nn.Parameter(torch.zeros(self.dim, r))
            w_a_linear_v = nn.Parameter(torch.zeros(r, self.dim))
            w_b_linear_v = nn.Parameter(torch.zeros(self.dim, r))
            i_w_q = nn.Parameter(torch.ones(r))
            i_w_k = nn.Parameter(torch.ones(r))
            i_w_v = nn.Parameter(torch.ones(r))
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_k)
            self.w_Bs.append(w_b_linear_k)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.i_ws.append(i_w_q)
            self.i_ws.append(i_w_k)
            self.i_ws.append(i_w_v)
            blk.attn.qkv = _CoDyRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_k,
                w_b_linear_k,
                w_a_linear_v,
                w_b_linear_v,
                i_w_q,
                i_w_k,
                i_w_v,
                max_kappa,
                lambda_reg,
                num_epochs,
                dense_ratio
            )

            w_attn_proj = blk.attn.proj
            w_a_attn_proj = nn.Parameter(torch.zeros(r, w_attn_proj.in_features))
            w_b_attn_proj = nn.Parameter(torch.zeros(w_attn_proj.out_features, r))
            i_w_attn_proj = nn.Parameter(torch.ones(r))
            self.w_As.append(w_a_attn_proj)
            self.w_Bs.append(w_b_attn_proj)
            self.i_ws.append(i_w_attn_proj)
            blk.attn.proj = _CoDyRA_linear(
                w_attn_proj,
                w_a_attn_proj,
                w_b_attn_proj,
                i_w_attn_proj,
                max_kappa,
                lambda_reg,
                num_epochs,
                dense_ratio,
            )

            print("Attn CoDyRA layer added")

            w_fc1_linear = blk.mlp.fc1
            w_a_linear_fc1 = nn.Parameter(torch.zeros(r, w_fc1_linear.in_features))
            w_b_linear_fc1 = nn.Parameter(torch.zeros(w_fc1_linear.out_features, r))
            i_w_fc1 = nn.Parameter(torch.ones(r))
            self.w_As.append(w_a_linear_fc1)
            self.w_Bs.append(w_b_linear_fc1)
            self.i_ws.append(i_w_fc1)
            blk.mlp.fc1 = _CoDyRA_linear(
                w_fc1_linear,
                w_a_linear_fc1,
                w_b_linear_fc1,
                i_w_fc1,
                max_kappa,
                lambda_reg,
                num_epochs,
                dense_ratio,
            )

            w_fc2_linear = blk.mlp.fc2
            w_a_linear_fc2 = nn.Parameter(torch.zeros(r, w_fc2_linear.in_features))
            w_b_linear_fc2 = nn.Parameter(torch.zeros(w_fc2_linear.out_features, r))
            i_w_fc2 = nn.Parameter(torch.ones(r))
            self.w_As.append(w_a_linear_fc2)
            self.w_Bs.append(w_b_linear_fc2)
            self.i_ws.append(i_w_fc2)
            blk.mlp.fc2 = _CoDyRA_linear(
                w_fc2_linear,
                w_a_linear_fc2,
                w_b_linear_fc2,
                i_w_fc2,
                max_kappa,
                lambda_reg,
                num_epochs,
                dense_ratio,
            )

            print("MLP CoDyRA layer added")

        self.reset_parameters()
        self.dav2_codyra = dav2_model
        self.is_frozen = False

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B)
        for i_w in self.i_ws:
            nn.init.uniform_(i_w, -self.max_kappa * 10, self.max_kappa * 10)

    def freeze_codyra(self):
        for w_A in self.w_As:
            w_A.requires_grad = False

        for w_B in self.w_Bs:
            w_B.requires_grad = False

        for i_w in self.i_ws:
            i_w.requires_grad = False

        for t_layer_i, blk in enumerate(self.dav2_codyra.pretrained.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            blk.attn.qkv.freeze()
            blk.attn.proj.freeze()
            blk.mlp.fc1.freeze()
            blk.mlp.fc2.freeze()

        self.is_frozen = True
        print("CoDyRA layers frozen")

    def unfreeze_codyra(self):
        for w_A in self.w_As:
            w_A.requires_grad = True

        for w_B in self.w_Bs:
            w_B.requires_grad = True

        for i_w in self.i_ws:
            i_w.requires_grad = True

        for t_layer_i, blk in enumerate(self.dav2_codyra.pretrained.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            blk.attn.qkv.unfreeze()
            blk.attn.proj.unfreeze()
            blk.mlp.fc1.unfreeze()
            blk.mlp.fc2.unfreeze()

        self.is_frozen = False
        print("CoDyRA layers unfreezed")

    def merge_weights(self):
        """Freeze all LoRA layers by setting requires_grad = False"""
        for t_layer_i, blk in enumerate(self.dav2_codyra.pretrained.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            blk.attn.qkv.merge_weights()
            blk.attn.proj.merge_weights()
            blk.mlp.fc1.merge_weights()
            blk.mlp.fc2.merge_weights()

        print("CoDyRA weights merged")

    def update_iws(self):
        for t_layer_i, blk in enumerate(self.dav2_codyra.pretrained.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            blk.attn.qkv.update_iws()
            blk.attn.proj.update_iw()
            blk.mlp.fc1.update_iw()
            blk.mlp.fc2.update_iw()

    def get_active_ranks(self):
        active_ranks = []
        for t_layer_i, blk in enumerate(self.dav2_codyra.pretrained.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            active_ranks.extend(blk.attn.qkv.get_active_ranks())
            active_ranks.append(blk.attn.proj.get_active_ranks())
            active_ranks.append(blk.mlp.fc1.get_active_ranks())
            active_ranks.append(blk.mlp.fc2.get_active_ranks())

        print(self.i_ws[0])
        print(self.w_As[0])
        print(f"Min rank numbers: {min(active_ranks)}")

    def compute_sparsity_loss(self):
        """Compute total sparsity loss for all CoDyRA layers"""
        total_loss = 0
        for t_layer_i, blk in enumerate(self.dav2_codyra.pretrained.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            total_loss += blk.attn.qkv.compute_sparsity_loss()
            total_loss += blk.attn.proj.compute_sparsity_loss()
            total_loss += blk.mlp.fc1.compute_sparsity_loss()
            total_loss += blk.mlp.fc2.compute_sparsity_loss()

        return total_loss

    def next_epoch(self):
        for t_layer_i, blk in enumerate(self.dav2_codyra.pretrained.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            blk.attn.qkv.next_epoch()
            blk.attn.proj.next_epoch()
            blk.mlp.fc1.next_epoch()
            blk.mlp.fc2.next_epoch()

    def reset(self, num_epochs):
        if num_epochs == 0:
            self.freeze_codyra()
            return

        for t_layer_i, blk in enumerate(self.dav2_codyra.pretrained.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            blk.attn.qkv.reset(num_epochs)
            blk.attn.proj.reset(num_epochs)
            blk.mlp.fc1.reset(num_epochs)
            blk.mlp.fc2.reset(num_epochs)

        self.reset_parameters()
        if self.is_frozen:
            self.unfreeze_codyra()
        print(f"CoDyRA weights reset with {num_epochs} epochs")

    def forward(self, x):
        return self.dav2_codyra(x)


def get_dav2_codyra_backbone(
    backbone_name="dinov2b",
    pretrained=True,
    r=16,
    max_kappa: float = 0.005,
    lambda_reg: float = 1e-4,
    num_epochs: int = 20,
    dense_ratio: float = 0.5,
):
    dav2_model, enc_ch_num = get_dinov2_dpt_backbone(
        backbone_name=backbone_name.replace("codyra", ""), pretrained=pretrained
    )
    model = DAv2_CoDyRA_Backbone(dav2_model=dav2_model, r=r, max_kappa=max_kappa, lambda_reg=lambda_reg, num_epochs=num_epochs, dense_ratio=dense_ratio)

    return model, enc_ch_num
