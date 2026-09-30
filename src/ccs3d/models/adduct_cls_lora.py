# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdductCLSEmbedding(nn.Module):
    N_ADDUCTS = 3

    def __init__(self, encoder_dim: int = 512):
        super().__init__()
        self.adduct_delta = nn.Embedding(self.N_ADDUCTS, encoder_dim)
        nn.init.zeros_(self.adduct_delta.weight)
        self._current_adduct_idx: torch.Tensor = None
        self._hook_registered: bool = False

    def set_adduct(self, adduct_idx: torch.Tensor) -> None:
        self._current_adduct_idx = adduct_idx

    def clear_adduct(self) -> None:
        self._current_adduct_idx = None

    def register_on(self, encoder) -> None:
        if self._hook_registered:
            return

        def _inject_hook(module, input, output):
            if self._current_adduct_idx is None:
                return output
            delta = self.adduct_delta(
                self._current_adduct_idx.to(output.device)
            )
            new_cls = output[:, 0, :] + delta
            return torch.cat([new_cls.unsqueeze(1), output[:, 1:, :]], dim=1)

        encoder.emb_layer_norm.register_forward_hook(_inject_hook)
        self._hook_registered = True


class QVLoRAInProj(nn.Module):
    def __init__(self, frozen_in_proj: nn.Linear,
                 rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.frozen = frozen_in_proj
        for p in self.frozen.parameters():
            p.requires_grad_(False)

        self.D     = frozen_in_proj.in_features
        self.rank  = rank
        self.scale = alpha / rank

        self.q_A = nn.Parameter(torch.empty(rank, self.D))
        self.q_B = nn.Parameter(torch.zeros(self.D, rank))
        self.v_A = nn.Parameter(torch.empty(rank, self.D))
        self.v_B = nn.Parameter(torch.zeros(self.D, rank))

        nn.init.kaiming_uniform_(self.q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.frozen.weight, self.frozen.bias)
        D = self.D

        q_lora = F.linear(F.linear(x, self.q_A), self.q_B) * self.scale
        v_lora = F.linear(F.linear(x, self.v_A), self.v_B) * self.scale

        return torch.cat([
            base[..., :D]    + q_lora,
            base[..., D:2*D],
            base[..., 2*D:]  + v_lora,
        ], dim=-1)


def apply_adduct_cls_lora(
    unimol_model,
    rank:  int   = 16,
    alpha: float = 32.0,
    lora:  bool  = True,
) -> AdductCLSEmbedding:
    encoder = unimol_model.encoder

    for p in encoder.parameters():
        p.requires_grad_(False)

    if lora:
        for layer in encoder.layers:
            layer.self_attn.in_proj = QVLoRAInProj(
                layer.self_attn.in_proj, rank=rank, alpha=alpha
            )

    D = encoder.emb_layer_norm.normalized_shape[0]
    adduct_cls_emb = AdductCLSEmbedding(encoder_dim=D)
    adduct_cls_emb.register_on(encoder)

    return adduct_cls_emb
