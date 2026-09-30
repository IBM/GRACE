# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class ConformerPooler(nn.Module):
    def __init__(self, pooling: str, encoder_dim: int):
        super().__init__()
        assert pooling in ("uniform", "boltzmann", "learned")
        self.pooling = pooling
        if pooling == "learned":
            self.attn = nn.Linear(encoder_dim, 1, bias=False)

    def forward(
        self,
        cls_embs: torch.Tensor,
        boltzmann_weights: torch.Tensor,
        conformer_mask: torch.Tensor,
    ) -> torch.Tensor:

        if self.pooling == "uniform":
            mask_f = conformer_mask.float().unsqueeze(-1)
            denom  = mask_f.sum(1).clamp(min=1.0)
            return (cls_embs * mask_f).sum(1) / denom

        elif self.pooling == "boltzmann":
            w = boltzmann_weights.unsqueeze(-1)
            return (cls_embs * w).sum(1)

        else:
            scores = self.attn(cls_embs).squeeze(-1)
            scores = scores.masked_fill(~conformer_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (cls_embs * weights).sum(1)


class GasteigerAtomPooler(nn.Module):
    _SIGN_VEC = [+1.0, -1.0, +1.0]

    def __init__(self, encoder_dim: int):
        super().__init__()
        self.score_proj = nn.Linear(encoder_dim, 1, bias=False)
        self.lam = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        atom_embs: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor,
        adduct_onehot: torch.Tensor = None,
    ) -> torch.Tensor:
        scores = self.score_proj(atom_embs).squeeze(-1)
        if adduct_onehot is not None:
            sign_vec = atom_embs.new_tensor(self._SIGN_VEC)
            N = atom_embs.shape[0]
            B = adduct_onehot.shape[0]
            if B != N:
                K = N // B
                adduct_onehot = adduct_onehot.unsqueeze(1).expand(B, K, 3).reshape(N, 3)
            adduct_sign = adduct_onehot @ sign_vec
            charge_bias = adduct_sign.unsqueeze(1) * (-charges)
        else:
            charge_bias = -charges
        scores = scores + self.lam * charge_bias
        scores = scores.masked_fill(~atom_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * atom_embs).sum(1)


class CCS3D(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "single",
        gasteiger_atom: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        self.gasteiger_atom = gasteiger_atom

        if pooling != "single":
            self.pooler = ConformerPooler(pooling, encoder_dim)
        if gasteiger_atom:
            self.atom_pooler = GasteigerAtomPooler(encoder_dim)

        input_dim = 2 * encoder_dim if gasteiger_atom else encoder_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cls_emb: torch.Tensor,
        boltzmann_weights: torch.Tensor = None,
        conformer_mask: torch.Tensor = None,
        atom_embs: torch.Tensor = None,
        atom_mask: torch.Tensor = None,
        charges: torch.Tensor = None,
        adduct_onehot: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.gasteiger_atom:
            h_attn = self.atom_pooler(atom_embs, charges, atom_mask, adduct_onehot)
            if self.pooling != "single":
                B, K, D = cls_emb.shape
                h_attn = h_attn.view(B, K, D)
                cls_pooled = self.pooler(cls_emb, boltzmann_weights, conformer_mask)
                h_pooled   = self.pooler(h_attn,  boltzmann_weights, conformer_mask)
            else:
                cls_pooled = cls_emb
                h_pooled   = h_attn
            x = torch.cat([cls_pooled, h_pooled], dim=-1)
        elif self.pooling != "single":
            x = self.pooler(cls_emb, boltzmann_weights, conformer_mask)
        else:
            x = cls_emb
        return self.net(x).squeeze(-1)
