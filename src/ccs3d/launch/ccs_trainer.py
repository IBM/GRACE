# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn
import lightning as L

from ccs3d.models import CCS3D, apply_adduct_cls_lora
from ccs3d.utils.metrics import compute_metrics


class CCS3DTrainer(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        head: CCS3D,
        lr_head: float = 1e-4,
        lr_encoder: float = 1e-5,
        freeze_epochs: int = 10,
        weight_decay: float = 1e-2,
        num_conformers: int = 1,
        gasteiger_atom: bool = False,
        lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        scheduler: str = "cosine",
        max_epochs: int = 100,
        lora_warmup_epochs: int = 10,
        residual_target: bool = False,
        multi_adduct_loss: bool = False,
        mal_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "head"])

        self.encoder = encoder
        self.head = head
        self.loss_fn = nn.MSELoss()

        self.adduct_cls_emb = apply_adduct_cls_lora(
            encoder, rank=lora_rank, alpha=lora_alpha, lora=lora
        )

        if residual_target:
            nn.init.zeros_(self.head.net[-1].weight)
            nn.init.zeros_(self.head.net[-1].bias)


    def _lora_params(self):
        if not self.hparams.lora:
            raise RuntimeError("_lora_params() called but lora=False")
        params = []
        for layer in self.encoder.encoder.layers:
            in_proj = layer.self_attn.in_proj
            params += [in_proj.q_A, in_proj.q_B, in_proj.v_A, in_proj.v_B]
        return params


    def _encode(self, net_inputs: dict,
                adduct_idx: torch.Tensor) -> torch.Tensor:
        self.adduct_cls_emb.set_adduct(adduct_idx)
        try:
            repr_out = self.encoder(
                net_inputs["src_tokens"],
                net_inputs["src_distance"],
                net_inputs["src_coord"],
                net_inputs["src_edge_type"],
                return_repr=True,
                return_atomic_reprs=True,
            )
        finally:
            self.adduct_cls_emb.clear_adduct()
        return repr_out["cls_repr"]

    def _encode_with_atoms(self, net_inputs: dict,
                           adduct_idx: torch.Tensor):
        self.adduct_cls_emb.set_adduct(adduct_idx)
        try:
            repr_out = self.encoder(
                net_inputs["src_tokens"],
                net_inputs["src_distance"],
                net_inputs["src_coord"],
                net_inputs["src_edge_type"],
                return_repr=True,
                return_atomic_reprs=True,
            )
        finally:
            self.adduct_cls_emb.clear_adduct()
        cls_emb = repr_out["cls_repr"]
        atomic_reprs = repr_out["atomic_reprs"]

        D = cls_emb.shape[-1]
        L_max = max(r.shape[0] for r in atomic_reprs)
        N = len(atomic_reprs)
        atom_embs = cls_emb.new_zeros(N, L_max, D)
        atom_mask = torch.zeros(N, L_max, dtype=torch.bool, device=cls_emb.device)
        for i, r in enumerate(atomic_reprs):
            L_i = r.shape[0]
            atom_embs[i, :L_i] = r
            atom_mask[i, :L_i] = True

        return cls_emb, atom_embs, atom_mask

    def _tile_charges(self, charges: torch.Tensor, L_emb: int, B: int, K: int):
        L_charges = charges.shape[1]
        charges_bk = charges.unsqueeze(1).expand(B, K, L_charges).reshape(B * K, L_charges)
        if L_charges < L_emb:
            pad = charges_bk.new_zeros(B * K, L_emb - L_charges)
            charges_bk = torch.cat([charges_bk, pad], dim=1)
        elif L_charges > L_emb:
            charges_bk = charges_bk[:, :L_emb]
        return charges_bk

    def forward(
        self,
        net_inputs: dict,
        adduct_onehot: torch.Tensor,
        boltzmann_weights: torch.Tensor = None,
        conformer_mask: torch.Tensor = None,
        charges: torch.Tensor = None,
    ) -> torch.Tensor:
        B = adduct_onehot.shape[0]
        K = self.hparams.num_conformers

        adduct_idx = adduct_onehot.argmax(dim=-1)
        if K > 1:
            adduct_idx_n = adduct_idx.unsqueeze(1).expand(B, K).reshape(B * K)
        else:
            adduct_idx_n = adduct_idx

        if self.hparams.gasteiger_atom:
            cls_emb, atom_embs, atom_mask = self._encode_with_atoms(
                net_inputs, adduct_idx=adduct_idx_n
            )
            L_emb = atom_embs.shape[1]
            if K > 1:
                cls_emb = cls_emb.view(B, K, -1)
                charges_bk = self._tile_charges(charges, L_emb, B, K)
            else:
                charges_bk = charges[:, :L_emb] if charges.shape[1] >= L_emb else \
                    torch.cat([charges, charges.new_zeros(B, L_emb - charges.shape[1])], dim=1)
            return self.head(cls_emb, boltzmann_weights, conformer_mask,
                             atom_embs=atom_embs, atom_mask=atom_mask, charges=charges_bk,
                             adduct_onehot=adduct_onehot)

        cls_emb = self._encode(net_inputs, adduct_idx=adduct_idx_n)
        if K > 1:
            cls_emb = cls_emb.view(B, K, -1)
            return self.head(cls_emb, boltzmann_weights, conformer_mask)
        return self.head(cls_emb)


    def _unpack_batch(self, batch):
        if self.hparams.residual_target:
            *batch_head, ridge_preds = batch
            batch = tuple(batch_head)
        else:
            ridge_preds = None

        if self.hparams.gasteiger_atom:
            if self.hparams.num_conformers > 1:
                net_inputs, bw, mask, adduct_onehot, labels, charges = batch
                return net_inputs, adduct_onehot, labels, bw, mask, charges, ridge_preds, None
            else:
                net_inputs, adduct_onehot, labels, charges, mol_ids = batch
                return net_inputs, adduct_onehot, labels, None, None, charges, ridge_preds, mol_ids
        elif self.hparams.num_conformers > 1:
            net_inputs, bw, mask, adduct_onehot, labels = batch
            return net_inputs, adduct_onehot, labels, bw, mask, None, ridge_preds, None
        else:
            net_inputs, adduct_onehot, labels, mol_ids = batch
            return net_inputs, adduct_onehot, labels, None, None, None, ridge_preds, mol_ids

    def _shared_step(self, batch):
        net_inputs, adduct_onehot, labels, bw, mask, charges, ridge_preds, mol_ids = \
            self._unpack_batch(batch)
        preds = self(net_inputs, adduct_onehot, bw, mask, charges)
        if ridge_preds is not None:
            ridge_preds = ridge_preds.to(preds.device)
            residuals = labels.to(preds.device) - ridge_preds
            loss = self.loss_fn(preds, residuals)
            full_preds = preds.detach() + ridge_preds
            train_targets = residuals
        else:
            labels_dev = labels.to(preds.device)
            loss = self.loss_fn(preds, labels_dev)
            full_preds = preds.detach()
            train_targets = labels_dev
        rmse = torch.sqrt(loss)
        return loss, rmse, full_preds, labels, preds, train_targets, mol_ids

    def on_train_epoch_start(self):
        if self.hparams.freeze_epochs > 0 and self.current_epoch == self.hparams.freeze_epochs:
            if self.hparams.lora:
                if self.hparams.scheduler == "cosine":
                    import warnings
                    warnings.warn(
                        "freeze_epochs > 0 with lora=True and scheduler='cosine': "
                        "the dynamically added LoRA param group will not be covered by "
                        "CosineAnnealingLR and will run at a fixed lr=lr_encoder. "
                        "Use freeze_epochs=0 to get proper per-group LR schedules.",
                        UserWarning,
                    )
                self.trainer.optimizers[0].add_param_group({
                    "params": list(self.adduct_cls_emb.parameters()) + self._lora_params(),
                    "lr": self.hparams.lr_encoder,
                    "weight_decay": self.hparams.weight_decay,
                })
            self.log("train/stage", 2.0, prog_bar=True)
        elif self.current_epoch == 0 and self.hparams.freeze_epochs > 0:
            self.log("train/stage", 1.0, prog_bar=True)
        self._train_preds = []
        self._train_targets = []

    def _compute_mal(self, preds, targets, mol_ids):
        if mol_ids is None or (mol_ids < 0).all():
            return None
        B = preds.shape[0]
        eq = mol_ids.unsqueeze(0) == mol_ids.unsqueeze(1)
        upper = torch.triu(torch.ones(B, B, dtype=torch.bool,
                                      device=preds.device), diagonal=1)
        mask = eq & upper
        if not mask.any():
            return None
        i_idx, j_idx = mask.nonzero(as_tuple=True)
        pred_diff  = preds[i_idx]   - preds[j_idx]
        label_diff = targets[i_idx] - targets[j_idx]
        return ((pred_diff - label_diff) ** 2).mean()

    def training_step(self, batch, batch_idx):
        loss, rmse, full_preds, labels, preds_raw, train_targets, mol_ids = \
            self._shared_step(batch)

        if self.hparams.multi_adduct_loss:
            mal = self._compute_mal(preds_raw, train_targets, mol_ids)
            if mal is not None:
                self.log("train/mal", mal.detach(), on_step=True, on_epoch=True,
                         prog_bar=False)
                loss = loss + self.hparams.mal_weight * mal

        self._train_preds.append(full_preds.detach().cpu())
        self._train_targets.append(labels.detach().cpu())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/rmse", rmse, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        preds   = torch.cat(self._train_preds).float().numpy()
        targets = torch.cat(self._train_targets).numpy()
        m = compute_metrics(targets, preds)
        self.log("train/rmse_epoch",  m["RMSE"],        prog_bar=False)
        self.log("train/pct_diff",    m["MeanPctDiff"],  prog_bar=False)
        self.log("train/pearson_r",   m["PearsonR"],     prog_bar=False)

    def on_validation_epoch_start(self):
        self._val_preds = []
        self._val_targets = []

    def validation_step(self, batch, batch_idx):
        loss, _rmse, full_preds, labels, *_ = self._shared_step(batch)
        self._val_preds.append(full_preds.detach().cpu())
        self._val_targets.append(labels.detach().cpu())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        preds   = torch.cat(self._val_preds).float().numpy()
        targets = torch.cat(self._val_targets).numpy()
        m = compute_metrics(targets, preds)
        self.log("val/rmse",      m["RMSE"],       prog_bar=True)
        self.log("val/pct_diff",  m["MeanPctDiff"], prog_bar=False)
        self.log("val/pearson_r", m["PearsonR"],    prog_bar=True)
        self.log("val/spearman",  m["SpearmanR"],   prog_bar=False)
        self.log("val/kendall",   m["KendallTau"],  prog_bar=False)
        for i, pg in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"lr/group{i}", pg["lr"], prog_bar=False)


    def configure_optimizers(self):
        head_params  = list(self.head.parameters())
        delta_params = list(self.adduct_cls_emb.parameters())
        if self.hparams.freeze_epochs == 0 and self.hparams.lora:
            param_groups = [
                {"params": head_params,                              "lr": self.hparams.lr_head},
                {"params": delta_params + self._lora_params(),       "lr": self.hparams.lr_encoder},
            ]
        else:
            param_groups = [
                {"params": head_params + delta_params, "lr": self.hparams.lr_head},
            ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler == "cosine":
            T = self.hparams.max_epochs
            W = self.hparams.lora_warmup_epochs

            if self.hparams.lora and self.hparams.freeze_epochs == 0:
                def head_lambda(epoch):
                    return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * epoch / T)))

                def lora_lambda(epoch):
                    if epoch < W:
                        return epoch / max(1, W)
                    progress = (epoch - W) / max(1, T - W)
                    return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * progress)))

                sched = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=[head_lambda, lora_lambda]
                )
            else:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T, eta_min=1e-7,
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1},
            }
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "val/rmse",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
